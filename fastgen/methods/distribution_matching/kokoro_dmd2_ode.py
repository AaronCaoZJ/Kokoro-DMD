# SPDX-License-Identifier: Apache-2.0
#
# KokoroDMD2ODE — KokoroDMD2 plus several variants that unroll the student ODE during training.
#
# One entry class, a few switches (fields of config.model; config_kokoro_dmd2.py reads each one from the
# upper-case environment variable of the same name):
#
#   arm       student_rollout  dmd_noise    dmd_target  dpdmd  ode_init
#   A         real_noised      fresh        clean       False  False   baseline: delegates to KokoroDMD2 unchanged
#   B         ode              fresh        clean       False  False   DMD2 + training-time ODE rollout
#   C         ode              trajectory   clean       False  False   B + Hyper-Bagel DMDO trajectory noise
#   D         ode              fresh        clean       True   False   DP-DMD
#   TDM       ode              fresh        landing     False  False   compare at this step's landing point, only inside this step's noise interval
#   ODE-init  (other switches ignored)                         True    online teacher ODE trajectories → regress the teacher endpoint (no fake score)
#
# Where each method lives (fastgen/methods/distribution_matching/):
#   kokoro_dmd2.py        A    DMD2 (+ GAN) on BAGEL packed sequences (parent class)
#   kokoro_rollout.py     B    training-time ODE rollout: student / fake-score steps
#   kokoro_dmdo.py        C    DMDO trajectory noise
#   kokoro_dpdmd.py       D    DP-DMD student step and teacher anchor
#   kokoro_tdm.py         TDM  landing-point noising, TDM losses, first-step anchor
#   kokoro_pdd.py         PDD  parallel decoding heads, step and sampler
#   kokoro_ode_init.py    ODE-init teacher-trajectory regression
#   kokoro_ode_common.py  shared rollout / time-schedule / VSD helpers, notation and numerical conventions
# This file keeps the switch validation and the dispatch.
#
# GAN: not supported on the new paths (all arms here run with GAN off); GAN is only available in arm A.
#   See _validate_switches.

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from fastgen.methods.distribution_matching.kokoro_dmd2 import KokoroDMD2
from fastgen.methods.distribution_matching.kokoro_dmdo import DMDOMixin
from fastgen.methods.distribution_matching.kokoro_dpdmd import _DPDMD_GRAD_STEPS, DPDMDMixin
from fastgen.methods.distribution_matching.kokoro_ode_common import ODECommonMixin, _opt, ode_sample
from fastgen.methods.distribution_matching.kokoro_ode_init import _ODE_INIT_POINTS, ODEInitMixin
from fastgen.methods.distribution_matching.kokoro_pdd import PDDMixin
from fastgen.methods.distribution_matching.kokoro_rollout import RolloutMixin
from fastgen.methods.distribution_matching.kokoro_tdm import TDMMixin
from fastgen.utils import basic_utils
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.methods.config_dmd2 import ModelConfig


_ROLLOUTS = ("real_noised", "ode")
_DMD_NOISES = ("fresh", "trajectory")
_DMD_TARGETS = ("clean", "landing")
_ROLLOUT_EXITS = ("uniform", "last")
_FAKE_INPUTS = ("matched", "final")
_VSD_NORMS = ("x0", "xt")


class KokoroDMD2ODE(PDDMixin, ODEInitMixin, TDMMixin, DPDMDMixin, DMDOMixin, RolloutMixin, ODECommonMixin, KokoroDMD2):
    """Entry point for all ODE-rollout arms; each method's code lives in its own mixin (see the table above).
    With ``student_rollout=real_noised`` and all other switches at their defaults, every call is delegated unchanged
    to the parent KokoroDMD2 (= arm A)."""

    def __init__(self, config: ModelConfig):
        self._validate_switches(config)
        super().__init__(config)
        self.config = config
        logger.info(f"KokoroDMD2ODE switches: {self._switch_summary()}")

    # ── Switches ──────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_switches(config: Any) -> None:
        rollout = _opt(config, "student_rollout", "real_noised")
        noise = _opt(config, "dmd_noise", "fresh")
        target = _opt(config, "dmd_target", "clean")
        dpdmd = bool(_opt(config, "dpdmd", False))
        ode_init = bool(_opt(config, "ode_init", False))
        checks = {
            "student_rollout": (rollout, _ROLLOUTS),
            "dmd_noise": (noise, _DMD_NOISES),
            "dmd_target": (target, _DMD_TARGETS),
            "rollout_exit": (_opt(config, "rollout_exit", "uniform"), _ROLLOUT_EXITS),
            "fake_input": (_opt(config, "fake_input", "matched"), _FAKE_INPUTS),
            "vsd_norm": (_opt(config, "vsd_norm", "x0"), _VSD_NORMS),
            "dpdmd_grad_steps": (_opt(config, "dpdmd_grad_steps", "after_first"), _DPDMD_GRAD_STEPS),
            "ode_init_points": (_opt(config, "ode_init_points", "all"), _ODE_INIT_POINTS),
        }
        for name, (val, allowed) in checks.items():
            if val not in allowed:
                raise ValueError(f"KokoroDMD2ODE: {name}={val!r} is not in {allowed}")

        n_steps = int(config.student_sample_steps)
        gan_on = float(_opt(config, "gan_loss_weight_gen", 0.0)) > 0
        if bool(_opt(config, "pdd", False)):
            PDDMixin._validate_pdd(config, n_steps, gan_on, ode_init, dpdmd, target, noise)
            return
        if ode_init:
            ODEInitMixin._validate_ode_init(config, gan_on, dpdmd, target)
            return
        # All arms below have a fake score. With freq=1, iteration % 1 == 0 always takes the student step and the fake score is never updated.
        if int(config.student_update_freq) < 2:
            raise ValueError(f"arms with a fake score need STUDENT_UPDATE_FREQ >= 2 (got {config.student_update_freq})")
        if rollout == "real_noised":
            if noise != "fresh" or target != "clean" or dpdmd:
                raise ValueError(
                    "student_rollout=real_noised (arm A) only supports dmd_noise=fresh, dmd_target=clean, dpdmd=0; "
                    f"got dmd_noise={noise}, dmd_target={target}, dpdmd={dpdmd}"
                )
            return
        # The remaining arms use the ODE rollout
        if _opt(config, "student_sample_type", "ode") != "ode":
            raise ValueError("ODE-rollout training requires student_sample_type='ode' at inference (training and inference must match)")
        if gan_on:
            raise NotImplementedError(
                "GAN is not wired into the ODE-rollout paths (the discriminator's detach mode has not been validated there); "
                "only arm A (student_rollout=real_noised) supports GAN"
            )
        TDMMixin._validate_tdm(config, target, noise, dpdmd)
        if dpdmd:
            DPDMDMixin._validate_dpdmd(config, n_steps)

    @property
    def _rollout(self) -> str:
        return _opt(self.config, "student_rollout", "real_noised")

    @property
    def _uses_fake_score(self) -> bool:
        """PDD / ODE-init never call the fake score, so for i2i its reference-image KV is not prefilled."""
        return not (self._pdd or self._ode_init)

    def _switch_summary(self) -> str:
        keys = (
            "student_rollout", "rollout_exit", "fake_input", "dmd_noise", "dmd_target", "vsd_norm",
            "dpdmd", "dpdmd_div_weight", "dpdmd_teacher_steps", "dpdmd_teacher_shift", "dpdmd_k_anchor",
            "dpdmd_grad_steps", "tdm_min_t", "tdm_max_margin", "tdm_huber_c", "tdm_fake_w_floor", "tdm_anchor_weight", "tdm_anchor_hold_iters", "tdm_anchor_final_ratio", "ode_init", "ode_init_teacher_steps", "ode_init_teacher_shift",
            "ode_init_points", "pdd", "pdd_grid_steps", "pdd_shift", "pdd_solver", "pdd_l_min", "pdd_l_max",
        )
        return ", ".join(f"{k}={getattr(self.config, k, '<unset>')}" for k in keys)

    # ── Training step ─────────────────────────────────────────────────────────

    def single_train_step(self, data: Dict[str, Any], iteration: int) -> tuple[dict, dict]:
        if self._pdd:
            return self._pdd_step(data, iteration)
        if self._ode_init:
            return self._ode_init_step(data, iteration)
        if self._rollout == "real_noised":
            return super().single_train_step(data, iteration)

        real_data, condition, _ = self._prepare_training_data(data)
        self._setup_grad_requirements(iteration)
        # Start from pure noise: real_data only provides the shape (and the clean_x0 visualization)
        noise = torch.randn(real_data.shape, device=self.device, dtype=torch.float32)

        self._iteration = iteration
        if iteration % self.config.student_update_freq == 0:
            if self._dpdmd:
                loss_map, outputs = self._dpdmd_student_step(noise, real_data, condition)
            else:
                loss_map, outputs = self._rollout_student_step(noise, real_data, condition)
            self._attach_4nfe_preview(outputs, condition, iteration)
            return loss_map, outputs
        return self._rollout_fake_step(noise, condition)

    # ── Inference ─────────────────────────────────────────────────────────────

    def packed_generator_fn(
        self,
        net,
        noise_packed: torch.Tensor,
        condition: Dict[str, Any],
        student_sample_steps: int = 1,
        student_sample_type: str = "sde",
        t_list: Optional[List[float]] = None,
        precision_amp: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """ODE sampling goes through ode_sample (same code as the training rollout, fp32 state, quantized t); sde goes to the parent.
        PDD: block-step sampling with one forward per block (pdd_sample), block size = N / student_sample_steps; t_list is unused."""
        if self._pdd:
            return self._pdd_generate(net, noise_packed, condition, student_sample_steps, precision_amp)
        if student_sample_type != "ode":
            return super().packed_generator_fn(
                net, noise_packed, condition, student_sample_steps, student_sample_type, t_list, precision_amp
            )
        if t_list is None:
            t_vals = self._student_t_values()
        else:
            t_vals = [self._q(v) for v in t_list]
            assert t_vals[-1] == 0.0, "t_list[-1] must be zero"
        with basic_utils.inference_mode(net, precision_amp=precision_amp, device_type=noise_packed.device.type):
            x0 = ode_sample(net, noise_packed, t_vals, condition, self.precision)
        return x0.to(dtype=noise_packed.dtype)
