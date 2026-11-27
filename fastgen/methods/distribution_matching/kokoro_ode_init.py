# SPDX-License-Identifier: Apache-2.0
#
# ODE-init — online teacher-ODE initialisation (an online version of CausVid's ODE regression / FastGen KD): the
# teacher integrates from pure noise to 0 through the student time steps and the student regresses the teacher
# endpoint from each recorded state. Used as a warm start before TDM or PDD (kokoro_ode/run_odeinit_pipeline.sh);
# the exported student.safetensors is loaded by ODECommonMixin.load_student_weights_and_ema.
#
# Switches: ODE_INIT=1, ODE_INIT_TEACHER_STEPS, ODE_INIT_TEACHER_SHIFT, ODE_INIT_POINTS (all | random).

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from fastgen.methods.distribution_matching.kokoro_ode_common import _opt, shifted_grid

_ODE_INIT_POINTS = ("all", "random")


class ODEInitMixin:
    """ODE-init training step."""

    @staticmethod
    def _validate_ode_init(config: Any, gan_on: bool, dpdmd: bool, target: str) -> None:
        if dpdmd or target != "clean":
            raise ValueError("ode_init is a standalone warm-up stage and cannot be combined with dpdmd / dmd_target=landing")
        if int(config.student_update_freq) != 1:
            raise ValueError("ode_init has no fake-score updates: STUDENT_UPDATE_FREQ must be 1")
        if gan_on:
            raise NotImplementedError("ode_init does not support GAN")

    @property
    def _ode_init(self) -> bool:
        return bool(_opt(self.config, "ode_init", False))

    def _ode_init_step(self, data: Dict[str, Any], iteration: int) -> tuple[dict, dict]:
        """ODE-init (an online version of CausVid's ode_regression / FastGen KD): the teacher (cfg embedding) runs Euler from
        pure noise to 0 along a shifted grid that passes exactly through the student t_list, recording the state at each
        student time; the student does one forward on each of those states to predict x̂0 and regresses the teacher endpoint
        (MSE, as in CausVid). ode_init_points=random uses one point per trajectory (= CausVid drawing one denoising step
        per sample); all uses all N points (an unbiased estimate of the same objective that saves teacher compute).
        Storing trajectory pairs offline is equivalent to computing them online (same compute when each pair is used once);
        online avoids ~2.5 MB per pair of disk writes and rebuilding the prompt / reference image."""
        real_data, condition, _ = self._prepare_training_data(data)
        self._setup_grad_requirements(iteration)
        noise = torch.randn(real_data.shape, device=self.device, dtype=torch.float32)
        t_vals = self._student_t_values()
        n_steps = len(t_vals) - 1

        grid = [self._q(t) for t in shifted_grid(
            int(_opt(self.config, "ode_init_teacher_steps", 40)),
            float(_opt(self.config, "ode_init_teacher_shift", 3.0)),
            t_vals[0], include=t_vals,
        )]
        grid = sorted(set(grid), reverse=True)
        assert all(t in grid for t in t_vals), f"teacher grid does not pass exactly through the student t_list: {t_vals}"
        x_init = noise * t_vals[0]
        x0_teacher, states = self._teacher_euler(
            x_init, grid, len(grid) - 1, condition, record={i: t_vals[i] for i in range(n_steps)},
        )

        if _opt(self.config, "ode_init_points", "all") == "all":
            points = list(range(n_steps))
        else:
            # CausVid: draw one denoising step uniformly per sample; the student always runs exactly one forward, so no broadcast is needed
            points = [int(torch.randint(0, n_steps, ()).item())]
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        stats: Dict[str, torch.Tensor] = {}
        x0_first = None
        for k in points:
            _, x0 = self._exit_forward(states[k], t_vals[k], condition)
            loss_k = F.mse_loss(x0, x0_teacher)          # CausVid ode_regression.py: mse(pred, target), mean
            total = total + loss_k / len(points)
            stats[f"ode_init_loss_k{k}"] = loss_k.detach()
            if x0_first is None:
                x0_first = x0.detach().to(self.precision)

        loss_map = {
            "total_loss": total,
            "ode_init_loss": total.detach(),
            "ode_init_teacher_nfe": self._scalar(float(len(grid) - 1)),
            **stats,
        }
        shape = condition["patchified_vae_latent_shapes"][0]
        n_tok = shape[0] * shape[1]
        outputs = {
            "gen_packed": x0_first,
            "kokoro_x0_vis": {
                "shape": shape,
                "t_student_first": self._scalar(t_vals[points[0]]),
                "t_vsd_first": self._scalar(0.0),
                "clean_x0": real_data[:n_tok].detach(),
                "student_x0": x0_first[:n_tok],
                "teacher_x0": x0_teacher[:n_tok].detach().to(self.precision),
            },
        }
        self._attach_4nfe_preview(outputs, condition, iteration)
        return loss_map, outputs
