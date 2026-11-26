# SPDX-License-Identifier: Apache-2.0
#
# KokoroDMD2ODE — KokoroDMD2 plus several variants that unroll the student ODE during training.
#
# One class, a few switches (fields of config.model; config_kokoro_dmd2.py reads each one from the
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
# Notation (same as BAGEL / FastGen): x_t = (1−t)·x0 + t·ε, network output v = ε − x0,
#   x̂0 = x − t·v, ε̂ = x + (1−t)·v. One ODE step (= DDIM reusing the predicted noise = Euler): x_{t'} = x + (t' − t)·v.
#
# Numerical conventions:
#   * The rollout state x is accumulated in fp32 and cast to self.precision only at the network input
#     (multi-step accumulation in bf16 drifts).
#   * Every t used by the rollout is quantized to self.precision before the Euler update: the network's time
#     embedding consumes a bf16 t anyway, so the step size matches the t the network sees exactly
#     (in bf16, 0.999 → 1.0 and 0.624 → 0.625).
#   * Training rollouts, in-training previews (packed_generator_fn) and offline evaluation
#     (scripts/inference/batch_eval.py) share the module-level ode_rollout / ode_sample, so results from the
#     same noise are bit-identical.
#   * VSD noising / comparison / normalization reuse the parent class (bf16 τ, _vsd_loss); arms differ only in switches.
#
# FSDP: every rank must run the same number of forwards in each single_train_step. The exit step k is drawn on
#   rank 0 and broadcast; DP-DMD's rollout length, the teacher anchor steps and the ODE-init teacher grid are all
#   deterministic; the task type (t2i/i2i) is broadcast by MixedTaskLoader.
#
# GAN: not supported on the new paths (all arms here run with GAN off); GAN is only available in arm A.
#   See _validate_switches.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastgen.methods.distribution_matching.kokoro_dmd2 import KokoroDMD2
from fastgen.utils import basic_utils
from fastgen.utils.distributed import is_rank0, synchronize, world_size
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.methods.config_dmd2 import ModelConfig


_ROLLOUTS = ("real_noised", "ode")
_DMD_NOISES = ("fresh", "trajectory")
_DMD_TARGETS = ("clean", "landing")
_ROLLOUT_EXITS = ("uniform", "last")
_FAKE_INPUTS = ("matched", "final")
_VSD_NORMS = ("x0", "xt")
_DPDMD_GRAD_STEPS = ("after_first", "last")
_ODE_INIT_POINTS = ("all", "random")
_PDD_SOLVERS = ("midpoint", "euler")


def _opt(config: Any, name: str, default: Any) -> Any:
    """Read a config field; if it is absent (older configs lack it) or None, return the default, i.e. the baseline behaviour."""
    val = getattr(config, name, default)
    return default if val is None else val


def shifted_grid(n_steps: int, shift: float, t_start: float, include: Sequence[float] = ()) -> List[float]:
    """Teacher ODE grid: an n_steps-step shifted schedule (s ∈ linspace(1,0), σ = shift·s / (1 + (shift−1)·s)),
    clipped at t_start (the student's start time), then merged with ``include`` (the student t_list; ODE-init needs
    the grid to pass exactly through these points).
    Returns a strictly decreasing list of floats ending at 0."""
    s = torch.linspace(1.0, 0.0, n_steps + 1, dtype=torch.float64)
    sig = shift * s / (1.0 + (shift - 1.0) * s)
    vals = {min(float(v), float(t_start)) for v in sig}
    vals |= {float(v) for v in include if float(v) <= float(t_start)}
    grid = sorted(vals, reverse=True)
    assert grid[0] == float(t_start) and grid[-1] == 0.0, grid
    return grid


def quantize_t(t: float, dtype: torch.dtype) -> float:
    """Quantize t to the network precision (the value the time embedding actually sees; in bf16, 0.999 → 1.0)."""
    return float(torch.tensor(float(t), dtype=torch.float64).to(dtype))


def net_velocity(net, x: torch.Tensor, t: float, condition: dict, dtype: torch.dtype, role: str) -> torch.Tensor:
    """One inference-branch forward returning the velocity v (in ``dtype``); x is the fp32 state and is cast to
    ``dtype`` before entering the network. Gradients follow the caller's context."""
    t_tok = torch.full((x.shape[0],), float(t), device=x.device, dtype=dtype)
    return net(x.to(dtype), t_tok, condition=condition, fwd_pred_type="v", role=role)


def tdm_fake_weight(tau: float, floor: float = 1.0) -> float:
    """TDM fake-score v-MSE weight min(1, 5/SNR) · max((1−τ)², f), with SNR = ((1−τ)/τ)².
    f = 1: default (the demo's weight min(1, 5/SNR) on the ε output applied directly to v; equals 1 for τ > 0.31);
    f = 0: the demo's x0 objective ‖δx0‖²·min(SNR, 5) rewritten as the equivalent weight on v, min((1−τ)², 5τ²) (under RF, δx0 = −τ·δv);
    0 < f < 1: identical to the demo while τ is small enough that (1−τ)² ≥ f; at higher noise it stops decreasing and stays at f·min(1, 5/SNR)."""
    return min(1.0, 5.0 * (tau / (1.0 - tau)) ** 2) * max((1.0 - tau) ** 2, floor)


@torch.no_grad()
def ode_rollout(net, noise: torch.Tensor, t_vals: Sequence[float], k: int, condition: dict,
                dtype: torch.dtype, role: str = "student") -> torch.Tensor:
    """Start from x = t_0·ε and take k Euler steps along t_vals with the network's own velocity (= DDIM reusing the
    predicted noise); return the fp32 state at t_k.
    t_vals should already be quantized (quantize_t). Training rollouts, in-training previews and batch_eval all use this function."""
    x = noise.float() * t_vals[0]
    for i in range(k):
        v = net_velocity(net, x, t_vals[i], condition, dtype, role)
        x = x + (t_vals[i + 1] - t_vals[i]) * v.float()
    return x


@torch.no_grad()
def ode_sample(net, noise: torch.Tensor, t_vals: Sequence[float], condition: dict,
               dtype: torch.dtype, role: str = "student") -> torch.Tensor:
    """Full N-step ODE sampling; returns x̂0 (fp32) from the last step."""
    n_steps = len(t_vals) - 1
    x = ode_rollout(net, noise, t_vals, n_steps - 1, condition, dtype, role)
    v = net_velocity(net, x, t_vals[n_steps - 1], condition, dtype, role)
    return x - t_vals[n_steps - 1] * v.float()


def pdd_grid(n_steps: int, shift: float) -> List[float]:
    """PDD's fine time grid (paper Eqs. 7 and 16): with the shift transform flipped to this repo's direction
    (σ=1 noise → σ=0 clean) it is σ_n = s·u/(1+(s−1)·u), u = 1 − n/N. **Not quantized**: with N=128 and shift 5,
    neighbouring grid points near σ≈1 differ by only ~0.0016, below the bf16 spacing, so quantizing would merge
    them; the network's time embedding casts t to bf16 by itself."""
    return shifted_grid(n_steps, shift, 1.0)


def pdd_advance(x: torch.Tensor, u: torch.Tensor, grid: Sequence[float], n: int, length: int) -> torch.Tensor:
    """Parallel update (paper Eqs. 9/10): x̄_{n+L} = x̄_n + Σ_{k=n}^{n+L−1} (t_{k+1} − t_k)·ū(k|x_n).
    u holds all N heads from a single forward, shape (N_vae, N, D); in this repo's direction h_k = σ_{k+1} − σ_k < 0. Returns fp32."""
    if length == 0:
        return x.float()
    h = torch.tensor([grid[k + 1] - grid[k] for k in range(n, n + length)], device=x.device, dtype=torch.float32)
    return x.float() + torch.einsum("k,tkd->td", h, u[:, n:n + length].float())


@torch.no_grad()
def pdd_sample(net, noise: torch.Tensor, grid: Sequence[float], block: int, condition: dict,
               dtype: torch.dtype) -> torch.Tensor:
    """PDD sampling (paper Algorithm 1): one forward per block, then a weighted sum of the block's heads covers the whole block;
    equivalent to the fused linear layer of paper Eqs. 14/15 (Σ Δ_k W_k), except that the heads are computed first and then
    weighted. NFE = N / block. Returns the fp32 state at σ=0."""
    n_steps = len(grid) - 1
    assert n_steps % block == 0, f"grid N={n_steps} is not divisible by block size {block}"
    x = noise.float() * grid[0]
    for n in range(0, n_steps, block):
        t_tok = torch.full((x.shape[0],), float(grid[n]), device=x.device, dtype=dtype)
        u = net(x.to(dtype), t_tok, condition=condition, role="student", pdd=True)
        x = pdd_advance(x, u, grid, n, block)
    return x


class KokoroDMD2ODE(KokoroDMD2):
    """KokoroDMD2 + training-time ODE rollout (B / C / D / TDM / ODE-init). With ``student_rollout=real_noised``
    and all other switches at their defaults, every call is delegated unchanged to the parent (= arm A)."""

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
            solver = _opt(config, "pdd_solver", "midpoint")
            if solver not in _PDD_SOLVERS:
                raise ValueError(f"pdd_solver={solver!r} is not in {_PDD_SOLVERS}")
            if ode_init or dpdmd or target != "clean" or noise != "fresh":
                raise ValueError("pdd is a standalone method: it cannot be combined with ode_init / dpdmd / dmd_target=landing / dmd_noise=trajectory")
            if int(config.student_update_freq) != 1:
                raise ValueError("pdd has no fake score: STUDENT_UPDATE_FREQ must be 1")
            if gan_on:
                raise NotImplementedError("pdd does not support GAN")
            big_n, l_min, l_max = KokoroDMD2ODE._pdd_sizes(config)
            if big_n % l_min or l_max < l_min or l_max > big_n:
                raise ValueError(f"Invalid PDD block sizes: N={big_n}, L_min={l_min}, L_max={l_max}")
            if solver == "euler" and l_min < 2:
                raise ValueError("pdd_solver=euler draws 2 intervals per sample inside a block (PDD paper, Sec. 5), so L_min must be at least 2")
            block = big_n // n_steps if big_n % n_steps == 0 else 0
            if not block or block % l_min or block > l_max:
                raise ValueError(
                    f"{n_steps}-step inference requires the block size N/steps={big_n}/{n_steps} to be a multiple of L_min={l_min} and ≤ L_max={l_max}"
                )
            net_n = _opt(getattr(config, "net", None), "pdd_grid_steps", None) if hasattr(config, "net") else None
            if net_n is not None and int(net_n) != big_n:
                raise ValueError(f"network pdd_grid_steps={net_n} does not match the method's {big_n}")
            return
        if ode_init:
            if dpdmd or target != "clean":
                raise ValueError("ode_init is a standalone warm-up stage and cannot be combined with dpdmd / dmd_target=landing")
            if int(config.student_update_freq) != 1:
                raise ValueError("ode_init has no fake-score updates: STUDENT_UPDATE_FREQ must be 1")
            if gan_on:
                raise NotImplementedError("ode_init does not support GAN")
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
        if target == "landing" and (noise != "fresh" or dpdmd):
            raise ValueError("dmd_target=landing (TDM) only supports dmd_noise=fresh, dpdmd=0")
        anchor_w = float(_opt(config, "tdm_anchor_weight", 0.0))
        if anchor_w < 0:
            raise ValueError(f"tdm_anchor_weight={anchor_w} must be non-negative")
        if anchor_w > 0:
            hold = int(_opt(config, "tdm_anchor_hold_iters", -1))
            ratio = float(_opt(config, "tdm_anchor_final_ratio", 1.0))
            if hold >= 0 and (not 0.0 <= ratio <= 1.0 or int(_opt(config, "tdm_anchor_end_iter", 0)) <= hold):
                raise ValueError(f"anchor decay requires 0 ≤ final_ratio ≤ 1 and end_iter > hold (hold={hold}, ratio={ratio})")
            if target != "landing":
                raise ValueError("tdm_anchor_weight > 0 (TDM + DP-DMD first-step anchor) requires dmd_target=landing")
            k = int(_opt(config, "dpdmd_k_anchor", 5))
            m = int(_opt(config, "dpdmd_teacher_steps", 30))
            if not 1 <= k <= m:
                raise ValueError(f"dpdmd_k_anchor={k} must be in [1, dpdmd_teacher_steps={m}]")
        if dpdmd:
            if n_steps < 2:
                raise ValueError("DP-DMD needs at least 2 steps (the first step does the diversity regression, the later steps do DMD)")
            k = int(_opt(config, "dpdmd_k_anchor", 5))
            m = int(_opt(config, "dpdmd_teacher_steps", 30))
            if not 1 <= k <= m:
                raise ValueError(f"dpdmd_k_anchor={k} must be in [1, dpdmd_teacher_steps={m}]")

    @property
    def _rollout(self) -> str:
        return _opt(self.config, "student_rollout", "real_noised")

    @property
    def _ode_init(self) -> bool:
        return bool(_opt(self.config, "ode_init", False))

    @property
    def _uses_fake_score(self) -> bool:
        """PDD / ODE-init never call the fake score, so for i2i its reference-image KV is not prefilled."""
        return not (self._pdd or self._ode_init)

    @property
    def _pdd(self) -> bool:
        return bool(_opt(self.config, "pdd", False))

    @staticmethod
    def _pdd_sizes(config: Any) -> tuple[int, int, int]:
        """(N, L_min, L_max). An L of 0 falls back to the PDD paper's Qwen-Image setting: inference NFE ∈ {2, 4, 8} → L_min = N/8, L_max = N/2."""
        big_n = int(_opt(config, "pdd_grid_steps", 128))
        l_min = int(_opt(config, "pdd_l_min", 0)) or big_n // 8
        l_max = int(_opt(config, "pdd_l_max", 0)) or big_n // 2
        return big_n, l_min, l_max

    @property
    def _trainable_param_patterns(self) -> list[str]:
        pats = super()._trainable_param_patterns
        if self._pdd and pats and "pdd_heads" not in pats:
            pats = pats + ["pdd_heads"]
        return pats

    def _setup_ema(self) -> None:
        """PDD: before EMA copies the student, initialize the N heads from llm2vae (left alone when the student weight file already contains pdd_heads)."""
        if self._pdd and not getattr(self, "_pdd_heads_loaded", False):
            if (not self.config.fsdp_meta_init) or is_rank0():
                self.net.init_pdd_heads_from_llm2vae()
                logger.info(f"PDD: bagel.pdd_heads ← llm2vae × {self.net.pdd_grid_steps}")
        super()._setup_ema()

    @property
    def _dpdmd(self) -> bool:
        return bool(_opt(self.config, "dpdmd", False))

    @property
    def _landing(self) -> bool:
        return _opt(self.config, "dmd_target", "clean") == "landing"

    @property
    def _trajectory_noise(self) -> bool:
        return _opt(self.config, "dmd_noise", "fresh") == "trajectory"

    def _switch_summary(self) -> str:
        keys = (
            "student_rollout", "rollout_exit", "fake_input", "dmd_noise", "dmd_target", "vsd_norm",
            "dpdmd", "dpdmd_div_weight", "dpdmd_teacher_steps", "dpdmd_teacher_shift", "dpdmd_k_anchor",
            "dpdmd_grad_steps", "tdm_min_t", "tdm_max_margin", "tdm_huber_c", "tdm_fake_w_floor", "tdm_anchor_weight", "tdm_anchor_hold_iters", "tdm_anchor_final_ratio", "ode_init", "ode_init_teacher_steps", "ode_init_teacher_shift",
            "ode_init_points", "pdd", "pdd_grid_steps", "pdd_shift", "pdd_solver", "pdd_l_min", "pdd_l_max",
        )
        return ", ".join(f"{k}={getattr(self.config, k, '<unset>')}" for k in keys)

    # ── Student weights: accept exported safetensors (ODE-init output as the DMD starting point) ──

    def load_student_weights_and_ema(self) -> None:
        """When ``pretrained_student_net_path`` is ``exported/<iter>/student.safetensors`` (keys = BAGEL weights without
        the ``bagel.`` prefix), load it with load_bagel_weights: the parent's torch.load cannot read safetensors, and the
        different key prefix would be silently skipped by strict=False. Everything else goes to the parent."""
        path = self.config.pretrained_student_net_path
        if not (path and str(path).endswith(".safetensors") and self.config.load_student_weights):
            return super().load_student_weights_and_ema()

        from fastgen.networks.kokoro.kokoro_network import KokoroNetwork

        if (not self.config.fsdp_meta_init) or is_rank0():
            from safetensors import safe_open

            with safe_open(path, framework="pt") as f:
                file_keys = set(f.keys())
            # This branch does not copy the teacher weights first, so a key missing from the file would silently keep
            # its value from instantiation; require the full key set
            # (only PDD's pdd_heads may be missing; _setup_ema initializes them from llm2vae)
            want = set(self.net.bagel.state_dict().keys())
            missing = sorted(k for k in want - file_keys if not k.startswith("pdd_heads."))
            if missing:
                raise RuntimeError(
                    f"student weights {path} are missing {len(missing)} tensors (first 5: {missing[:5]}) -- "
                    "wrong key prefix, or not an exported student/ema.safetensors"
                )
            KokoroNetwork.load_bagel_weights(self.net.bagel, path)
            self._pdd_heads_loaded = "pdd_heads.weight" in file_keys
        synchronize()
        self._setup_ema()

    # ── Time schedule / helpers ───────────────────────────────────────────────

    def _q(self, t: float) -> float:
        return quantize_t(t, self.precision)

    def _student_t_values(self) -> List[float]:
        """Student time schedule (quantized), [t_0, …, t_{N−1}, 0], length N+1."""
        vals = [self._q(v) for v in self._get_student_t_list(device=torch.device("cpu")).tolist()]
        assert len(vals) == int(self.config.student_sample_steps) + 1, (
            f"t_list length {len(vals)} ≠ student_sample_steps+1 = {int(self.config.student_sample_steps) + 1}"
        )
        assert all(a > b for a, b in zip(vals[:-1], vals[1:])), f"t_list is not strictly decreasing after quantization: {vals}"
        return vals

    def _scalar(self, x: float) -> torch.Tensor:
        return torch.tensor(float(x), device=self.device)

    def _t_tok(self, t: float, n_tok: int) -> torch.Tensor:
        return torch.full((n_tok,), float(t), device=self.device, dtype=self.precision)

    def _net_v(self, net, x: torch.Tensor, t: float, condition: dict, role: str) -> torch.Tensor:
        return net_velocity(net, x, t, condition, self.precision, role)

    def _broadcast_int(self, value: int) -> int:
        if world_size() > 1:
            buf = torch.tensor(int(value), dtype=torch.long, device=self.device)
            dist.broadcast(buf, src=0)
            return int(buf.item())
        return int(value)

    def _sample_exit_step(self, n_steps: int) -> int:
        """Exit step k ∈ {0..N−1}: drawn on rank 0 and broadcast to all ranks (otherwise ranks run different numbers of
        forwards and FSDP collectives hang). rollout_exit=last always picks the last step; with uniform, step 0 is
        up-weighted by student_first_step_weight."""
        if _opt(self.config, "rollout_exit", "uniform") == "last":
            return n_steps - 1
        k = 0
        if is_rank0():
            probs = torch.ones(n_steps, dtype=torch.float64)
            probs[0] = max(1.0, float(_opt(self.config, "student_first_step_weight", 1.0)))
            k = int(torch.multinomial(probs / probs.sum(), 1).item())
        return self._broadcast_int(k)

    def _vsd_t(self, n_tok: int) -> torch.Tensor:
        """VSD noise level τ: same distribution as the parent (sample_t_cfg, shifted), quantized to self.precision and expanded to every token."""
        t_img = self._sample_t_per_image(1).to(self.precision)
        return t_img.expand(n_tok).contiguous()

    def _vsd_loss_norm(
        self, gen_data: torch.Tensor, teacher_x0: torch.Tensor, fake_x0: torch.Tensor, x_t_vsd: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """vsd_norm=x0: the parent's _vsd_loss (1/mean|x − x̂_real|, as in DMD2 and arm A).
        vsd_norm=xt: the original DP-DMD denominator 1/mean|x_t − x̂_real|; gradient direction, clipping and loss form match the parent."""
        if _opt(self.config, "vsd_norm", "x0") == "x0":
            return self._vsd_loss(gen_data, teacher_x0, fake_x0)
        with torch.no_grad():
            diff = (x_t_vsd.float() - teacher_x0.float()).abs().mean()
            weight_raw = 1.0 / (diff + 1e-6)
            weight = weight_raw.clamp(max=self._VSD_WEIGHT_MAX).to(gen_data.dtype)
            pseudo_target = gen_data - (fake_x0 - teacher_x0) * weight
        loss = (0.5 * F.mse_loss(gen_data.float(), pseudo_target.float(), reduction="mean")).to(gen_data.dtype)
        stats = {
            "vsd_weight": weight.detach(),
            "vsd_weight_raw_max": weight_raw.detach(),
            "vsd_weight_clipped": (weight_raw > self._VSD_WEIGHT_MAX).float().detach(),
            "student_teacher_l1": (gen_data - teacher_x0).abs().mean().detach(),
            "fake_teacher_l1": (fake_x0 - teacher_x0).abs().mean().detach(),
        }
        return loss, stats

    def _tdm_generator_loss(
        self, gen_data: torch.Tensor, teacher_x0: torch.Tensor, fake_x0: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generator loss of the TDM demo (train_tdm_demo.py:1165-1186, with --use_huber as in its README):
            coop = sg(x̂0 + (x̂_real − x̂_fake)), w = mean|x̂0 − x̂_real| (per sample)
            huber: mean((sqrt((x̂0 − coop)² + c²) − c) / w), c = 1e-3; for c ≤ 0 it falls back to mse(x̂0, coop) / w.
        CFG is already in the teacher's cfg embedding (the demo uses sd_cond + (cfg−1)(sd_cond − sd_uncond)). The lower bound on w matches _vsd_loss (1/1e3)."""
        c = float(_opt(self.config, "tdm_huber_c", 1e-3))
        with torch.no_grad():
            w = (gen_data.float() - teacher_x0.float()).abs().mean().clamp(min=1.0 / self._VSD_WEIGHT_MAX)
            coop = gen_data.float() + (teacher_x0.float() - fake_x0.float())
        diff = gen_data.float() - coop
        if c > 0:
            loss = ((diff.pow(2) + c * c).sqrt() - c).div(w).mean()
        else:
            loss = diff.pow(2).div(w).mean()
        stats = {
            "vsd_weight": (1.0 / w).detach(),
            "student_teacher_l1": (gen_data - teacher_x0).abs().mean().detach(),
            "fake_teacher_l1": (fake_x0 - teacher_x0).abs().mean().detach(),
        }
        return loss.to(gen_data.dtype), stats

    # ── ODE rollout ───────────────────────────────────────────────────────────

    def _ode_rollout_to(
        self, noise: torch.Tensor, t_vals: Sequence[float], k: int, condition: dict, net=None
    ) -> torch.Tensor:
        """no_grad: the student takes k Euler steps from t_0·ε to t_k (module-level ode_rollout). Same starting point as packed_generator_fn."""
        return ode_rollout(self.net if net is None else net, noise, t_vals, k, condition, self.precision)

    def _exit_forward(self, x_k: torch.Tensor, t_k: float, condition: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Exit-step forward (gradients follow the caller's context): returns (v, x̂0), with x̂0 in fp32."""
        v = self._net_v(self.net, x_k, t_k, condition, "student")
        return v, x_k - t_k * v.float()

    def _teacher_euler(self, x: torch.Tensor, grid: Sequence[float], n_steps: int, condition: dict,
                       record: Optional[Dict[int, float]] = None) -> tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """no_grad teacher Euler (cfg embedding, role=teacher): take n_steps steps along grid.
        record = {name: t}: save the state when the rollout reaches t (used by ODE-init)."""
        states: Dict[int, torch.Tensor] = {}
        want = {} if record is None else {t: name for name, t in record.items()}
        with torch.no_grad():
            for i in range(n_steps):
                if grid[i] in want:
                    states[want[grid[i]]] = x.clone()
                v = self._net_v(self.teacher, x, grid[i], condition, "teacher")
                x = x + (grid[i + 1] - grid[i]) * v.float()
        return x, states

    def _anchor_weight(self, iteration: int) -> float:
        """TDM anchor weight λ(iteration): constant λ up to hold, then cosine-decayed to λ·final_ratio (reached at end_iter). hold < 0 = constant."""
        lam = float(_opt(self.config, "tdm_anchor_weight", 0.0))
        hold = int(_opt(self.config, "tdm_anchor_hold_iters", -1))
        if lam <= 0 or hold < 0 or iteration <= hold:
            return lam
        end = int(_opt(self.config, "tdm_anchor_end_iter", 0))
        ratio = float(_opt(self.config, "tdm_anchor_final_ratio", 1.0))
        x = min(1.0, (iteration - hold) / max(end - hold, 1))
        return lam * (ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * x)))

    def _dpdmd_anchor_target(
        self, x_init: torch.Tensor, t0: float, condition: dict
    ) -> tuple[torch.Tensor, float]:
        """DP-DMD anchor (no_grad): from the same start x_init = t_0·ε, the teacher (cfg embedding) takes dpdmd_k_anchor ODE steps
        on a dpdmd_teacher_steps-step shifted schedule to z_K; returns the mean velocity v_target = (x_init − z_K)/(t_0 − t_K) and t_K.
        Shared by arm D and the TDM anchor."""
        grid = [self._q(t) for t in shifted_grid(
            int(_opt(self.config, "dpdmd_teacher_steps", 30)),
            float(_opt(self.config, "dpdmd_teacher_shift", 3.0)),
            t0,
        )]
        grid = sorted(set(grid), reverse=True)
        k_anchor = min(int(_opt(self.config, "dpdmd_k_anchor", 5)), len(grid) - 1)
        z_k, _ = self._teacher_euler(x_init, grid, k_anchor, condition)
        t_anchor = grid[k_anchor]
        return (x_init - z_k) / (t0 - t_anchor), t_anchor

    def _tdm_noised(
        self, x_k: torch.Tensor, v: torch.Tensor, x0: torch.Tensor, t_k: float, t_land: float
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """TDM: first step to this step's endpoint (the landing point) with the student's own velocity,
        x_land = x_k + (t_land − t_k)·v, then add fresh noise from the landing point up to
        τ ∈ [max(t_land, tdm_min_t), t_k − tdm_max_margin) (the TDM demo's --use_separate: lowt = max(20, t_mid),
        upt = t_g − 10, i.e. 0.02 / 0.01 after normalizing by 1000 steps). The RF forward transition is
            x_τ = a·x_land + β·z, a = (1−τ)/(1−t_land), β = sqrt(τ² − (a·t_land)²)
        Writing x_τ = (1−τ)·x̂0 + τ·m, m = (a·t_land·ε̂ + β·z)/τ is the "mixed noise" (partly the student's predicted ε̂).
        The fake-score importance weight is exp(−½·mean m²) / exp(−½·mean z²) (as in the TDM demo). At the last step
        t_land=0, so m = z and this reduces to plain DMD compared only on [tdm_min_t, t_{N−1}). All inputs are detached."""
        lo = max(t_land, float(_opt(self.config, "tdm_min_t", 0.02)))
        hi = t_k - float(_opt(self.config, "tdm_max_margin", 0.01))
        assert hi > lo, f"TDM: empty noise interval for this step (t_k={t_k}, interval=[{lo}, {hi}))"
        tau = self._q(lo + (hi - lo) * float(torch.rand((), device=self.device)))
        tau = min(max(tau, lo), hi)
        eps_hat = x_k + (1.0 - t_k) * v.float()
        x_land = (1.0 - t_land) * x0 + t_land * eps_hat
        a = (1.0 - tau) / (1.0 - t_land)
        beta = math.sqrt(max(tau * tau - (a * t_land) ** 2, 0.0))
        z = torch.randn_like(x0)
        x_tau = a * x_land + beta * z
        mixed = (a * t_land * eps_hat + beta * z) / max(tau, 1e-8)
        is_weight = torch.exp(-0.5 * mixed.pow(2).mean()) / torch.exp(-0.5 * z.pow(2).mean())
        info = {"tau": tau, "t_land": t_land, "mixed": mixed, "is_weight": is_weight.detach()}
        return x_tau.to(self.precision), self._t_tok(tau, x0.shape[0]), info

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

    def _fake_teacher_x0(
        self, x_t_vsd: torch.Tensor, t_vsd: torch.Tensor, condition: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fake and teacher x0 predictions on x_τ (no_grad, same order as the parent). The networks output v and
        x0 = x_τ − τ·v is computed in fp32: inside the network that step is bf16 (ulp 0.004–0.008), and the rounding
        noise from subtracting two separately rounded x0s exceeds the true fake−teacher gap
        (especially with TDM's Huber c=1e-3)."""
        tau = t_vsd.float().unsqueeze(-1)
        with torch.no_grad():
            fake_v = self.fake_score(x_t_vsd, t_vsd, condition=condition, fwd_pred_type="v", role="fake")
            teacher_v = self.teacher(x_t_vsd, t_vsd, condition=condition, fwd_pred_type="v", role="teacher")
        x_t = x_t_vsd.float()
        return x_t - tau * fake_v.float(), x_t - tau * teacher_v.float()

    def _dmd_compare(
        self, gen_data: torch.Tensor, x_t_vsd: torch.Tensor, t_vsd: torch.Tensor, condition: dict
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Fake and teacher x0 predictions on x_τ (fp32); the VSD gradient is applied to gen_data."""
        fake_x0, teacher_x0 = self._fake_teacher_x0(x_t_vsd, t_vsd, condition)
        loss, stats = self._vsd_loss_norm(gen_data, teacher_x0, fake_x0, x_t_vsd)
        return loss, stats, fake_x0, teacher_x0

    def _rollout_student_step(
        self, noise: torch.Tensor, real_data: torch.Tensor, condition: dict
    ) -> tuple[dict, dict]:
        """Student step for B / C / TDM: no_grad rollout to the exit step k, then one forward with gradients whose x̂0 is the "generated sample"."""
        t_vals = self._student_t_values()
        n_steps = len(t_vals) - 1
        k = self._sample_exit_step(n_steps)
        t_k = t_vals[k]
        x_k = self._ode_rollout_to(noise, t_vals, k, condition)
        v, x0 = self._exit_forward(x_k, t_k, condition)
        gen_data = x0.to(self.precision)

        extra: Dict[str, torch.Tensor] = {}
        if self._landing:
            x_t_vsd, t_vsd, info = self._tdm_noised(x_k, v.detach(), x0.detach(), t_k, t_vals[k + 1])
            extra = {"tdm_tau": self._scalar(info["tau"]), "tdm_is_weight_gen": info["is_weight"]}
        else:
            t_vsd = self._vsd_t(gen_data.shape[0])
            eps = noise.to(self.precision) if self._trajectory_noise else torch.randn_like(gen_data)
            x_t_vsd = self._flow_forward_process(gen_data, eps, t_vsd.to(gen_data.dtype))

        if self._landing:
            fake_x0, teacher_x0 = self._fake_teacher_x0(x_t_vsd, t_vsd, condition)
            vsd_loss, vsd_stats = self._tdm_generator_loss(gen_data, teacher_x0, fake_x0)
        else:
            vsd_loss, vsd_stats, fake_x0, teacher_x0 = self._dmd_compare(gen_data, x_t_vsd, t_vsd, condition)
        total = vsd_loss
        anchor_w = self._anchor_weight(getattr(self, "_iteration", 0)) if self._landing else 0.0
        if anchor_w > 0:
            # TDM + sample-level anchor: match the student's first-step velocity from pure noise to the DP-DMD teacher anchor
            # (same noise, mean teacher velocity over K steps). With exit step k = 0 that step is the forward with gradients above
            # and is reused; for k > 0 run an extra first-step forward with gradients (k is broadcast, so all ranks agree).
            x_init = noise.float() * t_vals[0]
            v_target, t_anchor = self._dpdmd_anchor_target(x_init, t_vals[0], condition)
            v_first = v if k == 0 else self._net_v(self.net, x_init, t_vals[0], condition, "student")
            anchor_loss = F.mse_loss(v_first.float(), v_target.detach().float())
            total = vsd_loss + anchor_w * anchor_loss.to(vsd_loss.dtype)
            extra.update({"anchor_loss": anchor_loss.detach(), "anchor_weight": self._scalar(anchor_w),
                          "dpdmd_t_anchor": self._scalar(t_anchor),
                          "dpdmd_v_target_abs": v_target.abs().mean().detach()})
        loss_map = {
            "total_loss": total,
            "vsd_loss": vsd_loss,
            f"vsd_loss_exit{k}": vsd_loss.detach(),
            "rollout_exit_k": self._scalar(float(k)),
            **vsd_stats,
            **extra,
        }
        outputs = {
            "gen_packed": gen_data.detach(),
            "kokoro_x0_vis": self._first_image_vis_payload(
                condition["patchified_vae_latent_shapes"][0],
                real_data,
                gen_data, fake_x0, teacher_x0,
                t_student_first=self._scalar(t_k), t_vsd_first=t_vsd[0],
            ),
        }
        return loss_map, outputs

    def _dpdmd_student_step(
        self, noise: torch.Tensor, real_data: torch.Tensor, condition: dict
    ) -> tuple[dict, dict]:
        """D (DP-DMD, arXiv 2602.03139; official train_sd35_dpdmd.py:430–526):
        the student runs all N steps from pure noise; the first-step velocity v₁ is matched to the mean velocity of the
        teacher ODE anchor (diversity term) and detached afterwards, and the DMD gradient flows through steps 2..N
        (with dpdmd_grad_steps=last only the final step carries gradients). L = L_DMD + λ·L_div."""
        t_vals = self._student_t_values()
        n_steps = len(t_vals) - 1
        x_init = noise.float() * t_vals[0]

        # Teacher anchor: same noise, the teacher (cfg embedding) takes K ODE steps on the shifted schedule → z_K (no_grad)
        v_target, t_anchor = self._dpdmd_anchor_target(x_init, t_vals[0], condition)

        # Student rollout: the first step carries gradients (for the diversity term only), then detach
        v_first = self._net_v(self.net, x_init, t_vals[0], condition, "student")
        x = (x_init + (t_vals[1] - t_vals[0]) * v_first.float()).detach()
        grad_all = _opt(self.config, "dpdmd_grad_steps", "after_first") == "after_first"
        x0 = None
        for i in range(1, n_steps):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (grad_all or i == n_steps - 1)):
                v = self._net_v(self.net, x, t_vals[i], condition, "student")
                if i < n_steps - 1:
                    x = x + (t_vals[i + 1] - t_vals[i]) * v.float()
                else:
                    x0 = x - t_vals[i] * v.float()
        gen_data = x0.to(self.precision)

        div_loss = F.mse_loss(v_first.float(), v_target.detach().float())
        t_vsd = self._vsd_t(gen_data.shape[0])
        eps = noise.to(self.precision) if self._trajectory_noise else torch.randn_like(gen_data)
        x_t_vsd = self._flow_forward_process(gen_data, eps, t_vsd.to(gen_data.dtype))
        vsd_loss, vsd_stats, fake_x0, teacher_x0 = self._dmd_compare(gen_data, x_t_vsd, t_vsd, condition)

        lam = float(_opt(self.config, "dpdmd_div_weight", 0.05))
        total = vsd_loss + lam * div_loss.to(vsd_loss.dtype)
        loss_map = {
            "total_loss": total,
            "vsd_loss": vsd_loss,
            "div_loss": div_loss.detach(),
            "dpdmd_t_anchor": self._scalar(t_anchor),
            "dpdmd_v_target_abs": v_target.abs().mean().detach(),
            "dpdmd_v_first_abs": v_first.detach().float().abs().mean(),
            **vsd_stats,
        }
        outputs = {
            "gen_packed": gen_data.detach(),
            "kokoro_x0_vis": self._first_image_vis_payload(
                condition["patchified_vae_latent_shapes"][0],
                real_data,
                gen_data, fake_x0, teacher_x0,
                t_student_first=self._scalar(0.0), t_vsd_first=t_vsd[0],
            ),
        }
        return loss_map, outputs

    def _rollout_fake_step(self, noise: torch.Tensor, condition: dict) -> tuple[dict, dict]:
        """Fake-score step: fit the distribution of whatever the student step compares (samples built under no_grad, fake trained with DSM).
          B / C: the same rollout with a freshly drawn exit step (as in Self-Forcing); with fake_input=final, the full N-step output
          D: the full N-step output (as in the official implementation)
          TDM: x_τ built the same way (noised from the landing point), target is the student's x̂0, scaled by the importance weight"""
        t_vals = self._student_t_values()
        n_steps = len(t_vals) - 1
        if self._dpdmd or _opt(self.config, "fake_input", "matched") == "final":
            k = n_steps - 1
        else:
            k = self._sample_exit_step(n_steps)
        t_k = t_vals[k]
        x_k = self._ode_rollout_to(noise, t_vals, k, condition)
        with torch.no_grad():
            v, x0 = self._exit_forward(x_k, t_k, condition)
        gen_data = x0.to(self.precision)

        stats: Dict[str, torch.Tensor] = {"rollout_exit_k_fake": self._scalar(float(k))}
        if self._landing:
            # Demo (train_tdm_demo.py:1119-1134): ‖x̂0_fake − x̂0‖² × min(SNR, 5) × is_weight, with an ε-parameterized fake.
            # Our fake outputs v: L = is_weight · tdm_fake_weight(τ, f) · ‖v_fake − (m − x̂0)‖², target m − x̂0 = (x_τ − x̂0)/τ;
            # see tdm_fake_weight for the two extremes of the weight and the compromise between them.
            x_t, t_tok, info = self._tdm_noised(x_k, v, x0, t_k, t_vals[k + 1])
            fake_v = self.fake_score(x_t, t_tok, condition=condition, fwd_pred_type="v", role="fake")
            tau = info["tau"]
            w_eps = tdm_fake_weight(tau, float(_opt(self.config, "tdm_fake_w_floor", 1.0)))
            target = info["mixed"] - x0
            loss = (info["is_weight"].float() * w_eps) * F.mse_loss(fake_v.float(), target.float(), reduction="mean")
            stats.update({"tdm_is_weight": info["is_weight"], "tdm_tau_fake": self._scalar(info["tau"])})
        else:
            t_tok = self._vsd_t(gen_data.shape[0])
            eps = noise.to(self.precision) if self._trajectory_noise else torch.randn_like(gen_data)
            x_t = self._flow_forward_process(gen_data, eps, t_tok.to(gen_data.dtype))
            fake_v = self.fake_score(x_t, t_tok, condition=condition, fwd_pred_type="v", role="fake")
            loss = F.mse_loss(fake_v.float(), (eps - gen_data).float(), reduction="mean")

        loss_map = {
            "total_loss": loss,
            "fake_score_loss": loss.detach(),
            "gan_loss_disc": torch.zeros((), device=self.device),
            **stats,
        }
        return loss_map, {"gen_packed": gen_data.detach()}

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

    def _pdd_grid_vals(self) -> List[float]:
        return pdd_grid(self._pdd_sizes(self.config)[0], float(_opt(self.config, "pdd_shift", 5.0)))

    def _pdd_heads(self, x: torch.Tensor, t: float, condition: dict) -> torch.Tensor:
        return self.net(x.to(self.precision), self._t_tok(t, x.shape[0]), condition=condition, role="student", pdd=True)

    def _teacher_mean_velocity(self, x_k: torch.Tensor, k: int, grid: Sequence[float], condition: dict) -> torch.Tensor:
        """Teacher mean velocity over interval k (paper Eqs. 5/6), no_grad, cfg embedding:
        euler: v(x_k, t_k); midpoint: x_mid = x_k + h/2·v(x_k, t_k), returns v(x_mid, (t_k + t_{k+1})/2)."""
        t_k, h = grid[k], grid[k + 1] - grid[k]
        with torch.no_grad():
            v = self._net_v(self.teacher, x_k, t_k, condition, "teacher").float()
            if _opt(self.config, "pdd_solver", "midpoint") == "euler":
                return v
            return self._net_v(self.teacher, x_k + 0.5 * h * v, t_k + 0.5 * h, condition, "teacher").float()

    def _pdd_step(self, data: Dict[str, Any], iteration: int) -> tuple[dict, dict]:
        """PDD (arXiv 2607.26004) data-free training (Algorithm 3). The paper carries (x̄_n, n) across iterations and advances
        by L_min each time; here every micro-step uses a new prompt, so instead the block start n ∈ {0, L_min, …, N−L_min} is
        drawn uniformly (broadcast from rank 0) and the student's own block steps (L_min each, stop-grad) take pure noise to
        x̄_n — the same on-policy state distribution as Algorithm 3.
        The student then runs one forward at x̄_n (with gradients) to get all N heads; k ∈ [n, min(n+L_max, N)) is drawn
        uniformly, x̄_k = x̄_n + Σ_{l=n}^{k−1} h_l·ū(l|x̄_n) (stop-grad), and L = ‖ū(k|x̄_n) − u_k^teacher(x̄_k)‖² (Eq. 11).
        With euler, each sample averages over 2 distinct k (paper Sec. 5: matches the 2 teacher forwards of midpoint)."""
        real_data, condition, _ = self._prepare_training_data(data)
        self._setup_grad_requirements(iteration)
        grid = self._pdd_grid_vals()
        big_n, l_min, l_max = self._pdd_sizes(self.config)

        n = 0
        if is_rank0():
            n = l_min * int(torch.randint(0, big_n // l_min, ()).item())
        n = self._broadcast_int(n)

        noise = torch.randn(real_data.shape, device=self.device, dtype=torch.float32)
        x = noise * grid[0]
        with torch.no_grad():
            for m in range(0, n, l_min):
                x = pdd_advance(x, self._pdd_heads(x, grid[m], condition), grid, m, l_min)
        u = self._pdd_heads(x, grid[n], condition)                       # (N_vae, N, D), with gradients

        k_hi = min(n + l_max, big_n)
        n_k = 2 if _opt(self.config, "pdd_solver", "midpoint") == "euler" else 1
        ks = (torch.randperm(k_hi - n)[:n_k] + n).tolist()
        losses, stats = [], {}
        for k in ks:
            x_k = pdd_advance(x, u.detach(), grid, n, k - n)
            u_t = self._teacher_mean_velocity(x_k, k, grid, condition)
            losses.append(F.mse_loss(u[:, k].float(), u_t))
        loss = torch.stack(losses).mean()

        with torch.no_grad():
            x_jump = pdd_advance(x, u.detach(), grid, n, big_n - n)       # jump straight to σ=0 with all heads from this forward (visualization only)
        stats.update({
            "pdd_loss": loss.detach(),
            "pdd_block_start_n": self._scalar(n),
            "pdd_t_n": self._scalar(grid[n]),
            "pdd_k_offset": self._scalar(float(ks[0] - n)),
            "pdd_u_teacher_abs": u_t.abs().mean().detach(),
        })
        shape = condition["patchified_vae_latent_shapes"][0]
        n_tok = shape[0] * shape[1]
        outputs = {
            "gen_packed": x_jump.to(self.precision),
            "kokoro_x0_vis": {
                "shape": shape,
                "t_student_first": self._scalar(grid[n]),
                "t_vsd_first": self._scalar(0.0),
                "clean_x0": real_data[:n_tok].detach(),
                "student_x0": x_jump[:n_tok].to(self.precision),
            },
        }
        self._attach_4nfe_preview(outputs, condition, iteration)
        return {"total_loss": loss, **stats}, outputs

    # ── Preview / inference ───────────────────────────────────────────────────

    def _attach_4nfe_preview(self, outputs: dict, condition: dict, iteration: int) -> None:
        """Same gate as the parent's single_train_step: on iteration 1 and every sample_logging_iter student steps, run one full N-step inference."""
        every = self._sample_logging_iter()
        if not (iteration == 1 or (every > 0 and iteration % every == 0)):
            return
        with torch.no_grad():
            gen = outputs["gen_packed"]
            preview = self.packed_generator_fn(
                net=self.net_inference,
                noise_packed=torch.randn(gen.shape, device=gen.device, dtype=torch.float32),
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=getattr(self.config, "student_sample_type", "ode"),
                t_list=getattr(self.config.sample_t_cfg, "t_list", None),
                precision_amp=self.precision_amp_infer,
            )
        shape = condition["patchified_vae_latent_shapes"][0]
        outputs["kokoro_x0_vis"]["student_4nfe"] = preview[: shape[0] * shape[1]].detach()

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
            grid = self._pdd_grid_vals()
            block = (len(grid) - 1) // int(student_sample_steps)
            with basic_utils.inference_mode(net, precision_amp=precision_amp, device_type=noise_packed.device.type):
                x = pdd_sample(net, noise_packed, grid, block, condition, self.precision)
            return x.to(dtype=noise_packed.dtype)
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
