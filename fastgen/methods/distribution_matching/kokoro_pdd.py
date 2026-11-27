# SPDX-License-Identifier: Apache-2.0
#
# PDD — Parallel Decoding Distillation (arXiv 2607.26004): the network gets N output heads that predict the
# velocities of N fine grid intervals from a single forward; the student advances a whole block per forward and is
# trained data-free (no fake score) to match the teacher's mean velocity on each interval.
#
# Switches: PDD=1, PDD_GRID_STEPS, PDD_SHIFT, PDD_SOLVER (midpoint | euler), PDD_L_MIN, PDD_L_MAX.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from fastgen.methods.distribution_matching.kokoro_ode_common import _opt, shifted_grid
from fastgen.utils import basic_utils
from fastgen.utils.distributed import is_rank0
import fastgen.utils.logging_utils as logger

_PDD_SOLVERS = ("midpoint", "euler")


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


class PDDMixin:
    """PDD training step, heads initialisation and block sampling."""

    @staticmethod
    def _validate_pdd(config: Any, n_steps: int, gan_on: bool, ode_init: bool, dpdmd: bool,
                      target: str, noise: str) -> None:
        solver = _opt(config, "pdd_solver", "midpoint")
        if solver not in _PDD_SOLVERS:
            raise ValueError(f"pdd_solver={solver!r} is not in {_PDD_SOLVERS}")
        if ode_init or dpdmd or target != "clean" or noise != "fresh":
            raise ValueError("pdd is a standalone method: it cannot be combined with ode_init / dpdmd / dmd_target=landing / dmd_noise=trajectory")
        if int(config.student_update_freq) != 1:
            raise ValueError("pdd has no fake score: STUDENT_UPDATE_FREQ must be 1")
        if gan_on:
            raise NotImplementedError("pdd does not support GAN")
        big_n, l_min, l_max = PDDMixin._pdd_sizes(config)
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

    def _pdd_generate(self, net, noise_packed: torch.Tensor, condition: Dict[str, Any], student_sample_steps: int,
                      precision_amp: Optional[torch.dtype]) -> torch.Tensor:
        """PDD sampling for packed_generator_fn: one forward per block (pdd_sample), block size = N / student_sample_steps."""
        grid = self._pdd_grid_vals()
        block = (len(grid) - 1) // int(student_sample_steps)
        with basic_utils.inference_mode(net, precision_amp=precision_amp, device_type=noise_packed.device.type):
            x = pdd_sample(net, noise_packed, grid, block, condition, self.precision)
        return x.to(dtype=noise_packed.dtype)
