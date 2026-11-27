# SPDX-License-Identifier: Apache-2.0
#
# TDM — Trajectory Distribution Matching (arXiv 2503.06674): the student step compares at the landing point of the
# current step and re-noises only inside that step's noise interval; importance weights correct the mixed noise,
# the generator loss is pseudo-Huber, and the fake score is trained with an importance-weighted v-MSE.
# Optional first-step anchor (TDM_ANCHOR_WEIGHT > 0): the DP-DMD teacher anchor (kokoro_dpdmd.py) added to the
# TDM loss, with a hold-then-cosine schedule.
#
# Switches: DMD_TARGET=landing, TDM_MIN_T, TDM_MAX_MARGIN, TDM_HUBER_C, TDM_FAKE_W_FLOOR,
#   TDM_ANCHOR_WEIGHT / TDM_ANCHOR_HOLD_ITERS / TDM_ANCHOR_FINAL_RATIO / TDM_ANCHOR_END_ITER.

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F

from fastgen.methods.distribution_matching.kokoro_ode_common import _opt


def tdm_fake_weight(tau: float, floor: float = 1.0) -> float:
    """TDM fake-score v-MSE weight min(1, 5/SNR) · max((1−τ)², f), with SNR = ((1−τ)/τ)².
    f = 1: default (the demo's weight min(1, 5/SNR) on the ε output applied directly to v; equals 1 for τ > 0.31);
    f = 0: the demo's x0 objective ‖δx0‖²·min(SNR, 5) rewritten as the equivalent weight on v, min((1−τ)², 5τ²) (under RF, δx0 = −τ·δv);
    0 < f < 1: identical to the demo while τ is small enough that (1−τ)² ≥ f; at higher noise it stops decreasing and stays at f·min(1, 5/SNR)."""
    return min(1.0, 5.0 * (tau / (1.0 - tau)) ** 2) * max((1.0 - tau) ** 2, floor)


class TDMMixin:
    """TDM noising, losses and the optional first-step anchor."""

    @staticmethod
    def _validate_tdm(config: Any, target: str, noise: str, dpdmd: bool) -> None:
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

    @property
    def _landing(self) -> bool:
        return _opt(self.config, "dmd_target", "clean") == "landing"

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

    def _tdm_student_loss(
        self, noise: torch.Tensor, t_vals: Sequence[float], k: int, x_k: torch.Tensor, v: torch.Tensor,
        x0: torch.Tensor, gen_data: torch.Tensor, condition: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """TDM part of the student step (called from RolloutMixin._rollout_student_step): noise from this step's landing
        point, the TDM generator loss and, if tdm_anchor_weight > 0, the DP-DMD first-step anchor.
        Returns (total, vsd_loss, vsd_stats, extra, fake_x0, teacher_x0, t_vsd)."""
        t_k = t_vals[k]
        x_t_vsd, t_vsd, info = self._tdm_noised(x_k, v.detach(), x0.detach(), t_k, t_vals[k + 1])
        extra: Dict[str, torch.Tensor] = {"tdm_tau": self._scalar(info["tau"]), "tdm_is_weight_gen": info["is_weight"]}
        fake_x0, teacher_x0 = self._fake_teacher_x0(x_t_vsd, t_vsd, condition)
        vsd_loss, vsd_stats = self._tdm_generator_loss(gen_data, teacher_x0, fake_x0)
        total = vsd_loss
        anchor_w = self._anchor_weight(getattr(self, "_iteration", 0))
        if anchor_w > 0:
            # TDM + sample-level anchor: match the student's first-step velocity from pure noise to the DP-DMD teacher anchor
            # (same noise, mean teacher velocity over K steps). With exit step k = 0 that step is the caller's exit-step forward
            # (with gradients, passed in as v) and is reused; for k > 0 run an extra first-step forward with gradients
            # (k is broadcast, so all ranks agree).
            x_init = noise.float() * t_vals[0]
            v_target, t_anchor = self._dpdmd_anchor_target(x_init, t_vals[0], condition)
            v_first = v if k == 0 else self._net_v(self.net, x_init, t_vals[0], condition, "student")
            anchor_loss = F.mse_loss(v_first.float(), v_target.detach().float())
            total = vsd_loss + anchor_w * anchor_loss.to(vsd_loss.dtype)
            extra.update({"anchor_loss": anchor_loss.detach(), "anchor_weight": self._scalar(anchor_w),
                          "dpdmd_t_anchor": self._scalar(t_anchor),
                          "dpdmd_v_target_abs": v_target.abs().mean().detach()})
        return total, vsd_loss, vsd_stats, extra, fake_x0, teacher_x0, t_vsd

    def _tdm_fake_loss(
        self, x_k: torch.Tensor, v: torch.Tensor, x0: torch.Tensor, t_k: float, t_land: float, condition: dict
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """TDM part of the fake-score step (called from RolloutMixin._rollout_fake_step).
        Demo (train_tdm_demo.py:1119-1134): ‖x̂0_fake − x̂0‖² × min(SNR, 5) × is_weight, with an ε-parameterized fake.
        Our fake outputs v: L = is_weight · tdm_fake_weight(τ, f) · ‖v_fake − (m − x̂0)‖², target m − x̂0 = (x_τ − x̂0)/τ;
        see tdm_fake_weight for the two extremes of the weight and the compromise between them."""
        x_t, t_tok, info = self._tdm_noised(x_k, v, x0, t_k, t_land)
        fake_v = self.fake_score(x_t, t_tok, condition=condition, fwd_pred_type="v", role="fake")
        tau = info["tau"]
        w_eps = tdm_fake_weight(tau, float(_opt(self.config, "tdm_fake_w_floor", 1.0)))
        target = info["mixed"] - x0
        loss = (info["is_weight"].float() * w_eps) * F.mse_loss(fake_v.float(), target.float(), reduction="mean")
        return loss, {"tdm_is_weight": info["is_weight"], "tdm_tau_fake": self._scalar(info["tau"])}
