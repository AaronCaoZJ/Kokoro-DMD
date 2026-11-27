# SPDX-License-Identifier: Apache-2.0
#
# D — DP-DMD (Diversity-Preserved DMD, arXiv 2602.03139): the student runs all N steps from pure noise; its
# first-step velocity is regressed onto the mean velocity of a K-step teacher ODE anchor (diversity term) and
# detached, and the DMD gradient flows through steps 2..N. The fake score is fit on the full N-step output
# (RolloutMixin._rollout_fake_step). The anchor target is shared with TDM's optional first-step anchor.
#
# Switches: DPDMD=1, DPDMD_DIV_WEIGHT, DPDMD_TEACHER_STEPS, DPDMD_TEACHER_SHIFT, DPDMD_K_ANCHOR,
#   DPDMD_GRAD_STEPS (after_first | last), VSD_NORM=xt (the official x_t-based normaliser).

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fastgen.methods.distribution_matching.kokoro_ode_common import _opt, shifted_grid

_DPDMD_GRAD_STEPS = ("after_first", "last")


class DPDMDMixin:
    """DP-DMD student step and teacher anchor."""

    @staticmethod
    def _validate_dpdmd(config: Any, n_steps: int) -> None:
        if n_steps < 2:
            raise ValueError("DP-DMD needs at least 2 steps (the first step does the diversity regression, the later steps do DMD)")
        k = int(_opt(config, "dpdmd_k_anchor", 5))
        m = int(_opt(config, "dpdmd_teacher_steps", 30))
        if not 1 <= k <= m:
            raise ValueError(f"dpdmd_k_anchor={k} must be in [1, dpdmd_teacher_steps={m}]")

    @property
    def _dpdmd(self) -> bool:
        return bool(_opt(self.config, "dpdmd", False))

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
        eps = self._vsd_eps(noise, gen_data)
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
