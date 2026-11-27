# SPDX-License-Identifier: Apache-2.0
#
# B — DMD2 with a training-time ODE rollout (Self-Forcing style): the student input is its own Euler rollout from
# pure noise to a random exit step k (drawn on rank 0 and broadcast); only the exit-step forward carries gradients.
# The student and fake-score steps here are shared with C (kokoro_dmdo.py: noise choice via _vsd_eps), D's fake
# step (kokoro_dpdmd.py: always the full N-step output) and TDM (kokoro_tdm.py: landing-point noising and losses).
#
# Switches: STUDENT_ROLLOUT=ode, ROLLOUT_EXIT (uniform | last), STUDENT_FIRST_STEP_WEIGHT, FAKE_INPUT (matched | final).

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from fastgen.methods.distribution_matching.kokoro_ode_common import _opt


class RolloutMixin:
    """Student and fake-score steps of the ODE-rollout arms (B / C / TDM; D uses the fake step)."""

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

        if self._landing:
            # TDM (kokoro_tdm.py): noise from this step's landing point, TDM generator loss, optional first-step anchor
            total, vsd_loss, vsd_stats, extra, fake_x0, teacher_x0, t_vsd = self._tdm_student_loss(
                noise, t_vals, k, x_k, v, x0, gen_data, condition
            )
        else:
            t_vsd = self._vsd_t(gen_data.shape[0])
            eps = self._vsd_eps(noise, gen_data)
            x_t_vsd = self._flow_forward_process(gen_data, eps, t_vsd.to(gen_data.dtype))
            vsd_loss, vsd_stats, fake_x0, teacher_x0 = self._dmd_compare(gen_data, x_t_vsd, t_vsd, condition)
            total, extra = vsd_loss, {}
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
            # TDM (kokoro_tdm.py): importance-weighted v-MSE on the landing-point noising
            loss, tdm_stats = self._tdm_fake_loss(x_k, v, x0, t_k, t_vals[k + 1], condition)
            stats.update(tdm_stats)
        else:
            t_tok = self._vsd_t(gen_data.shape[0])
            eps = self._vsd_eps(noise, gen_data)
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
