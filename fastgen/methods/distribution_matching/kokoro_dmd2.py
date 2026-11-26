# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# KokoroDMD2 — DMD2 training adapted for the Kokoro/Bagel packed-sequence format.
#
# Structural differences from standard DMD2Model:
#   - "Real data" lives in packed token space: (N_vae, patch_dim), not (B, C, H, W)
#   - Timesteps are per-image, broadcast to each image's VAE tokens
#   - The noise forward-process operates on packed tokens
#   - CFG is handled inside KokoroNetwork/Bagel; no separate neg_condition path
#   - VSD loss uses per-image normalisation weights

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastgen.methods.distribution_matching.dmd2 import DMD2Model
from fastgen.methods.common_loss import gan_loss_generator, gan_loss_discriminator
from fastgen.datasets.kokoro_cfgdistill_data import patchify_vae_latent
from fastgen.utils import basic_utils
import fastgen.utils.logging_utils as logger

# Default trainable-parameter selection — CFG lives only in the teacher (no cfg
# knob in the final model): only the gen MoE branch trains; cfg_*_embedder stays
# frozen because student/fake never call them.  Override via config.trainable_param_patterns;
# [""] / [] = whole net (which the KV-cache guard will reject).
_DEFAULT_TRAINABLE_PATTERNS = ["moe_gen"]

if TYPE_CHECKING:
    from fastgen.configs.methods.config_dmd2 import ModelConfig


class KokoroDMD2(DMD2Model):
    """
    DMD2 for the Kokoro/Bagel packed-sequence format.

    The ``data`` dict produced by ``KokoroPackedDataset`` / ``kokoro_collate_fn``
    has the following fields (among others):

        padded_latent             (N_img, C, H_max, W_max)  — clean VAE latents
        patchified_vae_latent_shapes  [(h_tok, w_tok), …]   — per-image shapes
        packed_vae_token_indexes  (N_vae,)  — positions of VAE tokens in sequence
        mse_loss_indexes          (N_vae,)  — positions where loss is computed
        packed_text_ids           (N_text,)
        packed_text_indexes       (N_text,)
        packed_position_ids       (L,)
        nested_attention_masks    list of (sl_i, sl_i) tensors
        sample_lens               list[int]
        sequence_length           int
        packed_latent_position_ids (N_vae,)

    The ``KokoroNetwork.forward(x_t, t, condition, fwd_pred_type, use_cfg, cfg_scale)``
    interface is used for all three networks (student / teacher / fake_score).
    """

    # ── model construction ────────────────────────────────────────────────────

    def build_teacher(self) -> None:
        """
        Override FastGenModel.build_teacher() to load Bagel weights from
        a safetensors checkpoint (standard FastGen only handles .pt/.pth via
        torch.load; Kokoro saves as ema.safetensors).

        Everything else (architecture instantiation, teacher.eval/requires_grad,
        synchronize) mirrors the base implementation exactly.
        """
        from fastgen.utils import instantiate
        from fastgen.utils.distributed import synchronize, is_rank0
        from fastgen.networks.kokoro.kokoro_network import KokoroNetwork

        meta_init_teacher = self.config.add_teacher_to_fsdp_dict and self.config.fsdp_meta_init
        logger.info(
            f"build_teacher: add_teacher_to_fsdp_dict={self.config.add_teacher_to_fsdp_dict}, "
            f"fsdp_meta_init={self.config.fsdp_meta_init}, meta_init_teacher={meta_init_teacher}"
        )
        with self._get_meta_init_context(meta_init_teacher):
            self.teacher = instantiate(self.teacher_config)

        model_path = self.config.pretrained_model_path
        if model_path is not None and len(model_path) > 0:
            if (not self.config.fsdp_meta_init) or is_rank0():
                KokoroNetwork.load_bagel_weights(self.teacher.bagel, model_path)
            synchronize()

        self.teacher.eval().requires_grad_(False)
        synchronize()

    # ── internal helpers ──────────────────────────────────────────────────────

    @property
    def _latent_patch_size(self) -> int:
        return getattr(self.config, "latent_patch_size", 2)

    @staticmethod
    def _sample_from_batch(data: Any) -> Dict[str, Any]:
        """The inference-style dataloader yields ``[sample_dict]`` (batch=1,
        passthrough collate), so take ``sample = batch[0]``."""
        if isinstance(data, (list, tuple)):
            assert len(data) == 1, (
                f"Kokoro inference dataloader is batch_size=1; got {len(data)}"
            )
            return data[0]
        return data  # already a sample dict (defensive)

    def _clean_packed_latent(self, data: Any) -> torch.Tensor:
        """Patchify the single clean VAE latent into packed tokens
        ``(N_vae, patch_dim)`` using ``patchify_vae_latent`` (token layout
        matches Bagel.prepare_vae_latent).
        """
        sample = self._sample_from_batch(data)
        tokens = patchify_vae_latent(sample["vae_latent"], self._latent_patch_size)
        return tokens.to(device=self.device, dtype=self.precision)

    # ── config-driven trainable scope ─────────────────────────────────────────

    @property
    def _trainable_param_patterns(self) -> list[str]:
        pats = getattr(self.config, "trainable_param_patterns", None)
        return list(pats) if pats is not None else list(_DEFAULT_TRAINABLE_PATTERNS)

    def build_model(self) -> None:
        """DMD2 build, then restrict student & fake_score to the configured
        trainable subset (default = ``["moe_gen"]``) BEFORE init_optimizers,
        so the optimizers (get_optimizer filters requires_grad) only cover that
        subset.  teacher stays fully frozen (build_teacher)."""
        super().build_model()
        pats = self._trainable_param_patterns
        logger.info(f"KokoroDMD2 trainable_param_patterns = {pats or '<all>'}")
        self.net.apply_trainable_param_patterns(pats)
        self.fake_score.apply_trainable_param_patterns(pats)

    def _setup_grad_requirements(self, iteration: int) -> None:
        """Pattern-aware student/fake alternation (overrides DMD2 base, which
        would wholesale-unfreeze fake_score and clobber the subset).  teacher
        always frozen; with GAN on, the discriminator trains on fake_score steps."""
        pats = self._trainable_param_patterns
        gan_on = self.config.gan_loss_weight_gen > 0
        if iteration % self.config.student_update_freq == 0:
            # student update: net active (subset), fake_score frozen.
            # discriminator frozen+eval — its activations still pass the
            # generator GAN gradient through to gen_data (no detach there).
            self.fake_score.eval().requires_grad_(False)
            self.net.train()
            self.net.apply_trainable_param_patterns(pats)
            if gan_on:
                self.discriminator.eval().requires_grad_(False)
        else:
            # fake_score update: fake_score active (subset), net frozen.
            # discriminator active (full module) — it is the only thing that
            # learns on this step besides fake_score.
            self.net.eval().requires_grad_(False)
            self.fake_score.train()
            self.fake_score.apply_trainable_param_patterns(pats)
            if gan_on:
                self.discriminator.train().requires_grad_(True)

    def _build_condition(self, data: Any) -> Dict[str, Any]:
        """Build the inference-branch condition (cond KV-cache context + flow
        layout) for the single sample, via the student network.

        The cond context is built ONCE from the student's Bagel and shared by
        student / fake_score / teacher (one ``InterleaveInferencer`` bound to
        ``student``).  Safe because the default trainable patterns
        (``["moe_gen"]``) leave the text/VIT/VAE understanding paths frozen
        and bit-identical across the three models.  The KV-cache guard in
        ``KokoroNetwork.apply_trainable_param_patterns`` rejects any pattern
        that would violate this.

        cfg_text_scale / cfg_img_scale stored in the condition are only used
        by the teacher forward (role="teacher"); student/fake ignore them.
        """
        sample = self._sample_from_batch(data)
        cfg_text = (
            float(self.config.guidance_scale)
            if self.config.guidance_scale is not None
            else 1.0
        )
        cfg_img = float(getattr(self.config, "cfg_img_teacher", 2.0))
        condition = self.net.build_inference_condition(
            sample, self.device, cfg_text_scale=cfg_text, cfg_img_scale=cfg_img
        )
        # i2i: the reference image's VAE tokens are prefilled through the gen
        # branch (Bagel.forward_cache_update_vae, mode="gen" -> moe_gen weights),
        # which the student / fake_score train.  Sharing the student-built cache
        # would make the teacher's (and fake's) conditioning drift with the
        # student, so teacher and fake_score prefill their own cache.  Text / ViT
        # prefill goes through the frozen und branch and is identical; t2i keeps
        # the single shared cache.  The layout (token positions) is identical.
        if condition["task_type"] == "i2i" and getattr(self.config, "cond_kv_per_role", True):
            roles = [("teacher", self.teacher)]
            if getattr(self, "_uses_fake_score", True):
                roles.append(("fake", self.fake_score))
            by_role = {}
            for role, net in roles:
                c = net.build_inference_condition(
                    sample, self.device, cfg_text_scale=cfg_text, cfg_img_scale=cfg_img
                )
                by_role[role] = c["cond_ctx"]
            condition["cond_ctx_by_role"] = by_role
            if not getattr(self, "_logged_kv_per_role", False):
                logger.info(f"cond_kv_per_role: i2i reference-image KV prefilled separately for {[r for r, _ in roles]}")
                self._logged_kv_per_role = True
        return condition

    def _first_image_vis_payload(
        self,
        shape: tuple[int, int],
        clean_x0: torch.Tensor,
        student_x0: torch.Tensor,
        fake_x0: torch.Tensor,
        teacher_x0: torch.Tensor,
        t_student_first: torch.Tensor,
        t_vsd_first: torch.Tensor,
        teacher_cond_x0: Optional[torch.Tensor] = None,
        teacher_uncond_x0: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Build a compact first-image payload for train-time x0 visualization."""
        n_tok = shape[0] * shape[1]
        payload = {
            "shape": shape,
            "t_student_first": t_student_first.float().detach(),
            "t_vsd_first": t_vsd_first.float().detach(),
            "clean_x0": clean_x0[:n_tok].detach(),
            "student_x0": student_x0[:n_tok].detach(),
            "fake_x0": fake_x0[:n_tok].detach(),
            "teacher_x0": teacher_x0[:n_tok].detach(),
        }
        if teacher_cond_x0 is not None:
            payload["teacher_cond_x0"] = teacher_cond_x0[:n_tok].detach()
        if teacher_uncond_x0 is not None:
            payload["teacher_uncond_x0"] = teacher_uncond_x0[:n_tok].detach()
        return payload

    def _teacher_branch_stats(
        self,
        teacher_x0: torch.Tensor,
        teacher_cond_x0: Optional[torch.Tensor],
        teacher_uncond_x0: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Summarize numerical gaps between teacher cond/uncond/CFG branches."""
        stats: Dict[str, torch.Tensor] = {}
        if teacher_cond_x0 is not None:
            stats["teacher_cfg_vs_cond_l1"] = (
                teacher_x0 - teacher_cond_x0
            ).abs().mean().detach()
        if teacher_uncond_x0 is not None:
            stats["teacher_cfg_vs_uncond_l1"] = (
                teacher_x0 - teacher_uncond_x0
            ).abs().mean().detach()
        if teacher_cond_x0 is not None and teacher_uncond_x0 is not None:
            stats["teacher_cond_vs_uncond_l1"] = (
                teacher_cond_x0 - teacher_uncond_x0
            ).abs().mean().detach()
        return stats

    # NOTE: there is no cond/uncond two-pass CFG here; every model runs a single
    # conditional forward, and CFG enters only through the teacher's cfg embedding.

    # VSD weight upper bound.  Without this, the
    # per-image weight 1/(diff+1e-6) can spike to ~1e6 when student converges
    # toward teacher, dominating gradient direction even after grad-norm clip.
    _VSD_WEIGHT_MAX = 1e3

    def _vsd_loss(
        self,
        gen_data: torch.Tensor,
        teacher_x0: torch.Tensor,
        fake_x0: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute VSD loss on packed tokens (single-image invariant).

        Mathematically equivalent to FastGen's
        ``variational_score_distillation_loss`` applied to the unpatchified
        per-image tensor: weight is the inverse mean abs diff (clamped), the
        gradient direction is ``(fake - teacher) * w`` (stop-grad), and the
        student is pulled toward ``gen - vsd_grad``.
        """
        assert gen_data.shape == teacher_x0.shape == fake_x0.shape

        with torch.no_grad():
            # fp32 weight computation, then cast back to gen_data dtype.
            diff = (gen_data.float() - teacher_x0.float()).abs().mean()
            weight_raw = 1.0 / (diff + 1e-6)
            weight = weight_raw.clamp(max=self._VSD_WEIGHT_MAX).to(gen_data.dtype)
            vsd_grad = (fake_x0 - teacher_x0) * weight
            pseudo_target = gen_data - vsd_grad

        loss = 0.5 * F.mse_loss(gen_data.float(), pseudo_target.float(), reduction="mean")
        loss = loss.to(gen_data.dtype)

        stats = {
            "vsd_weight": weight.detach(),
            "vsd_weight_raw_max": weight_raw.detach(),
            "vsd_weight_clipped": (weight_raw > self._VSD_WEIGHT_MAX).float().detach(),
            "student_teacher_l1": (gen_data - teacher_x0).abs().mean().detach(),
            "fake_teacher_l1": (fake_x0 - teacher_x0).abs().mean().detach(),
        }
        return loss, stats

    def _per_token_timesteps(
        self,
        shapes: list,
        t_per_image: torch.Tensor,          # (N_img,) or scalar tensor
    ) -> torch.Tensor:
        """
        Broadcast per-image timesteps to per-token timesteps.

        Returns:
            (N_vae,) float tensor.
        """
        parts = []
        for i, (h_tok, w_tok) in enumerate(shapes):
            n_tok = h_tok * w_tok
            parts.append(t_per_image[i].expand(n_tok))
        return torch.cat(parts, dim=0)                  # (N_vae,)

    def _flow_forward_process(
        self,
        x0: torch.Tensor,                   # (N_vae, D)
        eps: torch.Tensor,                  # (N_vae, D)
        t: torch.Tensor,                    # (N_vae,)
    ) -> torch.Tensor:
        """
        Rectified-flow forward process in packed token space.
            x_t = (1 − t) · x_0 + t · eps
        """
        t_col = t.unsqueeze(-1)             # (N_vae, 1) for broadcasting
        return (1.0 - t_col) * x0 + t_col * eps

    def _sample_t_per_image(self, n_images: int) -> torch.Tensor:
        """
        Sample one timestep per image using ``config.sample_t_cfg``.

        Returns:
            (n_images,) tensor in noise_scheduler.t_precision on self.device.
        """
        t_per_image = self.net.noise_scheduler.sample_t(n_images, device=self.device, **basic_utils.convert_cfg_to_dict(self.config.sample_t_cfg))
        return t_per_image                  # (n_images,) in noise_scheduler.t_precision

    def _sample_t_per_image_from_list(self, n_images: int) -> torch.Tensor:
        """
        Sample one discrete timestep per image from the student t_list.

        Used for multi-step training so every image is trained at an exact
        inference-time timestep rather than a uniformly random one.

        Returns:
            (n_images,) tensor in self.precision on self.device.
        """
        t_list = self._get_student_t_list(device=self.device)

        train_t_list = t_list[:-1]  # never train on the clean-data endpoint t=0
        first_step_weight = float(getattr(self.config, "student_first_step_weight", 1.0))
        if first_step_weight <= 1.0 or train_t_list.numel() <= 1:
            ids = torch.randint(0, train_t_list.numel(), (n_images,), device=self.device)
        else:
            probs = torch.ones(train_t_list.numel(), device=self.device, dtype=torch.float32)
            probs[0] = first_step_weight
            probs = probs / probs.sum()
            ids = torch.multinomial(probs, num_samples=n_images, replacement=True)
        return train_t_list[ids].to(self.precision)

    def _get_student_t_list(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Return the discrete student timestep schedule.

        For Kokoro we intentionally decouple the student schedule from the RF
        scheduler's continuous-sampling max_t. Student training and inference
        should start from pure noise, so the first step is hard-set to 1.0.
        """
        device = device or self.device
        t_list_cfg = getattr(self.config.sample_t_cfg, "t_list", None)
        if t_list_cfg is not None:
            t_list = torch.as_tensor(t_list_cfg, device=device, dtype=self.net.noise_scheduler.t_precision)
        else:
            t_list = torch.linspace(
                1.0,
                0.0,
                self.config.student_sample_steps + 1,
                device=device,
                dtype=self.net.noise_scheduler.t_precision,
            )
        assert t_list[-1].item() == 0.0, "student t_list[-1] must be zero"
        return t_list

    def _packed_student_sample_loop(
        self,
        x: torch.Tensor,            # (N_vae, D) initial noisy packed tokens
        t_list: torch.Tensor,       # [max_t, ..., 0], length = student_sample_steps + 1
        condition: dict,
        student_sample_type: str = "sde",
    ) -> torch.Tensor:              # (N_vae, D) denoised packed tokens
        """
        Multi-step denoising loop in packed token space.

        Equivalent to FastGenModel._student_sample_loop but operates on
        (N_vae, D) packed tokens instead of (B, C, H, W) image tensors.

        Args:
            x:                   Starting packed noisy tokens.
            t_list:              Descending timestep schedule from max_t to 0,
                                 length = student_sample_steps + 1.
            condition:           Packed batch dict; must have
                                 ``patchified_vae_latent_shapes``.
            student_sample_type: "sde" adds fresh noise at each step;
                                 "ode" re-uses noise implied by the trajectory.

        Returns:
            Final x0 prediction from the last denoising step.
        """
        shapes = condition["patchified_vae_latent_shapes"]
        n_images = len(shapes)
        x_pred = x

        for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
            t_img_cur = t_cur.expand(n_images).to(self.precision)
            t_per_token = self._per_token_timesteps(shapes, t_img_cur)   # (N_vae,)

            x_pred = self.net(
                x, t_per_token, condition=condition,
                fwd_pred_type="x0", role="student",
            )

            if t_next > 0:
                t_img_next = t_next.expand(n_images).to(self.precision)
                t_next_per_token = self._per_token_timesteps(shapes, t_img_next)

                if student_sample_type == "sde":
                    eps_infer = torch.randn_like(x_pred)
                else:
                    # ODE: reconstruct eps from x_t = (1-t)*x0 + t*eps
                    #   → eps = (x_t - (1-t)*x0) / t
                    eps_infer = (x - (1.0 - t_per_token.unsqueeze(-1)) * x_pred) / (
                        t_per_token.unsqueeze(-1).clamp(min=1e-8)
                    )

                x = self._flow_forward_process(x_pred, eps_infer, t_next_per_token)

        return x_pred

    # ── GAN discriminator helpers (inference-branch teacher features) ────────

    def _teacher_features(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: dict
    ) -> list:
        """Frozen-teacher inference-branch features for the discriminator.

        Returns a list of ``[1, hidden, h_tok, w_tok]`` spatial feature maps
        (one per ``discriminator.feature_indices`` layer).  The caller decides
        the grad context: the generator step calls this WITHOUT no_grad (so the
        GAN gradient reaches gen_data); the discriminator step wraps it in
        no_grad (only the discriminator learns there).
        """
        return self.teacher(
            x_t, t, condition=condition, role="teacher",
            return_features_early=True,
            feature_indices=set(self.discriminator.feature_indices),
        )

    def _compute_real_feat(
        self,
        real_data: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        condition: dict,
    ) -> tuple[list, torch.Tensor]:
        """Discriminator features on perturbed *real* clean latents (single image).

        ``gan_use_same_t_noise`` (config default True) reuses the fake branch's
        t/eps — the single-timestep invariant of ``_forward_flow_train`` then
        holds trivially.  When False, a fresh single per-image t (broadcast to
        tokens, still single-valued) is sampled.
        """
        if self.config.gan_use_same_t_noise:
            t_real, eps_real = t, eps
        else:
            shapes = condition["patchified_vae_latent_shapes"]
            t_img = self._sample_t_per_image(len(shapes)).to(self.precision)
            t_real = self._per_token_timesteps(shapes, t_img)
            eps_real = torch.randn_like(real_data)
        perturbed_real = self._flow_forward_process(
            real_data, eps_real, t_real.to(real_data.dtype)
        )
        real_feat = self._teacher_features(perturbed_real, t_real, condition)
        return real_feat, t_real

    def _compute_r1_regularization(
        self,
        real_feat_logit: torch.Tensor,
        real_data: torch.Tensor,
        t_real: torch.Tensor,
        condition: dict,
    ) -> torch.Tensor:
        """Approximate R1 (APT-style): MSE between the discriminator logits on
        real features and on features of real data + small Gaussian noise.

        Mirrors base DMD2 ``_compute_r1_regularization`` (additive alpha noise,
        not a flow-forward) adapted to the single-image packed teacher forward.
        """
        perturbed_real_alpha = real_data + self.config.gan_r1_reg_alpha * torch.randn_like(real_data)
        with torch.no_grad():
            real_feat_alpha = self._teacher_features(perturbed_real_alpha, t_real, condition)
        real_feat_alpha_logit = self.discriminator(real_feat_alpha)
        return F.mse_loss(real_feat_logit, real_feat_alpha_logit, reduction="mean")

    # ── overrides ─────────────────────────────────────────────────────────────

    def _prepare_training_data(
        self, data: Dict[str, Any]
    ) -> tuple[torch.Tensor, Any, Any]:
        """
        Override: extract packed clean latent; condition is the full batch dict.

        Returns:
            real_data:   (N_vae, patch_dim) clean patchified latents
            condition:   batch dict (all Bagel inputs except latent/timesteps)
            neg_condition: None  (CFG is handled inside KokoroNetwork/Bagel)
        """
        real_data = self._clean_packed_latent(data)
        condition = self._build_condition(data)
        return real_data, condition, None

    def _generate_noise_and_time(
        self, real_data: torch.Tensor, condition: Optional[Dict] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Override: per-image timestep sampling, forward-process in packed space.

        Args:
            real_data:  (N_vae, patch_dim) clean latents
            condition:  batch dict (must contain patchified_vae_latent_shapes)

        Returns:
            x_t_student: (N_vae, D) noisy latent for student input
            t_student:   (N_vae,)   per-token student timestep
            t:           (N_vae,)   per-token VSD timestep
            eps:         (N_vae, D) noise used for VSD forward process
        """
        shapes = condition["patchified_vae_latent_shapes"]
        n_images = len(shapes)

        # ── Student timestep ──────────────────────────────────────────────────
        if self.config.student_sample_steps == 1:
            # Single-step: input is pure noise (t = 1)
            t_img_student = torch.full(
                (n_images,), 1.0,
                device=self.device, dtype=self.precision,
            )
        else:
            # Multi-step: sample from the discrete t_list so each image is
            # trained at an exact inference-time timestep (not uniformly random).
            t_img_student = self._sample_t_per_image_from_list(n_images)

        t_student = self._per_token_timesteps(shapes, t_img_student)   # (N_vae,)

        eps_student = torch.randn_like(real_data)
        x_t_student = self._flow_forward_process(real_data, eps_student, t_student)

        # ── VSD timestep (for fake_score / teacher) ───────────────────────────
        t_img = self._sample_t_per_image(n_images).to(self.precision)
        t = self._per_token_timesteps(shapes, t_img)                    # (N_vae,)
        eps = torch.randn_like(real_data)

        return x_t_student, t_student, t, eps

    def _student_update_step(
        self,
        input_student: torch.Tensor,        # (N_vae, D)
        t_student: torch.Tensor,            # (N_vae,)
        t: torch.Tensor,                    # (N_vae,)
        eps: torch.Tensor,                  # (N_vae, D)
        data: Dict[str, Any],
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
    ) -> tuple[dict, dict]:
        """
        Packed-format student update step.

        Flow:
            1. Student forward → gen_data (N_vae, D)
            2. Add VSD noise → x_t_vsd (N_vae, D)
            3. fake_score(x_t_vsd) → fake_x0 (N_vae, D)
            4. teacher(x_t_vsd, use_cfg) → teacher_x0 (N_vae, D)
            5. VSD loss with per-image weights
        """
        assert condition is not None
        clean_x0 = self._clean_packed_latent(data)

        # ── 1. Student forward: x0 prediction (no cfg embed) ─────────────────
        gen_data = self.net(
            input_student, t_student, condition=condition,
            fwd_pred_type="x0", role="student",
        )  # (N_vae, D)

        # ── 2. Add VSD noise ──────────────────────────────────────────────────
        x_t_vsd = self._flow_forward_process(gen_data, eps, t.to(gen_data.dtype))

        # ── 3. Fake score prediction (no cfg embed, no grad) ─────────────────
        # fake / teacher output v and x0 = x_t − t·v is rebuilt in fp32 (same as
        # KokoroDMD2ODE._fake_teacher_x0).  Asking the network for x0 rounds each
        # x0 to bf16 (ulp 0.004–0.008) before fake − teacher, and the bf16
        # pseudo-target then drops small score differences.
        tau = t.float().unsqueeze(-1)
        x_t_f32 = x_t_vsd.float()
        with torch.no_grad():
            fake_v = self.fake_score(
                x_t_vsd, t, condition=condition,
                fwd_pred_type="v", role="fake",
            )  # (N_vae, D)
        fake_x0 = x_t_f32 - tau * fake_v.float()

        # ── 4. Teacher prediction (uses cfg embed at distilled scale) ────────
        # Teacher = frozen CFG-distilled model.  ``role="teacher"`` enables
        # ``use_cfg_embed=True`` with condition's cfg_text_scale (and cfg_img_scale
        # for i2i).  Student/fake do NOT use cfg embed, so cfg
        # knowledge gets absorbed into student's moe_gen.
        gan_stats: Dict[str, torch.Tensor] = {}
        if self.config.gan_loss_weight_gen > 0:
            # GAN on: ONE grad-enabled teacher forward yields teacher_x0 (VSD
            # target — detached below) AND fake_feat (carries the generator GAN
            # gradient back to gen_data through the shared first-N-layers graph).
            # detach() touches only teacher_x0; fake_feat's graph is intact.
            # The forward MUST stay outside no_grad or fake_feat goes graph-less.
            teacher_v, fake_feat = self.teacher(
                x_t_vsd, t, condition=condition,
                fwd_pred_type="v", role="teacher",
                feature_indices=set(self.discriminator.feature_indices),
            )
            teacher_x0 = (x_t_f32 - tau * teacher_v.float()).detach()
            # discriminator is eval()+requires_grad_(False) this step (grad
            # flows through its activations only, not its params).
            fake_logit = self.discriminator(fake_feat)
            gan_loss_gen = gan_loss_generator(fake_logit)
            gan_stats["gan_logit_fake_gen_mean"] = fake_logit.detach().mean()
        else:
            with torch.no_grad():
                teacher_v = self.teacher(
                    x_t_vsd, t, condition=condition,
                    fwd_pred_type="v", role="teacher",
                )
            teacher_x0 = (x_t_f32 - tau * teacher_v.float()).detach()
            gan_loss_gen = torch.zeros((), device=self.device, dtype=gen_data.dtype)

        # ── 5. VSD loss (clamped weight, packed-token flat form) ──────────────
        vsd_loss, vsd_stats = self._vsd_loss(gen_data, teacher_x0, fake_x0)
        teacher_branch_stats = self._teacher_branch_stats(teacher_x0, None, None)

        loss = vsd_loss + self.config.gan_loss_weight_gen * gan_loss_gen
        loss_map = {
            "total_loss": loss,
            "vsd_loss": vsd_loss,
            "gan_loss_gen": gan_loss_gen,
            **vsd_stats,
            **teacher_branch_stats,
            **gan_stats,
        }
        outputs = {
            "gen_packed": gen_data.detach(),
            "kokoro_x0_vis": self._first_image_vis_payload(
                condition["patchified_vae_latent_shapes"][0],
                clean_x0,
                gen_data,
                fake_x0,
                teacher_x0,
                t_student_first=t_student[0],
                t_vsd_first=t[0],
            ),
        }
        return loss_map, outputs

    def _fake_score_discriminator_update_step(
        self,
        input_student: torch.Tensor,        # (N_vae, D)
        t_student: torch.Tensor,            # (N_vae,)
        t: torch.Tensor,                    # (N_vae,)
        eps: torch.Tensor,                  # (N_vae, D)
        real_data: torch.Tensor,            # unused (real_data == clean latent)
        condition: Optional[Any] = None,
    ) -> tuple[dict, dict]:
        """
        Packed-format fake_score update step (DSM objective).

        Flow:
            1. Student forward (no grad) → gen_data
            2. Add VSD noise → x_t_sg
            3. fake_score(x_t_sg, t) → fake_v_pred  [velocity prediction]
            4. DSM target: flow_velocity = eps − gen_data
            5. MSE loss
        """
        assert condition is not None

        # ── 1-2. Generate data & add noise (no grad, no cfg embed) ───────────
        with torch.no_grad():
            gen_data = self.net(
                input_student, t_student, condition=condition,
                fwd_pred_type="x0", role="student",
            )
            x_t_sg = self._flow_forward_process(gen_data, eps, t.to(gen_data.dtype))

        # ── 3. Fake score prediction (velocity, DSM, no cfg embed) ───────────
        fake_v_pred = self.fake_score(
            x_t_sg, t, condition=condition,
            fwd_pred_type="v", role="fake",
        )

        # ── 4. Flow-matching DSM target: v = eps − x0 ─────────────────────────
        flow_v_target = eps - gen_data     # (N_vae, D)

        # ── 5. MSE loss ──────────────────────────────────────────────────────
        loss_fakescore = F.mse_loss(
            fake_v_pred.float(), flow_v_target.float(), reduction="mean"
        )

        # ── 6. Discriminator GAN loss (+ optional R1) ─────────────────────────
        # All teacher feature forwards are no_grad here (features detached); only
        # the discriminator learns this step.  real_data == clean packed latent
        # (from _prepare_training_data).  See base DMD2 _fake_score_discriminator
        # _update_step for the reference structure.
        gan_loss_disc = torch.zeros((), device=self.device, dtype=loss_fakescore.dtype)
        gan_loss_ar1 = torch.zeros((), device=self.device, dtype=loss_fakescore.dtype)
        gan_stats: Dict[str, torch.Tensor] = {}
        if self.config.gan_loss_weight_gen > 0:
            with torch.no_grad():
                fake_feat = self._teacher_features(x_t_sg, t, condition)
                real_feat, t_real = self._compute_real_feat(
                    real_data, t, eps, condition
                )
            real_logit = self.discriminator(real_feat)
            fake_logit = self.discriminator(fake_feat)
            gan_loss_disc = gan_loss_discriminator(real_logit, fake_logit)
            gan_stats = {
                "gan_logit_real_mean": real_logit.detach().mean(),
                "gan_logit_fake_disc_mean": fake_logit.detach().mean(),
            }
            if self.config.gan_r1_reg_weight > 0:
                gan_loss_ar1 = self._compute_r1_regularization(
                    real_logit, real_data, t_real, condition
                )

        loss = loss_fakescore + gan_loss_disc + self.config.gan_r1_reg_weight * gan_loss_ar1
        loss_map = {
            "total_loss": loss,
            "fake_score_loss": loss_fakescore,
            "gan_loss_disc": gan_loss_disc,
            **gan_stats,
        }
        if self.config.gan_loss_weight_gen > 0 and self.config.gan_r1_reg_weight > 0:
            loss_map["gan_loss_ar1"] = gan_loss_ar1
        outputs = {"gen_packed": gen_data.detach()}
        return loss_map, outputs

    # ── sample_logging_iter resolver ─────────────────────────────────────────

    def _sample_logging_iter(self) -> int:
        """How often to dump the full 4-NFE student inference into wandb.

        ``self.config`` is the *model* sub-config (``config.model``); it has no
        ``trainer``/``callbacks``.  The config builder mirrors the trainer's
        ``callbacks.wandb.sample_logging_iter`` into ``config.model`` so the
        model can read it from its own namespace.  Fall back to a sane interval
        if unset (omegaconf strict raises on a missing key, so use getattr)."""
        val = getattr(self.config, "sample_logging_iter", None)
        if val is not None:
            return int(val)
        return int(getattr(self.config, "logging_iter", 50))

    # ── single_train_step: delegate to DMD2Model with adjusted signature ─────

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict, dict]:
        """
        Override to (a) pass ``condition`` into ``_generate_noise_and_time``
        (needed for per-image timestep sampling), and (b) inject a full 4-NFE
        student inference into ``outputs["kokoro_x0_vis"]["student_4nfe"]`` on
        wandb sample-logging iterations so the wandb callback shows the actual
        few-step output (the per-step ``student_x0`` is only a single-forward
        x0 prediction at a random t, not the inference-time chain).
        """
        real_data, condition, neg_condition = self._prepare_training_data(data)
        self._setup_grad_requirements(iteration)
        input_student, t_student, t, eps = self._generate_noise_and_time(
            real_data, condition=condition
        )

        if iteration % self.config.student_update_freq == 0:
            loss_map, outputs = self._student_update_step(
                input_student, t_student, t, eps, data,
                condition=condition, neg_condition=neg_condition,
            )
            # Inject 4-NFE preview at logging boundaries.  ``packed_generator_fn``
            # is called under no_grad inside (it uses self.net in inference_mode),
            # so this adds ~4 forward passes wall-time only on those iterations.
            sample_log_iter = self._sample_logging_iter()
            if iteration == 1 or (sample_log_iter > 0 and iteration % sample_log_iter == 0):
                with torch.no_grad():
                    gen_packed = outputs["gen_packed"]
                    noise_packed = torch.randn_like(gen_packed)
                    student_4nfe = self.packed_generator_fn(
                        net=self.net_inference,
                        noise_packed=noise_packed,
                        condition=condition,
                        student_sample_steps=self.config.student_sample_steps,
                        student_sample_type=getattr(
                            self.config, "student_sample_type", "sde"
                        ),
                        t_list=getattr(self.config.sample_t_cfg, "t_list", None),
                        precision_amp=self.precision_amp_infer,
                    )
                    shape = condition["patchified_vae_latent_shapes"][0]
                    n_tok = shape[0] * shape[1]
                    outputs["kokoro_x0_vis"]["student_4nfe"] = (
                        student_4nfe[:n_tok].detach()
                    )
            return loss_map, outputs
        else:
            return self._fake_score_discriminator_update_step(
                input_student, t_student, t, eps, real_data, condition=condition
            )

    # ── _get_outputs ──────────────────────────────────────────────────────────

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: Optional[torch.Tensor] = None,
        condition: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Build the outputs dict returned by single_train_step.

        Single-step: packed gen tokens available immediately.
        Multi-step:  also expose a lazy generator callable (``gen_rand``) that
                     runs the full denoising chain from pure packed noise — useful
                     for validation callbacks that reconstruct images.
        """
        if self.config.student_sample_steps == 1:
            return {"gen_packed": gen_data}

        # Multi-step: provide a lazy evaluation function for inference callbacks.
        # The callable signature matches what FastGen validation expects:
        #   gen_rand() → (N_vae, D) packed denoised tokens
        assert condition is not None, "_get_outputs: condition required for multi-step"
        shapes = condition["patchified_vae_latent_shapes"]
        n_images = len(shapes)
        t_list_cfg = getattr(self.config.sample_t_cfg, "t_list", None)

        noise_packed = torch.randn_like(gen_data)   # (N_vae, D) pure noise

        gen_rand_func = partial(
            self.packed_generator_fn,
            net=self.net_inference,
            noise_packed=noise_packed,
            condition=condition,
            student_sample_steps=self.config.student_sample_steps,
            student_sample_type=getattr(self.config, "student_sample_type", "sde"),
            t_list=t_list_cfg,
            precision_amp=self.precision_amp_infer,
        )
        return {"gen_rand": gen_rand_func, "gen_packed": gen_data}

    # ── packed_generator_fn: multi-step inference in packed token space ───────

    def packed_generator_fn(
        self,
        net,
        noise_packed: torch.Tensor,         # (N_vae, D) initial pure noise
        condition: Dict[str, Any],
        student_sample_steps: int = 1,
        student_sample_type: str = "sde",
        t_list: Optional[List[float]] = None,
        precision_amp: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Multi-step packed-token generation with the distilled student.

        Replaces the base ``generator_fn`` for packed-sequence format.
        Starts from pure noise in packed token space and runs
        ``student_sample_steps`` denoising iterations using
        ``_packed_student_sample_loop``.

        Args:
            net:                 Student network (``KokoroNetwork`` instance).
            noise_packed:        ``(N_vae, D)`` i.i.d. Gaussian noise matching
                                 the packed token layout from ``condition``.
            condition:           Packed batch dict; must include
                                 ``patchified_vae_latent_shapes``.
            student_sample_steps: Number of denoising steps (≥ 1).
            student_sample_type: ``"sde"`` (default) or ``"ode"``.
            t_list:              Optional custom descending timestep schedule
                                 as a Python list of floats.  If ``None`` the
                                 scheduler's default uniform grid is used.
            precision_amp:       If not ``None``, run under autocast.

        Returns:
            ``(N_vae, D)`` denoised packed tokens.
        """
        with basic_utils.inference_mode(net, precision_amp=precision_amp,
                                        device_type=noise_packed.device.type):
            if t_list is None:
                t_sched = self._get_student_t_list(device=noise_packed.device)
            else:
                t_sched = torch.tensor(
                    t_list, device=noise_packed.device,
                    dtype=net.noise_scheduler.t_precision,
                )
            assert t_sched[-1].item() == 0, "t_list[-1] must be zero"

            # Scale initial noise to x_{t=max_t} = max_t * eps  (RF convention)
            shapes = condition["patchified_vae_latent_shapes"]
            n_images = len(shapes)
            t_init_per_token = self._per_token_timesteps(
                shapes,
                t_sched[0].expand(n_images).to(self.precision),
            )
            x_init = noise_packed * t_init_per_token.unsqueeze(-1)

            return self._packed_student_sample_loop(
                x_init, t_sched, condition,
                student_sample_type=student_sample_type,
            ).to(dtype=noise_packed.dtype)
