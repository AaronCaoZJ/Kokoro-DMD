# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# KokoroNetwork — FastGenNetwork wrapper around the Bagel model.
# Bridges FastGen's (x_t, t, condition) interface to Bagel's packed-sequence
# forward API.

import inspect
import os
import types
from typing import Any, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from fastgen.networks.network import FastGenNetwork
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger


class KokoroNetwork(FastGenNetwork):
    """
    FastGenNetwork wrapper for the Bagel model.

    FastGen DMD methods call ``forward(x_t, t, condition, fwd_pred_type)``.
    This class translates that interface to Bagel's packed-sequence forward.

    Args:
        bagel_model: An instantiated ``Bagel`` model (from fastgen.networks.kokoro.bagel).
        schedule_type: Noise schedule type.  Kokoro uses rectified flow → "rf".
        net_pred_type: Native prediction type of the network.  Bagel predicts
            flow velocity → "v".  KokoroNetwork can convert to "x0" on the fly.
    """

    def __init__(
        self,
        bagel_model: Optional[nn.Module] = None,
        bagel_ckpt_dir: Optional[str] = None,
        schedule_type: str = "rf",
        net_pred_type: str = "v",
        # _build_bagel kwargs (used when bagel_ckpt_dir is supplied)
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        visual_gen: bool = True,
        visual_und: bool = False,
        freeze_und: bool = False,
        timestep_shift: float = 1.0,
        # CFG-distill / inference-branch params
        cfg_text_max: float = 5.0,
        cfg_img_max: float = 2.5,
        # PDD (parallel decoding distillation): >0 adds N = pdd_grid_steps
        # copies of the final linear layer (bagel.pdd_heads), see
        # init_pdd_heads_from_llm2vae / forward(pdd=True).
        pdd_grid_steps: int = 0,
        **net_kwargs,
    ):
        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            **net_kwargs,
        )
        self.bagel_ckpt_dir = bagel_ckpt_dir
        self.vae = None
        self.cfg_text_max = cfg_text_max
        self.cfg_img_max = cfg_img_max
        # Lazily-loaded tokenizer + special-token id map for building the
        # inference KV-cache context in the trainer process (never in a
        # dataloader worker).
        self.tokenizer = None
        self.new_token_ids = None
        if bagel_model is None:
            if bagel_ckpt_dir is None:
                raise ValueError("KokoroNetwork requires either bagel_model or bagel_ckpt_dir")
            bagel_model = KokoroNetwork._build_bagel(
                ckpt_dir          = bagel_ckpt_dir,
                latent_patch_size = latent_patch_size,
                max_latent_size   = max_latent_size,
                visual_gen        = visual_gen,
                visual_und        = visual_und,
                freeze_und        = freeze_und,
                timestep_shift    = timestep_shift,
            )
        self.bagel = bagel_model
        self.pdd_grid_steps = int(pdd_grid_steps)
        if self.pdd_grid_steps > 0:
            # Lives inside bagel so it is saved / exported with the bagel.* keys.
            lin = self.bagel.llm2vae
            self.bagel.pdd_heads = nn.Linear(
                lin.in_features, lin.out_features * self.pdd_grid_steps, bias=lin.bias is not None
            )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Called by ``model_to_fsdp`` after ``to_empty()`` (FSDP meta-init sync), before the state dict
        is loaded.  ``to_empty`` leaves every buffer uninitialised, and non-persistent buffers are not in
        any state dict, so they must be recomputed here.  In BAGEL the only one is the LLM RoPE table
        (``Qwen2RotaryEmbedding.inv_freq``); left as garbage it scrambles attention between image tokens
        and the text / reference context (images stay coherent but ignore the prompt; i2i breaks)."""
        super().reset_parameters()
        n = 0
        for mod in self.modules():
            if hasattr(mod, "rope_init_fn") and isinstance(getattr(mod, "inv_freq", None), torch.Tensor):
                if mod.inv_freq.device.type == "meta":
                    continue
                # fp32 on purpose: the net was cast to bf16 before wrapping (Module.to casts buffers too);
                # a bf16 table carries a ~0.4% frequency error that grows with the token position.
                inv_freq, scaling = mod.rope_init_fn(mod.config, mod.inv_freq.device, **mod.rope_kwargs)
                mod.register_buffer("inv_freq", inv_freq.float(), persistent=False)
                mod.original_inv_freq = mod.inv_freq
                mod.attention_scaling = scaling
                n += 1
        for name, mod in self.named_modules():
            missing = [b for b in getattr(mod, "_non_persistent_buffers_set", ()) if b != "inv_freq"]
            if missing:
                raise RuntimeError(f"KokoroNetwork.reset_parameters: unhandled non-persistent buffers {name}.{missing}")
        logger.info(f"KokoroNetwork.reset_parameters: recomputed {n} RoPE inv_freq buffer(s)")

    @torch.no_grad()
    def _ensure_rope_fp32(self) -> None:
        """Nets that never go through to_empty() still get their RoPE table cast to bf16 by Module.to();
        put it back in fp32 (values recomputed, same as reset_parameters)."""
        rot = getattr(getattr(getattr(self.bagel, "language_model", None), "model", None), "rotary_emb", None)
        if rot is None or rot.inv_freq.dtype == torch.float32 or rot.inv_freq.device.type == "meta":
            return
        inv_freq, scaling = rot.rope_init_fn(rot.config, rot.inv_freq.device, **rot.rope_kwargs)
        rot.register_buffer("inv_freq", inv_freq.float(), persistent=False)
        rot.original_inv_freq = rot.inv_freq
        rot.attention_scaling = scaling

    @torch.no_grad()
    def init_pdd_heads_from_llm2vae(self) -> None:
        """PDD paper App. B: every parallel head starts as the pretrained final
        layer.  Output layout is head-major (N, patch_dim), so repeat the weight
        rows N times (equivalent to N copies of llm2vae)."""
        lin, heads, n = self.bagel.llm2vae, self.bagel.pdd_heads, self.pdd_grid_steps
        heads.weight.copy_(lin.weight.repeat(n, 1))
        if heads.bias is not None:
            heads.bias.copy_(lin.bias.repeat(n))

    # ── Bagel construction from checkpoint directory ────────────────────────

    @classmethod
    def _build_bagel(
        cls,
        ckpt_dir: str,
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        visual_gen: bool = True,
        visual_und: bool = True,
        freeze_und: bool = False,
        timestep_shift: float = 1.0,
    ) -> nn.Module:
        """
        Construct a Bagel model from a Kokoro checkpoint directory.

        The directory must contain:
            llm_config.json         — Qwen2 LLM config
            vit_config.json         — SiglipVision config (only if visual_und=True)

        Weight loading is NOT done here; use ``load_bagel_weights()`` after
        construction, or rely on FastGenModel._load_pretrained_model().

        Args:
            ckpt_dir:          Path to directory with llm_config.json etc.
            latent_patch_size: Latent patch size (default 2).
            max_latent_size:   Max latent size in patches per side (default 32).
            visual_gen:        Whether to enable visual generation (default True).
            visual_und:        Whether to enable visual understanding / VIT.
            freeze_und:        Whether to detach/freeze the understanding branch
                               during training updates.
            timestep_shift:    Flow-matching timestep shift used by the original
                               Bagel training/inference path. The Kokoro DMD fast
                               path bypasses this and feeds raw t in [0, 1].
        """
        # BAGEL modules are vendored under fastgen.networks.kokoro.
        from fastgen.networks.kokoro.bagel import Bagel, BagelConfig
        from fastgen.networks.kokoro.qwen2_navit import Qwen2ForCausalLM, Qwen2Config

        logger.info(f"Building Bagel architecture from {ckpt_dir}")

        # ── LLM ──────────────────────────────────────────────────────────────
        llm_config_path = os.path.join(ckpt_dir, "llm_config.json")
        if not os.path.isfile(llm_config_path):
            raise FileNotFoundError(f"llm_config.json not found in {ckpt_dir}")
        llm_config = Qwen2Config.from_json_file(llm_config_path)
        # Match the Bagel initialisation the teacher checkpoint was trained with.
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        llm_config.freeze_und = freeze_und

        language_model = Qwen2ForCausalLM(llm_config)
        language_model.init_moe()   # copy base weights → MoE gen path

        # ── VIT (optional) ────────────────────────────────────────────────────
        vit_model = None
        vit_config = None
        if visual_und:
            from fastgen.networks.kokoro.modeling.siglip_navit import SiglipVisionConfig, SiglipVisionModel
            vit_config_path = os.path.join(ckpt_dir, "vit_config.json")
            if not os.path.isfile(vit_config_path):
                raise FileNotFoundError(f"vit_config.json not found in {ckpt_dir}")
            vit_config = SiglipVisionConfig.from_json_file(vit_config_path)
            vit_config.rope = False
            vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
            vit_model = SiglipVisionModel(vit_config)
            vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        # ── VAE config (hardcoded Kokoro defaults) ────────────────────────────
        # These match the defaults in fastgen/networks/kokoro/autoencoder.py::load_ae().
        vae_config = types.SimpleNamespace(downsample=8, z_channels=16) if visual_gen else None

        # ── BagelConfig ───────────────────────────────────────────────────────
        config = BagelConfig(
            visual_gen        = visual_gen,
            visual_und        = visual_und,
            llm_config        = llm_config,
            vit_config        = vit_config,
            vae_config        = vae_config,
            vit_max_num_patch_per_side = 70,
            connector_act     = "gelu_pytorch_tanh",
            latent_patch_size = latent_patch_size,
            max_latent_size   = max_latent_size,
            timestep_shift    = timestep_shift,
        )

        bagel = Bagel(language_model, vit_model, config)
        logger.info("Bagel architecture built successfully")
        return bagel

    @staticmethod
    def load_bagel_weights(bagel: nn.Module, ckpt_path: str) -> None:
        """
        Load Bagel weights from a checkpoint file.

        Supports:
            *.safetensors   — saved via safetensors.torch.save_file()
            *.pt / *.pth    — saved via torch.save(state_dict, ...)

        Args:
            bagel:     The Bagel model to load weights into.
            ckpt_path: Path to the checkpoint file.
        """
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logger.info(f"Loading Bagel weights from {ckpt_path}")

        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path, device="cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        load_info = bagel.load_state_dict(state_dict, strict=False)
        logger.success(f"Bagel weights loaded. Missing: {len(load_info.missing_keys)}, "
                       f"unexpected: {len(load_info.unexpected_keys)}")

    # ── helpers ────────────────────────────────────────────────────────────

    def _loss_token_mask(self, x_t: torch.Tensor, condition: dict) -> torch.Tensor:
        """Return a bool mask selecting loss-token positions in the VAE tokens.

        ``x_t`` is indexed by ``packed_vae_token_indexes``.
        ``mse_loss_indexes`` (from condition) is a subset of those positions.
        The mask selects which entries of ``x_t`` correspond to loss tokens.

        For T2I training ``mse_loss_indexes == packed_vae_token_indexes``,
        so the mask is all-True.

        Returns:
            BoolTensor of shape ``(N_vae,)`` — True at loss-token positions.
        """
        vae_idx = condition["packed_vae_token_indexes"]
        mse_idx = condition["mse_loss_indexes"]
        return torch.isin(vae_idx, mse_idx)

    # ── FSDP2 sharding ─────────────────────────────────────────────────────

    def fully_shard(self, **kwargs):
        """Shard Bagel transformer layers for FSDP2.

        We shard ``self.bagel.language_model`` (the Qwen2 LLM) layer-by-layer
        rather than the KokoroNetwork wrapper itself, because the wrapper has
        complex multiple-inheritance with ABC that makes FSDP2's __class__
        reassignment fail.

        Root-param exclusion: the root ``fully_shard(self.bagel.language_model)`` call would
        otherwise sweep every param not in a nested layer shard — namely
        ``model.embed_tokens``, ``model.norm``, ``model.norm_moe_gen`` and the
        tied ``lm_head`` — into a sharded DTensor.  But the per-rank KV-cache
        build path (``build_inference_condition`` → ``inference_ctx`` →
        ``bagel.forward_cache_update_*``) calls ``embed_tokens`` and then
        ``language_model.forward_inference`` *directly*, never through
        ``language_model.__call__``, so the root FSDP2 pre-forward all-gather
        hook never fires.  ``embed_tokens.weight`` (DTensor) meets a plain
        ``packed_text_ids`` → ``aten.embedding got mixed torch.Tensor and
        DTensor``; the same hits ``norm``/``norm_moe_gen`` at the end of
        ``Qwen2Model.forward_inference``.  These modules are used only on the
        per-rank cache-build path (and as a frozen, replicated embedding in the
        training forward — the default recipe freezes everything but ``moe_gen``); they
        never participate in a cross-rank collective, so they do not need
        sharding.  Excluding them via ``ignored_params`` keeps them as plain
        replicated Tensors, making every direct call site DTensor-safe at once
        with no added collectives, while the transformer layers stay sharded.
        """
        from fastgen.networks.kokoro.qwen2_navit import (
            Qwen2DecoderLayer,
            Qwen2MoTDecoderLayer,
            Qwen2MoEDecoderLayer,
        )

        root = self.bagel.language_model  # Qwen2ForCausalLM
        llm = root.model  # Qwen2Model
        layer_types = (Qwen2DecoderLayer, Qwen2MoTDecoderLayer, Qwen2MoEDecoderLayer)

        apply_fsdp_checkpointing(llm, check_fn=lambda m: isinstance(m, layer_types))
        logger.info("Applied FSDP activation checkpointing to Bagel transformer layers")

        # Root non-layer params touched outside any FSDP ``__call__`` (the
        # per-rank cache-build / direct-forward_inference path). Keep them
        # unsharded (plain, replicated) so direct calls never hit a
        # DTensor/Tensor mismatch. lm_head.weight is tied to embed_tokens.weight
        # (same Parameter object); the set dedups, lm_head added defensively.
        ignored_params = set()
        ignored_params |= set(llm.embed_tokens.parameters())
        ignored_params |= set(llm.norm.parameters())
        if getattr(llm, "norm_moe_gen", None) is not None:
            ignored_params |= set(llm.norm_moe_gen.parameters())
        if getattr(root, "lm_head", None) is not None:
            ignored_params |= set(root.lm_head.parameters())
        logger.info(
            f"FSDP2: excluding {len(ignored_params)} root LLM params "
            f"(embed_tokens/norm/norm_moe_gen/lm_head) from sharding"
        )

        for layer in llm.layers:
            fully_shard(layer, **kwargs)
        if "ignored_params" in inspect.signature(fully_shard).parameters:
            fully_shard(root, ignored_params=ignored_params, **kwargs)
        else:
            # torch < 2.7 (e.g. 2.5.1) has no ignored_params.  Same
            # effect: detach the modules owning those params while the root is
            # wrapped (fully_shard collects managed params by walking children),
            # then re-attach them in their original order.  They stay plain
            # replicated tensors; the root FSDP group manages whatever else is
            # left at the root.  norm_moe_gen is trainable, so its grads need
            # the cross-rank average in ReplicatedGradSyncCallback.
            owners = [(llm, "embed_tokens"), (llm, "norm"), (llm, "norm_moe_gen"), (root, "lm_head")]
            orders = {id(parent): (parent, list(parent._modules.keys())) for parent, _ in owners}
            detached = {
                (id(parent), name): parent._modules.pop(name)
                for parent, name in owners
                if parent._modules.get(name) is not None
            }
            try:
                fully_shard(root, **kwargs)
            finally:
                for pid, (parent, order) in orders.items():
                    current = dict(parent._modules)
                    parent._modules.clear()
                    for name in order:
                        mod = detached.get((pid, name), current.get(name))
                        parent._modules[name] = mod
            assert all(not isinstance(p, DTensor) for p in ignored_params), "root LLM params got sharded"

    def init_vae(self) -> None:
        """Load the Kokoro autoencoder lazily for visualization/decoding."""
        if self.vae is not None:
            return
        if self.bagel_ckpt_dir is None:
            raise ValueError("KokoroNetwork.init_vae requires bagel_ckpt_dir to locate ae.safetensors")

        from fastgen.networks.kokoro.autoencoder import load_ae

        ae_path = os.path.join(self.bagel_ckpt_dir, "ae.safetensors")
        self.vae, _ = load_ae(local_path=ae_path)
        self.vae.eval().requires_grad_(False)

    def decode_packed_x0_first(self, packed_x0: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """Decode one image worth of packed x0 tokens to BCHW image tensor in [-1, 1]."""
        if self.vae is None:
            self.init_vae()

        h_tok, w_tok = shape
        patch_size = self.bagel.latent_patch_size
        latent_channels = self.bagel.latent_channel

        latent = packed_x0.reshape(1, h_tok, w_tok, patch_size, patch_size, latent_channels)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, latent_channels, h_tok * patch_size, w_tok * patch_size)

        vae_device = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        latent = latent.to(device=vae_device, dtype=vae_dtype)
        image = self.vae.decode(latent).to(dtype=packed_x0.dtype)
        return image

    # ── Inference-branch context / tokenizer ────────────────────────────────

    def _init_tokenizer(self) -> None:
        """Lazily load the tokenizer and special-token id map from the Bagel
        checkpoint dir, using the cfg-distill data pipeline's ``add_special_tokens``
        so the ids match the CFG-distill checkpoint vocab exactly."""
        if self.tokenizer is not None:
            return
        if self.bagel_ckpt_dir is None:
            raise ValueError(
                "KokoroNetwork inference-branch needs bagel_ckpt_dir to load the tokenizer"
            )
        from transformers import AutoTokenizer

        from fastgen.datasets.kokoro_cfgdistill_data import add_special_tokens

        tok = AutoTokenizer.from_pretrained(self.bagel_ckpt_dir, trust_remote_code=True)
        # split_diff_story=True (used by the 1024px data-pack adapter): registers <|story_mode|> / <|diff_mode|>,
        # whose ids are exactly the pack's MODE_TOKEN_IDS 151665 / 151666 (these two embedding rows are already trained
        # in the teacher checkpoint, norms 1.15 / 1.39). The i2i text "<|story_mode|>, xxx" then encodes as
        # [151665] + encode(", xxx"), token-for-token identical to how the pack was tokenized in training; unregistered,
        # each string splits into 6 ordinary tokens. Only text that literally contains these strings is affected;
        # cfg-distill prompts are unchanged.
        tok, new_token_ids, _ = add_special_tokens(tok, split_diff_story=True)
        self.tokenizer = tok
        self.new_token_ids = new_token_ids

    def _ensure_image_transforms(self) -> None:
        """Lazily create the VAE / VIT image transforms used by i2i context build.
        Values match the cfg-distill training transforms (ImageTransform(1024, 512, 16) for VAE,
        (980, 224, 14) for VIT) — do NOT change these without retraining the
        teacher, they are baked into the cfg-distill checkpoint."""
        if getattr(self, "_vae_transform", None) is not None:
            return
        from fastgen.datasets.kokoro_cfgdistill_data import ImageTransform
        self._vae_transform = ImageTransform(1024, 512, 16)
        self._vit_transform = ImageTransform(980, 224, 14)

    def build_inference_condition(
        self,
        sample: dict,
        device,
        cfg_text_scale: float = 1.0,
        cfg_img_scale: float = 1.0,
    ) -> dict:
        """Build the per-step inference condition (cond context + flow layout).

        Supports both T2I and I2I.  Runs in the trainer process on the current
        rank; the ``NaiveCache`` it creates never crosses a worker/rank boundary.

        Returns a dict consumed by ``forward``:
            task_type, image_shape, patchified_vae_latent_shapes,
            cond_ctx{past_key_values,kv_lens,ropes}, cond_layout(dict),
            cfg_text_scale, cfg_img_scale
        """
        from fastgen.networks.kokoro import inference_ctx as ictx

        self._ensure_rope_fp32()
        self._init_tokenizer()
        task_type = sample.get("task_type", "t2i")

        latent = sample["vae_latent"]
        p = self.bagel.latent_patch_size
        c, h_lat, w_lat = latent.shape
        h_tok, w_tok = h_lat // p, w_lat // p
        # pixel image shape = latent spatial size * vae_downsample;
        # bagel.latent_downsample = vae*patch.
        vae_downsample = self.bagel.latent_downsample // p
        image_shape = (h_lat * vae_downsample, w_lat * vae_downsample)

        if task_type == "t2i":
            cond_ctx = ictx.build_t2i_context(
                self.bagel, self.tokenizer, self.new_token_ids, device, sample["text"]
            )
        elif task_type == "i2i":
            if sample.get("source_image") is None:
                raise ValueError("i2i task_type requires non-None source_image")
            if self.vae is None:
                self.init_vae()
            self._ensure_image_transforms()
            cond_ctx = ictx.build_i2i_context(
                self.bagel, self.vae, self.tokenizer,
                self._vit_transform, self._vae_transform, self.new_token_ids,
                device, sample["text"], sample["source_image"],
            )
        else:
            raise ValueError(f"KokoroNetwork: unsupported task_type={task_type!r}")

        layout = ictx.prepare_flow_layout(
            self.bagel, cond_ctx, image_shape, self.new_token_ids
        )
        cond_layout = ictx.cond_layout_fields(layout, device)
        return {
            "task_type": task_type,
            "image_shape": image_shape,
            "patchified_vae_latent_shapes": [(h_tok, w_tok)],
            "cond_ctx": cond_ctx,
            "cond_layout": cond_layout,
            "cfg_text_scale": float(cfg_text_scale),
            "cfg_img_scale": float(cfg_img_scale),
        }

    # ── Config-driven trainable-parameter selection ─────────────────────────

    def apply_trainable_param_patterns(self, patterns) -> list[str]:
        """Set ``requires_grad`` on ``self.bagel`` params by name substring.

        Default DMD2 recipe: ``["moe_gen"]`` — student/fake_score never
        use the cfg embedders, so they stay frozen. Teacher is fully frozen
        elsewhere.  An empty list or a list containing ``""`` means "train
        everything" (whole-net escalation).

        Guard: enabling any LLM weight outside ``moe_gen`` / ``cfg_*_embedder``
        breaks the KV-cache sharing assumption (student's text-path KV is
        reused by teacher/fake_score; if student diverges, teacher sees
        corrupted KV with no error).  We raise instead of letting it through
        silently.

        Returns the sorted list of trainable parameter names (for logging).
        """
        train_all = (not patterns) or any(pat == "" for pat in patterns)
        trainable = []
        for name, param in self.bagel.named_parameters():
            want = train_all or any(pat in name for pat in patterns)
            param.requires_grad_(want)
            if want:
                trainable.append(name)

        # Runtime guard: trainable params that would corrupt the shared KV cache.
        # Skip the guard for whole-net escalation (`train_all`) since the caller
        # has explicitly opted out via empty patterns.
        if not train_all:
            forbidden = []
            for name in trainable:
                if "moe_gen" in name:
                    continue
                if "cfg_text_embedder" in name or "cfg_img_embedder" in name:
                    continue
                if any(tok in name for tok in (
                    "embed_tokens",
                    "language_model.model.layers",
                    "language_model.model.norm",
                    "vit_model",
                    "vae2llm", "llm2vae",
                    "latent_pos_embed",
                    "time_embedder",
                )):
                    forbidden.append(name)
            if forbidden:
                raise RuntimeError(
                    f"KokoroNetwork.apply_trainable_param_patterns: detected "
                    f"{len(forbidden)} trainable params that are neither moe_gen "
                    f"nor cfg_*_embedder. This breaks the shared-KV-cache "
                    f"assumption in DMD2 (student's text-path KV is reused by "
                    f"teacher/fake_score; if student diverges, teacher silently "
                    f"sees corrupted KV). Restrict patterns to ['moe_gen'] (default) "
                    f"or ['moe_gen', 'cfg_text_embedder', 'cfg_img_embedder'] "
                    f"(also train the cfg embedders). First 5 offending names: {forbidden[:5]}"
                )

        logger.info(
            f"KokoroNetwork trainable params: {len(trainable)} tensors "
            f"(patterns={'<all>' if train_all else list(patterns)})"
        )
        return sorted(trainable)

    # ── FastGenNetwork interface ────────────────────────────────────────────

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        role: str = "student",
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        cfg_renorm_type: str = "global",
        cfg_renorm_min: float = 0.0,
        pdd: bool = False,
        **fwd_kwargs,
    ) -> torch.Tensor:
        """Inference-branch forward: ``Bagel._forward_flow_train`` (cfg-embed).

        Args:
            x_t: (N_vae, patch_dim) noised packed VAE tokens (single image).
            t:   (N_vae,) per-token flow timestep in [0, 1].  All entries equal.
            condition: dict from ``build_inference_condition`` (includes
                ``cfg_text_scale`` and ``cfg_img_scale`` — only used when
                ``role == "teacher"``).
            role: one of {"student", "fake", "teacher"}.
                  - "teacher": uses cfg_embed with condition's cfg scales.  Teacher
                    is the frozen cfg-distill ckpt; its cfg_*_embedder produces a
                    high-quality velocity equivalent to running CFG=cfg_*_scale.
                  - "student" / "fake": uses cfg_embed=False.  Student
                    learns to absorb cfg knowledge into ``moe_gen`` main weights so
                    the final model has no cfg knob at inference.
            fwd_pred_type: "v" → velocity; "x0" → ``x_t - t*v``.  Default
                ``self.net_pred_type``.
            pdd: return all N parallel-decoder mean velocities, shape
                ``(N_vae, pdd_grid_steps, patch_dim)`` (head k = interval k of the
                PDD grid); student only, ignores fwd_pred_type / features.

        Returns:
            (N_vae, patch_dim) velocity or x0 prediction.
        """
        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        self._ensure_rope_fp32()

        # GAN: feature_indices may be a set or list (config gives a
        # list). Normalise to a set for the inference-branch collector.
        if feature_indices is not None:
            feature_indices = set(feature_indices)
        want_features = feature_indices is not None and len(feature_indices) > 0
        if return_features_early and not want_features:
            raise ValueError(
                "KokoroNetwork.forward: return_features_early=True requires non-empty feature_indices"
            )
        assert condition is not None and "cond_ctx" in condition, (
            "KokoroNetwork.forward requires the inference condition built by "
            "build_inference_condition (got a legacy/None condition)."
        )

        # Single-image / single-timestep invariant for _forward_flow_train.
        tv = t.reshape(-1)
        assert tv.numel() > 0 and bool((tv.max() == tv.min()).item()), (
            "KokoroNetwork.forward expects one timestep per forward (single "
            f"image); got {tv.float().unique().numel()} distinct t values."
        )
        timestep = tv[:1]

        is_i2i = (condition["task_type"] == "i2i")

        if role == "teacher":
            use_cfg_embed = True
            cfg_text = condition["cfg_text_scale"]
            cfg_img = condition["cfg_img_scale"]
            distill_text = torch.as_tensor(cfg_text, device=x_t.device, dtype=torch.float32)
            distill_img = (
                torch.as_tensor(cfg_img, device=x_t.device, dtype=torch.float32)
                if is_i2i else None
            )
            apply_img_embed = is_i2i
        elif role in ("student", "fake"):
            # student/fake never use cfg embedders.
            use_cfg_embed = False
            distill_text = None
            distill_img = None
            apply_img_embed = False
        else:
            raise ValueError(
                f"KokoroNetwork.forward: unknown role={role!r}; "
                f"expected one of 'student', 'fake', 'teacher'."
            )

        if pdd:
            assert self.pdd_grid_steps > 0 and role == "student" and not want_features, (
                "pdd=True needs pdd_grid_steps>0, role='student' and no feature request"
            )
        out = self.bagel._forward_flow_train(
            x_t=x_t,
            timestep=timestep,
            past_key_values=condition.get("cond_ctx_by_role", {}).get(role, condition["cond_ctx"])["past_key_values"],
            **condition["cond_layout"],
            cfg_text_scale=1.0,
            cfg_img_scale=1.0,
            cfg_renorm_type=cfg_renorm_type,
            cfg_renorm_min=cfg_renorm_min,
            use_cfg_embed=use_cfg_embed,
            distill_cfg_text_scale=distill_text,
            distill_cfg_img_scale=distill_img,
            cfg_text_max=self.cfg_text_max,
            cfg_img_max=self.cfg_img_max,
            apply_cfg_img_embed=apply_img_embed,
            feature_indices=feature_indices,
            return_features_early=return_features_early,
            return_vae_hidden=pdd,
        )
        if pdd:
            return self.bagel.pdd_heads(out).view(out.shape[0], self.pdd_grid_steps, -1)

        # GAN: features-only request -> unpatchify each (N_vae, hidden)
        # VAE-token feature into the spatial grid [1, hidden, h_tok, w_tok] the
        # Discriminator_ImageDiT consumes.
        if return_features_early:
            return self._unpatchify_features(out, condition)

        if want_features:
            v, vae_feats = out
            feats = self._unpatchify_features(vae_feats, condition)
        else:
            v = out

        if fwd_pred_type == "v":
            pred = v
        elif fwd_pred_type == "x0":
            # rectified flow: x0 = x_t - t * v
            pred = x_t - t.unsqueeze(-1) * v
        else:
            raise ValueError(
                f"KokoroNetwork: unsupported fwd_pred_type '{fwd_pred_type}'. "
                "Supported: 'v', 'x0'."
            )

        if want_features:
            return pred, feats
        return pred

    def _unpatchify_features(self, vae_feats, condition) -> list:
        """Reshape each (N_vae, hidden) VAE-token feature into the spatial grid
        ``[1, hidden, h_tok, w_tok]`` expected by ``Discriminator_ImageDiT``.

        Single-image invariant (one entry in patchified_vae_latent_shapes); the
        VAE token order is row-major (h_tok, w_tok) per ``patchify_vae_latent``
        (einsum ``chpwq->hwpqc``), matching ``decode_packed_x0_first``.
        """
        shapes = condition["patchified_vae_latent_shapes"]
        assert len(shapes) == 1, (
            f"KokoroNetwork feature unpatchify expects a single image; got {len(shapes)}"
        )
        h_tok, w_tok = shapes[0]
        spatial = []
        for feat in vae_feats:
            hidden = feat.shape[-1]
            assert feat.shape[0] == h_tok * w_tok, (
                f"feature/token mismatch: {feat.shape[0]} vs {h_tok}*{w_tok}"
            )
            grid = feat.reshape(h_tok, w_tok, hidden).permute(2, 0, 1).unsqueeze(0)
            spatial.append(grid.contiguous())
        return spatial
