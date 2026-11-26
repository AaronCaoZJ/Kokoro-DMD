# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# BAGEL model definition: BagelConfig and Bagel, with import paths
# adapted to the FastGen package layout.

import copy
from typing import List, Tuple, Optional, Dict, Any, Set
import tqdm
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA functionality will be disabled.")

# ── Kokoro data utilities ────────────────────────────────────────────────────
from .data_utils import (
    create_sparse_mask,
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    patchify,
)

# ── Kokoro internal imports ───────────────────────────────────────────────────
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from .modeling.cache_utils.taylorseer import cache_init


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        training_type: str = "naive", # naive and shortcut mode
        distill=False,
        # LoRA parameters
        use_lora=False,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=None,  # List of module names to apply LoRA
        _loading_from_checkpoint=False,
        consistency=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        # distill
        self.training_type = training_type
        self.distill = distill
        # if use consistency
        self.consistency = consistency
        # LoRA configuration
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "language_model.model.layers.*.self_attn.q_proj_moe_gen",
            "language_model.model.layers.*.self_attn.k_proj_moe_gen",
            "language_model.model.layers.*.self_attn.v_proj_moe_gen",
            "language_model.model.layers.*.self_attn.o_proj_moe_gen",
            "language_model.model.layers.*.mlp_moe_gen.gate_proj",
            "language_model.model.layers.*.mlp_moe_gen.up_proj",
            "language_model.model.layers.*.mlp_moe_gen.down_proj"
        ]
        self._loading_from_checkpoint=_loading_from_checkpoint



class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads
        self.training_type = config.training_type
        self.distill = config.distill
        self.consistency = config.consistency
        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            # CFG distillation: learnable text/img cfg-scale embedders, injected
            # into the VAE latent so CFG strength is delivered by a single
            # conditional forward (no cond/uncond two-pass).
            self.cfg_text_embedder = TimestepEmbedder(self.hidden_size)
            self.cfg_img_embedder = TimestepEmbedder(self.hidden_size)
            # if self.training_type == "shortcut":
            #     self.dt_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

        # Apply LoRA if enabled (but not when loading from checkpoint)
        if config.use_lora and PEFT_AVAILABLE and not getattr(config, '_loading_from_checkpoint', False):
            self._apply_lora(config)

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)
            # Zero-init cfg embedders so they start neutral (no effect on the
            # base model until CFG-distill training moves them).  TimestepEmbedder's
            # last layer is mlp[2].
            for emb in (self.cfg_text_embedder, self.cfg_img_embedder):
                nn.init.constant_(emb.mlp[2].weight, 0)
                nn.init.constant_(emb.mlp[2].bias, 0)

    def _apply_lora(self, config):
        """Apply LoRA to language model generation branch and freeze other parameters"""
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available, skipping LoRA application")
            return

        # Create LoRA config for generation branch
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[
                "self_attn.q_proj_moe_gen",
                "self_attn.k_proj_moe_gen",
                "self_attn.v_proj_moe_gen",
                "self_attn.o_proj_moe_gen",
                "mlp_moe_gen.gate_proj",
                "mlp_moe_gen.up_proj",
                "mlp_moe_gen.down_proj"
            ],
            bias="none",
        )

        # Apply LoRA to language model generation branch
        if hasattr(self, 'language_model'):
            # self.language_model.add_adapter(lora_config)
            # self.language_model.enable_adapters()
            self.language_model = get_peft_model(self.language_model, lora_config)
            print("Applied LoRA to language_model generation branch (attention *_proj_moe_gen and mlp_moe_gen layers)")
        print(f"LoRA applied to language model only with rank={config.lora_rank}, alpha={config.lora_alpha}")

        # Call separate freezing method
        self._freeze_non_lora_params()
        print("All other Bagel components have been frozen")

    def _freeze_non_lora_params(self):
        """Freeze non-LoRA parameters (separate method for reuse)"""
        config = self.config

        # Manually freeze all other parameters (non-LoRA components)
        if config.visual_gen:
            # Freeze VAE connector layers
            if hasattr(self, 'vae2llm'):
                for param in self.vae2llm.parameters():
                    param.requires_grad = False
                print("Frozen vae2llm parameters")

            if hasattr(self, 'llm2vae'):
                for param in self.llm2vae.parameters():
                    param.requires_grad = False
                print("Frozen llm2vae parameters")

            # Freeze time embedders
            if hasattr(self, 'time_embedder'):
                for param in self.time_embedder.parameters():
                    param.requires_grad = False
                print("Frozen time_embedder parameters")

            if hasattr(self, 'dt_embedder'):
                for param in self.dt_embedder.parameters():
                    param.requires_grad = False
                print("Frozen dt_embedder parameters")

            # Freeze position embeddings
            if hasattr(self, 'latent_pos_embed'):
                for param in self.latent_pos_embed.parameters():
                    param.requires_grad = False
                print("Frozen latent_pos_embed parameters")

        if config.visual_und:
            # Freeze VIT connector and position embeddings
            if hasattr(self, 'connector'):
                for param in self.connector.parameters():
                    param.requires_grad = False
                print("Frozen connector parameters")

            if hasattr(self, 'vit_pos_embed'):
                for param in self.vit_pos_embed.parameters():
                    param.requires_grad = False
                print("Frozen vit_pos_embed parameters")

    def _reshape_hidden_features_for_discriminator(
        self,
        hidden_features: List[torch.Tensor],
        packed_vae_token_indexes: torch.LongTensor,
        patchified_vae_latent_shapes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """Convert packed hidden states into padded image feature maps for GAN.

        Each feature tensor is gathered at VAE-token positions, split back into
        per-image latent grids, reshaped to ``[B, C, H, W]``, and padded to the
        largest latent size in the packed batch so the discriminator can stack them.
        """
        if len(hidden_features) == 0:
            return []

        max_h = max(h for h, _ in patchified_vae_latent_shapes)
        max_w = max(w for _, w in patchified_vae_latent_shapes)
        image_features = []

        for hidden in hidden_features:
            vae_hidden = hidden[packed_vae_token_indexes]
            per_image = []
            offset = 0
            for h_tok, w_tok in patchified_vae_latent_shapes:
                num_tokens = h_tok * w_tok
                img_feat = vae_hidden[offset: offset + num_tokens].reshape(h_tok, w_tok, -1).permute(2, 0, 1)
                if h_tok != max_h or w_tok != max_w:
                    img_feat = F.pad(img_feat, (0, max_w - w_tok, 0, max_h - h_tok))
                per_image.append(img_feat)
                offset += num_tokens
            image_features.append(torch.stack(per_image, dim=0))

        return image_features

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        split_groups: List = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        packed_dt_steps: Optional[torch.LongTensor] = None,
        packed_dt_steps_bts: Optional[torch.LongTensor] = None,
        packed_dt_bts: Optional[torch.LongTensor] = None,
        bts_loss_indexes: Optional[torch.BoolTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        # DMD-specific parameters
        packed_latent_input: Optional[torch.Tensor] = None,
        use_cfg: bool = False,
        cfg_text_scale: float = 4.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        # uncond_* fields for CFG (passed through to _forward_with_cfg)
        **extra_kwargs,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        # ── DMD CFG early dispatch ──────────────────────────────────────────
        if use_cfg and packed_latent_input is not None:
            return self._forward_with_cfg(
                sequence_length=sequence_length,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                sample_lens=sample_lens,
                packed_position_ids=packed_position_ids,
                nested_attention_masks=nested_attention_masks,
                split_lens=split_lens,
                attn_modes=attn_modes,
                split_groups=split_groups,
                ce_loss_indexes=ce_loss_indexes,
                packed_label_ids=packed_label_ids,
                packed_vit_tokens=packed_vit_tokens,
                packed_vit_token_indexes=packed_vit_token_indexes,
                packed_vit_position_ids=packed_vit_position_ids,
                vit_token_seqlens=vit_token_seqlens,
                padded_latent=padded_latent,
                patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                packed_latent_position_ids=packed_latent_position_ids,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_timesteps=packed_timesteps,
                packed_dt_steps=packed_dt_steps,
                packed_dt_steps_bts=packed_dt_steps_bts,
                packed_dt_bts=packed_dt_bts,
                bts_loss_indexes=bts_loss_indexes,
                mse_loss_indexes=mse_loss_indexes,
                packed_latent_input=packed_latent_input,
                cfg_text_scale=cfg_text_scale,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                **extra_kwargs,  # passes uncond_* fields to _forward_with_cfg
            )

        # ── DMD fast path (FSDP2 safe) ─────────────────────────────────────────
        # When language_model is FSDP2-wrapped, embed_tokens.weight is a DTensor.
        # Calling embed_tokens directly (outside language_model.forward) bypasses
        # FSDP2's pre-forward all-gather, causing a DTensor/Tensor mismatch error.
        # Fix: build image embeddings first, then pass packed_text_ids to
        # language_model so it calls embed_tokens internally (inside FSDP context).
        if self.config.visual_gen and packed_latent_input is not None:
            # Build flex attention block_mask from global split_lens/attn_modes.
            # sample_lens already includes a padding entry so sum(sample_lens) is
            # a multiple of BLOCK_SIZE=128 (added by KokoroPackedDataset._to_tensor).
            device = packed_latent_input.device
            seqlen = sum(sample_lens)
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, device)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen,
                device=device, BLOCK_SIZE=128, _compile=True,
            )
            attention_mask = block_mask

            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb   = self.latent_pos_embed(packed_latent_position_ids)
            vae_embed              = self.vae2llm(packed_latent_input)
            latent_embed           = vae_embed + packed_timestep_embeds + latent_token_pos_emb

            packed_sequence = latent_embed.new_zeros(sequence_length, self.hidden_size)
            packed_sequence[packed_vae_token_indexes] = latent_embed

            extra_inputs = {}
            if self.use_moe:
                extra_inputs.update(
                    packed_und_token_indexes=packed_text_indexes,
                    packed_gen_token_indexes=packed_vae_token_indexes,
                )

            last_hidden_state = self.language_model(
                packed_sequence=packed_sequence,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                feature_indices=feature_indices,
                return_features_early=return_features_early,
                **extra_inputs,
            )
            if return_features_early:
                return self._reshape_hidden_features_for_discriminator(
                    last_hidden_state,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                )
            if feature_indices is not None and len(feature_indices) > 0:
                last_hidden_state, hidden_features = last_hidden_state
                features = self._reshape_hidden_features_for_discriminator(
                    hidden_features,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                )
            else:
                features = None
            v_pred = self.llm2vae(last_hidden_state[mse_loss_indexes])
            out = dict(v_pred=v_pred)
            if features is not None:
                out["features"] = features
            return out

        # ── Standard (non-DMD) path ─────────────────────────────────────────────
        if self.config.use_lora:
            # Access embed_tokens through LoRA wrapper
            packed_text_embedding = self.language_model.model.model.embed_tokens(packed_text_ids)
        else:
            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen,
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und and packed_vit_tokens is not None:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(dtype=torch.int32, device=vit_token_seqlens.device)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            # make sure the dtypes match
            if packed_vit_token_embed.dtype != packed_sequence.dtype:
                packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            if packed_latent_input is not None:
                # ── DMD path: latent already patchified and noised externally ──
                # packed_timesteps are already in [0,1] — skip sigmoid/shift
                packed_timestep_embeds = self.time_embedder(packed_timesteps)
                latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
                vae_embed = self.vae2llm(packed_latent_input)
                latent_embed = vae_embed + packed_timestep_embeds + latent_token_pos_emb
                if latent_embed.dtype != packed_sequence.dtype:
                    latent_embed = latent_embed.to(packed_sequence.dtype)
                packed_sequence[packed_vae_token_indexes] = latent_embed
            else:
                # ── Standard training path ─────────────────────────────────────
                p = self.latent_patch_size
                packed_latent = []
                for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                    latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                    latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                    packed_latent.append(latent)
                packed_latent_clean = torch.cat(packed_latent, dim=0)

                noise = torch.randn_like(packed_latent_clean)
                if self.training_type == "naive":
                    packed_timesteps = torch.sigmoid(packed_timesteps)
                    if self.distill:
                        packed_timesteps = 0.8 + 0.19 * packed_timesteps  # map to [0.8, 0.99]
                        initial_timesteps = packed_timesteps.clone()
                    packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
                elif self.training_type =="shortcut":
                    assert packed_dt_steps is not None
                    packed_sequence_bts = packed_sequence.clone()
                    packed_dt_embeds = self.time_embedder(packed_dt_steps)
                try:
                    packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
                except:
                    print("packed timesteps.shape:", packed_timesteps.shape)
                    print("noise.shape:", noise.shape)
                    print("packed latent clean.shape:", packed_latent_clean.shape)
                if self.training_type == "shortcut":
                    packed_latent_bts = copy.deepcopy(packed_latent)
                packed_timestep_embeds = self.time_embedder(packed_timesteps)
                latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
                vae_embed = self.vae2llm(packed_latent)

                packed_latent = vae_embed + packed_timestep_embeds + latent_token_pos_emb
                if self.training_type == "shortcut":
                    packed_latent = packed_latent + packed_dt_embeds

                # make sure the dtypes match
                if packed_latent.dtype != packed_sequence.dtype:
                    packed_latent = packed_latent.to(packed_sequence.dtype)
                packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        # ── DMD path: run LLM forward and return v_pred ────────────────────
        if packed_latent_input is not None:
            last_hidden_state = self.language_model(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                feature_indices=feature_indices,
                return_features_early=return_features_early,
                **extra_inputs,
            )
            if return_features_early:
                return self._reshape_hidden_features_for_discriminator(
                    last_hidden_state,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                )
            if feature_indices is not None and len(feature_indices) > 0:
                last_hidden_state, hidden_features = last_hidden_state
                features = self._reshape_hidden_features_for_discriminator(
                    hidden_features,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    patchified_vae_latent_shapes=patchified_vae_latent_shapes,
                )
            else:
                features = None
            v_pred = self.llm2vae(last_hidden_state[mse_loss_indexes])
            out = dict(v_pred=v_pred)
            if features is not None:
                out["features"] = features
            return out

        if not self.distill and not self.consistency and self.config.visual_gen:
            last_hidden_state = self.language_model(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                **extra_inputs,
            )

            mse = None
            if self.config.visual_gen:
                packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
                target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise

                if self.training_type == "naive":
                    has_mse = packed_timesteps > 0
                    try:
                        mse = (packed_mse_preds - target[has_mse]) ** 2
                    except Exception as e:
                        # write the error details to a file
                        import os
                        import datetime
                        error_log_dir = "error_logs"
                        os.makedirs(error_log_dir, exist_ok=True)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        error_file = os.path.join(error_log_dir, f"shape_mismatch_{timestamp}.txt")

                        with open(error_file, "w") as f:
                            f.write("="*80 + "\n")
                            f.write(f"ERROR: Shape mismatch in MSE computation (naive mode)\n")
                            f.write(f"Timestamp: {timestamp}\n")
                            f.write(f"Exception: {e}\n")
                            f.write("-"*80 + "\n")
                            f.write(f"packed_mse_preds.shape: {packed_mse_preds.shape}\n")
                            f.write(f"target.shape: {target.shape}\n")
                            f.write(f"has_mse.shape: {has_mse.shape}\n")
                            f.write(f"has_mse sum (num True): {has_mse.sum().item()}\n")
                            f.write(f"target[has_mse].shape: {target[has_mse].shape}\n")
                            f.write("-"*80 + "\n")
                            f.write(f"mse_loss_indexes: {mse_loss_indexes.cpu().tolist() if torch.is_tensor(mse_loss_indexes) else list(mse_loss_indexes)}\n")
                            f.write(f"mse_loss_indexes length: {len(mse_loss_indexes)}\n")
                            f.write("-"*80 + "\n")
                            f.write(f"packed_vae_token_indexes: {packed_vae_token_indexes.cpu().tolist() if torch.is_tensor(packed_vae_token_indexes) else list(packed_vae_token_indexes)}\n")
                            f.write(f"packed_vae_token_indexes length: {len(packed_vae_token_indexes)}\n")
                            f.write("-"*80 + "\n")
                            f.write(f"has_mse values: {has_mse.cpu().tolist()}\n")
                            f.write(f"packed_timesteps: {packed_timesteps.cpu().tolist()}\n")
                            f.write("="*80 + "\n")

                        # print a short error message
                        print(f"\n{'='*80}")
                        print(f"ERROR: Shape mismatch in MSE computation (naive mode)!")
                        print(f"Error details saved to: {error_file}")
                        print(f"packed_mse_preds.shape: {packed_mse_preds.shape}")
                        print(f"target[has_mse].shape: {target[has_mse].shape}")
                        print(f"{'='*80}\n")

                        # return None and let the caller handle it
                        return dict(mse=None, ce=None, skip_batch=True)

                elif self.training_type == "shortcut":
                    assert packed_dt_bts is not None
                    # convert mse_loss_indexes to positions relative to packed_vae_token_indexes
                    vae_mask = torch.isin(packed_vae_token_indexes, mse_loss_indexes)
                    target_mse = target[vae_mask]
                    mse = (packed_mse_preds - target_mse) ** 2

            ce = None
            if ce_loss_indexes is not None:
                if self.config.use_lora:
                    packed_ce_preds = self.language_model.model.model.lm_head(last_hidden_state[ce_loss_indexes])
                else:
                    packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
                ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")
        elif self.distill:
            num_inference_steps = 4
            # decreasing step size, from the initial value down to near 0
            old_shifted_timesteps = packed_timesteps.clone()
            # final_timesteps = torch.zeros_like(packed_timesteps) + 0.01  # close to 0 but not 0
            timestep_step = (torch.ones_like(initial_timesteps) - torch.zeros_like(packed_timesteps)) / num_inference_steps
            # timestep_step = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            for progressive_id, timestep in enumerate(range(num_inference_steps-1)):
                if self.config.use_lora:
                    # For LoRA wrapped model, call the model directly
                    last_hidden_state = self.language_model.model(
                        packed_sequence=packed_sequence,
                        sample_lens=sample_lens,
                        attention_mask=attention_mask,
                        packed_position_ids=packed_position_ids,
                        **extra_inputs,
                    )
                else:
                    last_hidden_state = self.language_model(
                        packed_sequence=packed_sequence,
                        sample_lens=sample_lens,
                        attention_mask=attention_mask,
                        packed_position_ids=packed_position_ids,
                        **extra_inputs,
                    )
                packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
                # timestep_step at the mse_loss_indexes positions within packed_vae_token_indexes
                initial_timesteps = initial_timesteps - timestep_step  # creates a new tensor, avoiding view issues
                new_shifted_timesteps = self.timestep_shift * initial_timesteps / (1 + (self.timestep_shift - 1) * initial_timesteps)
                relevant_timestep_step = old_shifted_timesteps - new_shifted_timesteps
                vae_mask = torch.isin(packed_vae_token_indexes, mse_loss_indexes)
                relevant_timestep_step = relevant_timestep_step[vae_mask]

                packed_latent = self.llm2vae(packed_sequence[mse_loss_indexes]) - relevant_timestep_step[:, None] * packed_mse_preds 
                packed_sequence[mse_loss_indexes] = self.vae2llm(packed_latent)
                new_packed_timestep_embeds = self.time_embedder(new_shifted_timesteps)
                # avoid in-place ops: compute first, then assign
                updated_vae_embeddings = packed_sequence[packed_vae_token_indexes] + new_packed_timestep_embeds + latent_token_pos_emb
                packed_sequence[packed_vae_token_indexes] = updated_vae_embeddings
                old_shifted_timesteps = new_shifted_timesteps
                # free temporaries explicitly to save GPU memory
                del last_hidden_state, packed_mse_preds, vae_mask, relevant_timestep_step
                del new_packed_timestep_embeds, updated_vae_embeddings
                torch.cuda.empty_cache()
            vae_mask = torch.isin(packed_vae_token_indexes, mse_loss_indexes)
            target = packed_latent_clean[vae_mask]
            mse = (packed_latent - target) ** 2
            ce = None

        elif self.consistency:
            last_hidden_state = self.language_model(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                **extra_inputs,
            )

            
        if self.config.visual_gen and self.training_type == "shortcut":
            # the bootstrap indexes must be tracked
            bts = None
            hidden_states_for_mse = last_hidden_state[bts_loss_indexes]
            packed_mse_preds = self.llm2vae(hidden_states_for_mse)
            if bts_loss_indexes is not None:
                
                assert packed_dt_steps_bts is not None
                with torch.no_grad():
                    packed_dt_embeds_bts = self.time_embedder(packed_dt_steps_bts)
                    packed_latent_bts = self.vae2llm(packed_latent_bts) + packed_timestep_embeds + latent_token_pos_emb + packed_dt_embeds_bts
                    # make sure the dtypes match
                    if packed_latent_bts.dtype != packed_sequence_bts.dtype:
                        packed_latent_bts = packed_latent_bts.to(packed_sequence_bts.dtype)
                    packed_sequence_bts[packed_vae_token_indexes] = packed_latent_bts
                    if self.config.use_lora:
                        # For LoRA wrapped model, call the model directly
                        last_hidden_state_bts_1 = self.language_model.model(
                            packed_sequence=packed_sequence_bts,
                            sample_lens=sample_lens,
                            attention_mask=attention_mask,
                            packed_position_ids=packed_position_ids,
                            **extra_inputs,
                        )
                    else:
                        last_hidden_state_bts_1 = self.language_model(
                            packed_sequence=packed_sequence_bts,
                            sample_lens=sample_lens,
                            attention_mask=attention_mask,
                            packed_position_ids=packed_position_ids,
                            **extra_inputs,
                        )
                    # packed_bts_preds_1 = self.llm2vae(last_hidden_state_bts_1[bts_loss_indexes])
                    packed_bts_preds_target = self.llm2vae(last_hidden_state_bts_1[bts_loss_indexes])
                    # TODO: double-check this computation
                    vae_mask = torch.isin(packed_vae_token_indexes, bts_loss_indexes)
                    relevant_dt_bts = packed_dt_bts[vae_mask]
                    # expand relevant_dt_bts to match the shape of packed_bts_preds_1
                    relevant_dt_bts = relevant_dt_bts.unsqueeze(-1)
                    packed_bts_x1 = self.llm2vae(packed_sequence_bts[bts_loss_indexes]) - relevant_dt_bts * packed_bts_preds_target
                    packed_timesteps_bts = packed_timesteps - packed_dt_bts
                    packed_timesteps_embeds_bts = self.time_embedder(packed_timesteps_bts)
                    packed_sequence_bts[bts_loss_indexes] = self.vae2llm(packed_bts_x1)
                    packed_sequence_bts[packed_vae_token_indexes] = packed_sequence_bts[packed_vae_token_indexes] + packed_timesteps_embeds_bts + latent_token_pos_emb + packed_dt_embeds_bts
                    if self.config.use_lora:
                        # For LoRA wrapped model, call the model directly
                        last_hidden_state_bts_2 = self.language_model.model(
                            packed_sequence=packed_sequence_bts,
                            sample_lens=sample_lens,
                            attention_mask=attention_mask,
                            packed_position_ids=packed_position_ids,
                            **extra_inputs,
                        )
                    else:
                        last_hidden_state_bts_2 = self.language_model(
                            packed_sequence=packed_sequence_bts,
                            sample_lens=sample_lens,
                            attention_mask=attention_mask,
                            packed_position_ids=packed_position_ids,
                            **extra_inputs,
                        )
                    packed_bts_preds_2 = self.llm2vae(last_hidden_state_bts_2[bts_loss_indexes])
                    # packed_bts_x2 = packed_sequence_bts[packed_vae_token_indexes] + packed_dt_bts[:, None] * packed_bts_preds_2
                    packed_bts_preds = (packed_bts_preds_target + packed_bts_preds_2) / 2
                    
                    # explicitly free the large shortcut-mode intermediates
                    del packed_sequence_bts, last_hidden_state_bts_1, last_hidden_state_bts_2
                    del packed_bts_preds_target, packed_bts_preds_2
                    del packed_bts_x1, packed_timesteps_embeds_bts, packed_dt_embeds_bts
                bts = (packed_mse_preds - packed_bts_preds) ** 2
        if self.training_type == "naive":
            return dict(mse=mse, ce=ce)
        elif self.training_type == "shortcut":
            return dict(mse=mse, ce=ce, bts=bts)
        else:
            raise NotImplementedError

    def _forward_with_cfg(
        self,
        packed_latent_input: torch.Tensor,
        cfg_text_scale: float,
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        **kwargs,
    ) -> dict:
        """
        DMD CFG forward: two passes (conditional + unconditional), combined.

        Conditional pass  : standard forward with text tokens.
        Unconditional pass: text-free sequence built at packing time
                            (``uncond_*`` fields in kwargs), matching
                            training-time CFG dropout exactly — no text slot
                            in the sequence, not just zeroed embeddings.

        Returns dict with:
            velocity_pred  : CFG-combined velocity (N_loss_tokens, patch_dim)
            v_cond         : conditional velocity (for diagnostics)
        """
        device = packed_latent_input.device

        # ── Conditional pass (strip uncond_* before calling forward) ───────
        cond_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("uncond_")}
        out_cond = self.forward(
            packed_latent_input=packed_latent_input, use_cfg=False, **cond_kwargs
        )
        if kwargs.get("return_features_early", False):
            # GAN feature extraction only needs the conditional path.
            return out_cond
        v_cond = out_cond["v_pred"]
        features = out_cond.get("features")

        # ── Unconditional pass: swap in pre-computed uncond layout ──────────
        # This matches training-time CFG dropout: the text segment is absent
        # from the sequence entirely, so sequence_length is shorter and all
        # indexes are re-numbered accordingly.
        uncond_kwargs = dict(cond_kwargs)
        uncond_kwargs["packed_text_ids"]           = kwargs["uncond_packed_text_ids"]
        uncond_kwargs["packed_text_indexes"]       = kwargs["uncond_packed_text_indexes"]
        uncond_kwargs["sequence_length"]           = kwargs["uncond_sequence_length"]
        uncond_kwargs["sample_lens"]               = kwargs["uncond_sample_lens"]
        uncond_kwargs["split_lens"]                = kwargs["uncond_split_lens"]
        uncond_kwargs["attn_modes"]                = kwargs["uncond_attn_modes"]
        uncond_kwargs["nested_attention_masks"]    = None
        uncond_kwargs["packed_position_ids"]       = kwargs["uncond_packed_position_ids"]
        uncond_kwargs["packed_vae_token_indexes"]  = kwargs["uncond_packed_vae_token_indexes"]
        uncond_kwargs["mse_loss_indexes"]          = kwargs["uncond_mse_loss_indexes"]
        out_uncond = self.forward(
            packed_latent_input=packed_latent_input, use_cfg=False, **uncond_kwargs
        )
        v_uncond = out_uncond["v_pred"]

        # ── CFG combine ────────────────────────────────────────────────────
        v_cfg = v_uncond + cfg_text_scale * (v_cond - v_uncond)

        # ── Optional renorm ────────────────────────────────────────────────
        if cfg_renorm_min > 0.0:
            cond_norm = v_cond.norm(dim=-1, keepdim=True)
            cfg_norm  = v_cfg.norm(dim=-1, keepdim=True)
            if cfg_renorm_type == "global":
                scale = (cond_norm / cfg_norm.clamp(min=1e-8)).clamp(min=cfg_renorm_min)
            else:
                scale = cfg_renorm_min
            v_cfg = v_cfg * scale

        out = dict(velocity_pred=v_cfg, v_cond=v_cond)
        if features is not None:
            out["features"] = features
        return out

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids, token_weights=None):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()
        packed_token_weights = list()  # Store weights for each token

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))

            # Map token weights: account for BOS token at position 0
            # token_weights keys are relative to the encoded prompt (without BOS/EOS)
            # packed_token_weights should be aligned with packed_text_ids (with BOS/EOS)
            if token_weights is not None and len(token_weights) > 0:
                # Start with weight 1.0 for BOS token
                current_weights = [1.0]
                # Add weights for the actual prompt tokens (offset by BOS)
                for i in range(len(text_ids) - 2):  # Exclude BOS and EOS
                    weight = token_weights.get(i, 1.0)
                    current_weights.append(weight)
                # Add weight 1.0 for EOS token
                current_weights.append(1.0)
                packed_token_weights.extend(current_weights)
            else:
                # No weighting, all weights are 1.0
                packed_token_weights.extend([1.0] * len(text_ids))

            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_token_weights": torch.tensor(packed_token_weights, dtype=torch.float32),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_token_weights: Optional[torch.FloatTensor] = None,
    ):
        if self.config.use_lora:
            packed_text_embedding = self.language_model.model.model.embed_tokens(packed_text_ids)
        else:
            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        # Apply token weights to embeddings if provided
        if packed_token_weights is not None:
            # Reshape weights to match embedding dimensions: [num_tokens] -> [num_tokens, 1]
            # Important: cast both device and dtype, otherwise float32 * bfloat16 promotes to float32
            weights = packed_token_weights.unsqueeze(-1).to(
                device=packed_text_embedding.device, 
                dtype=packed_text_embedding.dtype
            )
            # Apply weights element-wise
            packed_text_embedding = packed_text_embedding * weights
        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        # Route through __call__ (not .forward_inference) so the root FSDP2
        # module runs first and lazily-inits the whole layer tree in order;
        # FastGen Qwen.forward dispatches to forward_inference (no
        # `packed_sequence` kwarg). Calling .forward_inference directly here
        # bypasses the root and trips FSDP's "already lazily initialized /
        # run forward through the root module first" guard.
        output = self.language_model(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        if self.config.use_lora:
            packed_text_embedding = self.language_model.model.model.embed_tokens(packed_text_ids)
        else:
            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(dtype=torch.int32, device=vit_token_seqlens.device)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        # Route through __call__ (root FSDP2 first); see forward_cache_update_text.
        output = self.language_model(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        if self.config.use_lora:
            packed_text_embedding = self.language_model.model.model.embed_tokens(packed_text_ids)
        else:
            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # The frozen VAE keeps fp32 params and is loaded on CPU by init_vae
        # (never moved to GPU). update_context_image also wraps this call in a
        # bf16 autocast (for the LLM cache forward), which would feed bf16 conv
        # input against the VAE's fp32 bias. Lazily move the (small, frozen) VAE
        # to the compute device and run it in its native fp32 with autocast off
        # (same contract as decode_packed_x0_first). The resulting latent stays
        # on-device and is reconciled to packed_sequence.dtype below.
        vae_model.to(padded_images.device)
        vae_param = next(vae_model.parameters())
        with torch.amp.autocast("cuda", enabled=False):
            padded_latent = vae_model.encode(
                padded_images.to(device=vae_param.device, dtype=vae_param.dtype)
            )

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        # Route through __call__ (root FSDP2 first); see forward_cache_update_text.
        output = self.language_model(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # shortcut
        is_shortcut: bool = False,
        dt_base: float = 7.0,
        # cache_args
        fresh_threshold=3,
        max_order=6,
        first_enhance=10,
    ):
        x_t = packed_init_noises

        if not self.training_type == "shortcut":
            timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
            timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        else:
            timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                # shortcut
                is_shortcut=is_shortcut,
                dt_base=dt_base,
                # cache
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # shortcut
        is_shortcut: bool = False,
        dt_base: float = 7.0,
        # cache
        model_pred_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_current: Optional[int] = None,
        model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_text_current: Optional[int] = None,
        model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_img_current: Optional[int] = None,
    ):
        if self.config.use_lora:
            packed_text_embedding = self.language_model.model.model.embed_tokens(packed_text_ids)
        else:
            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.float().unique().shape[0] == 1  # .float(): torch 2.5.1 has no CUDA unique for bf16
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if self.training_type == "shortcut" and is_shortcut:
            packed_dt_embeds = self.time_embedder(torch.tensor([dt_base], device=x_t.device))
            x_t = x_t + packed_dt_embeds
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }
        if self.config.use_lora:
            model_obj = self.language_model.model
        else:
            model_obj = self.language_model.model
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]
        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            # if model_obj.enable_taylorseer:
            #     model_obj.cache_dic = model_pred_img_cache_dic
            #     model_obj.current = model_pred_img_current
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def _forward_flow_train(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # CFG distillation: inject text/img cfg scale as learned embeddings
        use_cfg_embed: bool = False,
        distill_cfg_text_scale: Optional[torch.Tensor] = None,
        distill_cfg_img_scale: Optional[torch.Tensor] = None,
        cfg_text_max: float = 5.0,
        cfg_img_max: float = 2.5,
        apply_cfg_img_embed: bool = True,  # t2i: False, i2i: True
        # GAN (DMD2): collect intermediate per-layer features for the
        # discriminator. feature_indices selects layers; the returned features
        # are reduced to VAE tokens only ((N_vae, hidden) per layer).
        feature_indices: Optional[Set[int]] = None,
        return_features_early: bool = False,
        # PDD (parallel decoding): return the final hidden state of the VAE
        # tokens instead of llm2vae(hidden); the caller applies its own heads.
        return_vae_hidden: bool = False,
    ):
        """Trainable (grad-enabled) flow forward with cfg-embed injection.

        Differs from ``_forward_flow`` (kept unchanged, still
        ``@torch.no_grad``, used by ``generate_image`` / inference scripts):
          * NO ``@torch.no_grad`` — student/fake must back-prop.
          * Adds cfg-embed injection (use_cfg_embed / distill_cfg_*_scale /
            cfg_*_max / apply_cfg_img_embed).
          * Calls ``self.language_model(...)`` via ``__call__`` (FastGen Qwen
            dispatches to ``forward_inference`` when no ``packed_sequence``
            kwarg is present) so FSDP2 ``__call__`` hooks fire — instead of
            calling ``.forward_inference(...)`` directly.

        FSDP2-compatible implementation details (no FSDP1-only helpers):
          * token embedding via ``self.language_model[.model].model.embed_tokens``
            (same pattern as ``forward_cache_update_text``) rather than an
            FSDP1 ``summon_full_params`` wrapper.
          * compute dtype is ``next(self.time_embedder.parameters()).dtype``.
          * the taylorseer cache path is omitted.  It is dead code during
            training (``enable_taylorseer`` defaults False), so the omission
            is numerically neutral; ``self.language_model`` is used directly.

        With the default cfg_text_scale=cfg_img_scale=1.0 (DMD2 uses cfg-embed
        only), the cond/uncond multi-branch CFG below is dead code; it is kept
        unchanged for numerical parity with the CFG-distillation forward and
        for future i2i use.
        """
        if self.config.use_lora:
            packed_text_embedding = self.language_model.model.model.embed_tokens(packed_text_ids)
        else:
            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        seq_len = sum(packed_seqlens)
        packed_sequence = packed_text_embedding.new_zeros((seq_len, self.hidden_size))
        assert packed_text_indexes.max() < seq_len, (
            f"text_indexes OOB: max={packed_text_indexes.max()} seq_len={seq_len}"
        )
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.float().unique().shape[0] == 1  # .float(): torch 2.5.1 has no CUDA unique for bf16
        compute_dtype = next(self.time_embedder.parameters()).dtype
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep.to(compute_dtype))
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed

        if use_cfg_embed:
            # Normalize cfg scales from [1, cfg_*_max] -> [0, 1]
            cfg_text_norm = (distill_cfg_text_scale - 1.0) / (cfg_text_max - 1.0)
            cfg_text_emb = self.cfg_text_embedder(cfg_text_norm.view(1).to(compute_dtype))
            x_t = x_t + cfg_text_emb
            if apply_cfg_img_embed:
                # keeps i2i and t2i separate (img embed only for i2i)
                cfg_img_norm = (distill_cfg_img_scale - 1.0) / (cfg_img_max - 1.0)
                cfg_img_emb = self.cfg_img_embedder(cfg_img_norm.view(1).to(compute_dtype))
                x_t = x_t + cfg_img_emb
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        assert packed_vae_token_indexes.max() < seq_len, (
            f"vae_indexes OOB: max={packed_vae_token_indexes.max()} seq_len={seq_len}"
        )
        assert x_t.shape[0] == packed_vae_token_indexes.shape[0], (
            f"x_t/vae_indexes mismatch: {x_t.shape[0]} vs {packed_vae_token_indexes.shape[0]}"
        )
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        # NOTE: route through __call__ (not .forward_inference) so FSDP2 hooks
        # fire; FastGen Qwen.forward dispatches to forward_inference because no
        # `packed_sequence` kwarg is passed.
        want_features = feature_indices is not None and len(feature_indices) > 0
        output = self.language_model(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            feature_indices=feature_indices,
            return_features_early=return_features_early,
            **extra_inputs,
        )

        # GAN: reduce each collected full-sequence feature to VAE tokens
        # only -> (N_vae, hidden), the per-image patch grid the discriminator
        # consumes (unpatchified to spatial in KokoroNetwork.forward). The grad
        # graph from these features back to x_t is preserved (no detach here).
        if want_features:
            vae_feats = [feat[packed_vae_token_indexes] for feat in output.features]
            if return_features_early:
                # output.packed_query_sequence is None on the early-return path.
                return vae_feats

        if return_vae_hidden:
            return output.packed_query_sequence[packed_vae_token_indexes]

        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]
        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only
                # supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        if want_features:
            return v_t, vae_feats
        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            if self.config.use_lora:
                pred_logits = self.language_model.model.model.lm_head(packed_query_sequence)
            else:
                pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output
