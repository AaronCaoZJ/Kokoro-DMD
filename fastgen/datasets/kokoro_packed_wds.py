# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Kokoro packed-sequence dataset for FastGen DMD distillation.
# Adapted from PackedDataset and DanbooruWebDatasetT2I of the reference Kokoro training code.
#
# Key differences from the original:
#   - No timestep sampling (DMD injects noise externally via packed_latent_input)
#   - No shortcut / bootstrap logic
#   - No VIT tokens (generation-only)
#   - latent_patch_size is a parameter (not hardcoded 2)
#   - Text dropout applied at sub-dataset level

import io
import json
import math
import random
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.utils.data



# ──────────────────────────────────────────────────────────────────────────────
# KokoroT2ISubDataset
# ──────────────────────────────────────────────────────────────────────────────

class KokoroT2ISubDataset:
    """
    Iterable sub-dataset that reads Kokoro WDS shards (vae.pth + json).

    Each shard entry contains:
        "vae.pth"  : torch.save'd VAE latent tensor, shape (C, H_lat, W_lat).
        "json"     : JSON blob with {"post_id", "prompts": {caption_type: text}, ...}.

    Yields sample dicts compatible with KokoroPackedDataset:
        sequence_plan     : list of {type, enable_cfg, loss, special_token_loss}
        image_tensor_list : [vae_tensor]  — raw VAE latent, shape (C, H_lat, W_lat)
        text_ids_list     : [token_ids]  — or [] when CFG dropout fires
        num_tokens        : estimated total tokens (text + special + vae)
        data_indexes      : {key: wds_key}

    Args:
        shard_patterns : Single glob/URL or list of patterns fed to webdataset.
        tokenizer      : Object with ``encode(text) -> List[int]``.
        latent_patch_size    : Patch size used when patchifying the latent (default 2).
        max_latent_size      : Maximum number of patches per side for position IDs.
        text_cond_dropout_prob : Probability of dropping the text condition (CFG dropout).
    """

    def __init__(
        self,
        shard_patterns,
        tokenizer,
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        text_cond_dropout_prob: float = 0.1,
    ):
        if isinstance(shard_patterns, str):
            shard_patterns = [shard_patterns]
        self.shard_patterns = shard_patterns
        self.tokenizer = tokenizer
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.text_cond_dropout_prob = text_cond_dropout_prob

    @staticmethod
    def _extract_text(meta: dict) -> str:
        """
        Extract a caption string from a WDS JSON blob.

        Supports two schemas found in the Kokoro data directories:

        New format (114w_tagdict_1024, diff_new, …):
            {"prompts": {"tags": "...", "en_description": "...", ...}, ...}
            → pick "tags", fall back to any value.

        Old format (danbooru2024_wds_*, e621_*, …):
            {"tags_dict": {"tag_string_character": [...], ...}, ...}
            → flatten all tag lists into a comma-separated string.
        """
        # ── New format ────────────────────────────────────────────────────────
        prompts = meta.get("prompts")
        if prompts and isinstance(prompts, dict):
            # prefer "tags", then any non-empty value
            text = prompts.get("tags") or prompts.get("en_description") or ""
            if not text:
                for v in prompts.values():
                    if isinstance(v, str) and v.strip():
                        text = v
                        break
            return text.strip() if isinstance(text, str) else ""

        # ── Old format: tags_dict is a Python-dict-looking string or actual dict
        tags_dict = meta.get("tags_dict")
        if tags_dict:
            if isinstance(tags_dict, str):
                try:
                    import ast
                    tags_dict = ast.literal_eval(tags_dict)
                except Exception:
                    return tags_dict.strip()
            if isinstance(tags_dict, dict):
                parts = []
                for v in tags_dict.values():
                    if isinstance(v, list):
                        parts.extend(str(t) for t in v)
                    elif isinstance(v, str) and v.strip():
                        parts.append(v.strip())
                return ", ".join(parts)

        return ""

    def __iter__(self):
        import glob as _glob
        import webdataset as wds  # lazy import

        # wds.WebDataset does not support shell globs — expand patterns first.
        expanded: List[str] = []
        for p in self.shard_patterns:
            matched = sorted(_glob.glob(p))
            expanded.extend(matched)
        if not expanded:
            return

        dataset = wds.WebDataset(expanded, nodesplitter=wds.split_by_node)
        for raw in dataset:
            try:
                # ── Decode VAE latent ──────────────────────────────────────
                vae_tensor = torch.load(io.BytesIO(raw["vae.pth"]), weights_only=False)
                if not isinstance(vae_tensor, torch.Tensor):
                    continue
                # Some shards save (1, C, H, W); normalise to (C, H, W)
                if vae_tensor.ndim == 4 and vae_tensor.shape[0] == 1:
                    vae_tensor = vae_tensor.squeeze(0)
                if vae_tensor.ndim != 3:
                    continue

                # ── Decode prompts ─────────────────────────────────────────
                meta = json.loads(raw["json"])
                text = self._extract_text(meta)
                text_ids = self.tokenizer.encode(text)

                # ── Token count estimate ───────────────────────────────────
                H_lat, W_lat = vae_tensor.shape[1], vae_tensor.shape[2]
                h_tok = H_lat // self.latent_patch_size
                w_tok = W_lat // self.latent_patch_size
                n_vae = h_tok * w_tok
                n_text_with_special = len(text_ids) + 2  # BOS + EOS
                n_image_special = 2                       # startofimage + endofimage
                num_tokens = n_text_with_special + n_image_special + n_vae

                # ── Distillation data always keeps text in the packed sample ──
                # Teacher CFG is built explicitly from cond/uncond layouts, so
                # training data itself should not randomly drop the text branch.
                text_ids_list = [text_ids]
                sequence_plan = [
                    {"type": "text",      "enable_cfg": 0, "loss": 0, "special_token_loss": 0},
                    {"type": "vae_image", "enable_cfg": 0, "loss": 1, "special_token_loss": 0},
                ]

                yield {
                    "sequence_plan":     sequence_plan,
                    "image_tensor_list": [vae_tensor],
                    "text_ids_list":     text_ids_list,
                    "num_tokens":        num_tokens,
                    "data_indexes":      {"key": raw.get("__key__", "")},
                }

            except Exception as e:
                print(f"[KokoroT2ISubDataset] skip sample due to error: {e}")
                continue


# ──────────────────────────────────────────────────────────────────────────────
# KokoroPackedDataset + collate_fn
# ──────────────────────────────────────────────────────────────────────────────

def _get_latent_position_ids(
    h_tok: int, w_tok: int, max_latent_size: int
) -> torch.Tensor:
    """Row-major 1-D position IDs for a (h_tok × w_tok) latent grid."""
    coords_h = torch.arange(h_tok)
    coords_w = torch.arange(w_tok)
    return (coords_h[:, None] * max_latent_size + coords_w).flatten()


class KokoroPackedDataset(torch.utils.data.IterableDataset):
    """
    Packs multiple T2I samples into one long packed sequence (~32 k tokens).

    Ported from the reference PackedDataset, simplified for DMD:
    - No timestep sampling (DMD injects noise externally).
    - No shortcut / bootstrap / VIT handling.
    - latent_patch_size is configurable.

    Args:
        sub_datasets         : Iterable sub-datasets (e.g. KokoroT2ISubDataset).
        dataset_weights      : Sampling weight per sub-dataset.
        tokenizer            : Object with bos_token_id / eos_token_id attributes.
        special_tokens       : Dict {"bos", "eos", "start_of_image", "end_of_image"}.
        expected_num_tokens  : Target tokens per packed batch (default 32 768).
        max_num_tokens       : Hard upper limit per batch.
        max_num_tokens_per_sample : Skip samples larger than this.
        prefer_buffer_before : Use buffer when curr < this value.
        max_buffer_size      : Max samples held in overflow buffer.
        latent_patch_size    : Patch size for latent patchification (default 2).
        max_latent_size      : Max patches per side for position IDs (default 32).
    """

    def __init__(
        self,
        sub_datasets: List,
        dataset_weights: List[float],
        tokenizer,
        special_tokens: Dict[str, int],
        expected_num_tokens: int = 32768,
        max_num_tokens: int = 36864,
        max_num_tokens_per_sample: int = 16384,
        prefer_buffer_before: int = 16384,
        max_buffer_size: int = 50,
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
    ):
        super().__init__()
        self.sub_datasets           = sub_datasets
        self.dataset_weights        = dataset_weights
        self.expected_num_tokens    = expected_num_tokens
        self.max_num_tokens         = max_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before   = prefer_buffer_before
        self.max_buffer_size        = max_buffer_size
        self.latent_patch_size      = latent_patch_size
        self.max_latent_size        = max_latent_size

        # Special token IDs
        self.bos_token_id    = special_tokens["bos"]
        self.eos_token_id    = special_tokens["eos"]
        self.start_of_image  = special_tokens["start_of_image"]
        self.end_of_image    = special_tokens["end_of_image"]

    # ── internal state ──────────────────────────────────────────────────────

    def _new_sequence_status(self) -> dict:
        return dict(
            curr                       = 0,
            sample_lens                = [],
            split_lens                 = [],
            attn_modes                 = [],
            packed_position_ids        = [],
            packed_text_ids            = [],
            packed_text_indexes        = [],
            vae_tensors                = [],
            vae_latent_shapes          = [],
            packed_latent_position_ids = [],
            packed_vae_token_indexes   = [],
            mse_loss_indexes           = [],
            # uncond layout (matches training-time CFG dropout: no text segment)
            uncond_curr                       = 0,
            uncond_sample_lens                = [],
            uncond_split_lens                 = [],
            uncond_attn_modes                 = [],
            uncond_packed_text_ids            = [],
            uncond_packed_text_indexes        = [],
            uncond_packed_position_ids        = [],
            uncond_packed_vae_token_indexes   = [],
            uncond_mse_loss_indexes           = [],
        )

    def _to_tensor(self, ss: dict) -> dict:
        """Convert accumulated lists to tensors and build the batch dict."""
        # Pad sequence lengths to a multiple of BLOCK_SIZE=128 for flex attention.
        # Extra padding tokens are zero-filled by PackedAttention.forward_train.
        BLOCK_SIZE = 128
        seq_len = sum(ss["sample_lens"])
        pad_len = (-seq_len) % BLOCK_SIZE

        split_lens = list(ss["split_lens"])
        attn_modes = list(ss["attn_modes"])
        padded_sample_lens = list(ss["sample_lens"])
        if pad_len > 0:
            split_lens.append(pad_len)
            attn_modes.append("causal")
            padded_sample_lens.append(pad_len)

        uncond_seq_len = sum(ss["uncond_sample_lens"])
        uncond_pad_len = (-uncond_seq_len) % BLOCK_SIZE
        uncond_split_lens = list(ss["uncond_split_lens"])
        uncond_attn_modes = list(ss["uncond_attn_modes"])
        uncond_padded_sample_lens = list(ss["uncond_sample_lens"])
        if uncond_pad_len > 0:
            uncond_split_lens.append(uncond_pad_len)
            uncond_attn_modes.append("causal")
            uncond_padded_sample_lens.append(uncond_pad_len)

        data = dict(
            sequence_length    = seq_len,
            sample_lens        = padded_sample_lens,
            split_lens         = split_lens,
            attn_modes         = attn_modes,
            packed_text_ids    = torch.tensor(ss["packed_text_ids"],   dtype=torch.long),
            packed_text_indexes= torch.tensor(ss["packed_text_indexes"],dtype=torch.long),
            packed_position_ids= torch.tensor(ss["packed_position_ids"],dtype=torch.long),
            mse_loss_indexes   = torch.tensor(ss["mse_loss_indexes"],  dtype=torch.long),
            # uncond layout (text-free sequence; mirrors training-time CFG dropout)
            uncond_sequence_length          = uncond_seq_len,
            uncond_sample_lens              = uncond_padded_sample_lens,
            uncond_split_lens               = uncond_split_lens,
            uncond_attn_modes               = uncond_attn_modes,
            uncond_packed_text_ids          = torch.tensor(
                ss["uncond_packed_text_ids"], dtype=torch.long),
            uncond_packed_text_indexes      = torch.tensor(
                ss["uncond_packed_text_indexes"], dtype=torch.long),
            uncond_packed_position_ids      = torch.tensor(
                ss["uncond_packed_position_ids"], dtype=torch.long),
            uncond_packed_vae_token_indexes = torch.tensor(
                ss["uncond_packed_vae_token_indexes"], dtype=torch.long),
            uncond_mse_loss_indexes         = torch.tensor(
                ss["uncond_mse_loss_indexes"], dtype=torch.long),
        )

        # Pad VAE latents into (N_img, C, H_max, W_max)
        vae_tensors = ss["vae_tensors"]
        if vae_tensors:
            shapes = [t.shape for t in vae_tensors]  # list of (C, H, W)
            max_C = max(s[0] for s in shapes)
            max_H = max(s[1] for s in shapes)
            max_W = max(s[2] for s in shapes)
            padded = torch.zeros(len(vae_tensors), max_C, max_H, max_W)
            for i, t in enumerate(vae_tensors):
                padded[i, :t.shape[0], :t.shape[1], :t.shape[2]] = t
            data["padded_latent"]                = padded
            data["patchified_vae_latent_shapes"] = ss["vae_latent_shapes"]
            data["packed_latent_position_ids"]   = torch.cat(ss["packed_latent_position_ids"])
            data["packed_vae_token_indexes"]     = torch.tensor(
                ss["packed_vae_token_indexes"], dtype=torch.long
            )

        return data

    # ── packing logic ────────────────────────────────────────────────────────

    def _pack_sample(self, sample: dict, ss: dict) -> dict:
        """
        Pack one sample into the running sequence status ``ss``.
        Mirrors PackedDataset.pack_sequence() but simplified for DMD.
        """
        image_tensor_list = list(sample["image_tensor_list"])   # copy to avoid mutation
        text_ids_list     = list(sample["text_ids_list"])
        sequence_plan     = sample["sequence_plan"]

        split_lens  = []
        attn_modes  = []
        curr        = ss["curr"]
        rope_id     = 0     # Kokoro semantics: reset rope positions per sample
        sample_len  = 0
        # uncond layout (per-sample accumulators)
        uncond_split_lens = []
        uncond_attn_modes = []

        for item in sequence_plan:
            split_len = 0

            if item["type"] == "text":
                if not text_ids_list:
                    # dropout case: no text ids supplied
                    continue
                text_ids = text_ids_list.pop(0)
                shifted  = [self.bos_token_id] + text_ids   # prepend BOS

                ss["packed_text_ids"].extend(shifted)
                ss["packed_text_indexes"].extend(range(curr, curr + len(shifted)))
                curr    += len(shifted)
                split_len += len(shifted)

                # EOS token
                ss["packed_text_ids"].append(self.eos_token_id)
                ss["packed_text_indexes"].append(curr)
                curr    += 1
                split_len += 1

                # Position IDs: sequential for text
                ss["packed_position_ids"].extend(
                    range(rope_id, rope_id + split_len)
                )
                rope_id += split_len

                attn_modes.append("causal")
                split_lens.append(split_len)
                sample_len += split_len

            elif item["type"] == "vae_image":
                if not image_tensor_list:
                    continue
                vae_tensor = image_tensor_list.pop(0)
                H_lat, W_lat = vae_tensor.shape[1], vae_tensor.shape[2]
                h_tok = H_lat // self.latent_patch_size
                w_tok = W_lat // self.latent_patch_size
                n_vae = h_tok * w_tok

                # <|startofimage|>
                ss["packed_text_ids"].append(self.start_of_image)
                ss["packed_text_indexes"].append(curr)
                curr    += 1
                split_len += 1

                # VAE token indexes (positions in the packed sequence)
                ss["packed_vae_token_indexes"].extend(range(curr, curr + n_vae))
                curr    += n_vae
                split_len += n_vae

                # <|endofimage|>
                ss["packed_text_ids"].append(self.end_of_image)
                ss["packed_text_indexes"].append(curr)
                curr    += 1
                split_len += 1

                # All image tokens share the same rope position
                ss["packed_position_ids"].extend([rope_id] * split_len)

                # Track latent for batching
                ss["vae_tensors"].append(vae_tensor)
                ss["vae_latent_shapes"].append((h_tok, w_tok))
                ss["packed_latent_position_ids"].append(
                    _get_latent_position_ids(h_tok, w_tok, self.max_latent_size)
                )

                # MSE loss indexes (same as VAE positions for T2I loss=1)
                if item.get("loss", 0) == 1:
                    mse_start = curr - n_vae - 1  # first vae token index in seq
                    ss["mse_loss_indexes"].extend(range(mse_start, mse_start + n_vae))

                # rope_id not incremented for loss=1 image tokens (they are generated output)
                if item.get("loss", 0) == 0:
                    rope_id += 1

                attn_modes.append("noise" if item.get("loss", 0) == 1 else "full")
                split_lens.append(split_len)
                sample_len += split_len

                # ── uncond layout: same image, no text prefix ──────────────
                # Sequence: [SOI, vae_1, …, vae_n, EOI] — length n_vae+2
                ub = ss["uncond_curr"]
                n_img_total = n_vae + 2
                ss["uncond_packed_text_ids"].extend([self.start_of_image, self.end_of_image])
                ss["uncond_packed_text_indexes"].extend([ub, ub + n_vae + 1])
                # Diagnostic CFG alignment: keep the image block at the same rope
                # phase as the conditional branch, even though the text tokens
                # are absent from the uncond sequence.
                ss["uncond_packed_position_ids"].extend([rope_id] * n_img_total)
                # VAE tokens occupy positions ub+1 … ub+n_vae (after SOI)
                ss["uncond_packed_vae_token_indexes"].extend(
                    range(ub + 1, ub + 1 + n_vae)
                )
                if item.get("loss", 0) == 1:
                    ss["uncond_mse_loss_indexes"].extend(
                        range(ub + 1, ub + 1 + n_vae)
                    )
                ss["uncond_curr"] += n_img_total
                uncond_split_lens.append(n_img_total)
                uncond_attn_modes.append(
                    "noise" if item.get("loss", 0) == 1 else "full"
                )

        ss["curr"] = curr
        ss["sample_lens"].append(sample_len)
        ss["split_lens"].extend(split_lens)
        ss["attn_modes"].extend(attn_modes)

        ss["uncond_sample_lens"].append(sum(uncond_split_lens))
        ss["uncond_split_lens"].extend(uncond_split_lens)
        ss["uncond_attn_modes"].extend(uncond_attn_modes)

        return ss

    # ── main iterator ────────────────────────────────────────────────────────

    def __iter__(self):
        total_w = sum(self.dataset_weights)
        assert total_w > 0.0, "dataset_weights must be positive"
        cumprobs = [sum(self.dataset_weights[: i + 1]) / total_w
                    for i in range(len(self.dataset_weights))]

        iters   = [iter(ds) for ds in self.sub_datasets]
        ss      = self._new_sequence_status()
        buffer  = []

        while True:
            # ── pick next sample ──────────────────────────────────────────
            if ss["curr"] < self.prefer_buffer_before and buffer:
                sample = buffer.pop(0)
            else:
                r = random.random()
                gi = next(i for i, cp in enumerate(cumprobs) if r < cp)
                sample = next(iters[gi])

            # ── size check (conservative: +2 per plan item) ───────────────
            est = sample["num_tokens"] + 2 * len(sample["sequence_plan"])
            if est > self.max_num_tokens_per_sample:
                continue

            if ss["curr"] + est > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size:
                    buffer.append(sample)
                else:
                    # yield current batch and start fresh
                    if 0 < sum(ss["sample_lens"]) <= self.max_num_tokens:
                        yield self._to_tensor(ss)
                    ss = self._new_sequence_status()
                continue

            ss = self._pack_sample(sample, ss)

            if ss["curr"] >= self.expected_num_tokens:
                if sum(ss["sample_lens"]) <= self.max_num_tokens:
                    yield self._to_tensor(ss)
                ss = self._new_sequence_status()


# ──────────────────────────────────────────────────────────────────────────────
# Collate function (batch_size=1 pass-through)
# ──────────────────────────────────────────────────────────────────────────────

def kokoro_collate_fn(batch):
    """
    Collate function for KokoroPackedDataset.

    Since each packed batch is already a single large sequence (batch_size=1),
    this function simply unwraps the single-element list returned by DataLoader.
    """
    assert len(batch) == 1, "KokoroPackedDataset expects batch_size=1"
    return batch[0]
