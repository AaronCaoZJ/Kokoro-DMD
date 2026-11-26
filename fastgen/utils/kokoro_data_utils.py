# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from BAGEL's data/data_utils.py.
# Only the functions needed for packed-sequence data loading and
# position-id computation are included. No Kokoro-specific imports.

import math
import random

import torch
from torch.nn.attention.flex_attention import or_masks, and_masks


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    """
    Build flex_attention sparse mask for packed sequences.
    """
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []
    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ["full", "noise"] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == "noise" else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)
    document_id = torch.cat(
        [torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]
    ).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    (C, H, W) → (N_patches, patch_size² × C)
    H and W must be divisible by patch_size.
    """
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(
    img_h: int,
    img_w: int,
    patch_size: int,
    max_num_patches_per_side: int,
) -> torch.Tensor:
    """
    Generate 1-D RoPE position IDs for a (img_h × img_w) image by
    extrapolation: each patch maps to row * max_num_patches_per_side + col.

    Returns: LongTensor of shape (num_patches_h * num_patches_w,)
    """
    num_patches_h = img_h // patch_size
    num_patches_w = img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(
    img_h: int,
    img_w: int,
    patch_size: int,
    max_num_patches_per_side: int,
) -> torch.Tensor:
    """
    Generate 1-D RoPE position IDs by interpolation into max_num_patches_per_side bins.
    Smaller images are upscaled to fill the full ID range.

    Returns: LongTensor of shape (num_patches_h * num_patches_w,)
    """
    num_patches_h = img_h // patch_size
    num_patches_w = img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


def prepare_attention_mask_per_sample(
    split_lens: list,
    attn_modes: list,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a per-sample additive attention mask (0 = attend, -inf = masked).

    split_lens : list of ints, one per split (e.g. [n_text, n_vae])
    attn_modes : list of str, one per split. Each must be one of:
        "causal" — causal attention within the split + full access to all
                   preceding tokens (prefix attention).
        "full"   — bidirectional attention within the split + full prefix.
        "noise"  — bidirectional within the split + full prefix, but ALL
                   other splits are blocked from attending to this split
                   (i.e. columns for this split are zeroed out except for
                   tokens within the same split).

    Returns: Float tensor of shape (L, L) where L = sum(split_lens).
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    # Pass 1: set within-split and prefix access
    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ("causal", "full", "noise"), f"unknown attn_mode: {attn_mode}"
        if attn_mode == "causal":
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum : csum + s, :csum] = 1
        else:  # "full" or "noise"
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s), device=device)
            attention_mask[csum : csum + s, :csum] = 1
        csum += s

    # Pass 2: for "noise" splits, block all other splits from attending to them
    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = 0                    # block all rows
            attention_mask[csum : csum + s, csum : csum + s] = 1     # restore self-attention
        csum += s

    # Convert bool mask → additive float mask (0 = attend, -inf = masked)
    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )
    return attention_mask
