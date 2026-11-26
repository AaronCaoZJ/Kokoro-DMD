"""Inference-branch context / KV-cache helpers for Kokoro DMD2.

Adapted from BAGEL's ``InterleaveInferencer`` (``inferencer.py``:
init_gen_context, update_context_text, update_context_image), trimmed to the
cond branch, plus the context / layout builders used by CFG-distillation
training (build_t2i_context, build_i2i_context, prepare_flow_layout,
cond_layout_fields).

These run in the trainer process on the current rank; the resulting
``NaiveCache`` is never sent through a dataloader worker or scattered across
ranks.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from fastgen.networks.kokoro.qwen2_navit import NaiveCache

# Keys produced by Bagel.prepare_vae_latent that _forward_flow_train consumes.
_COND_LAYOUT_KEYS = (
    "packed_vae_token_indexes",
    "packed_vae_position_ids",
    "packed_text_ids",
    "packed_text_indexes",
    "packed_position_ids",
    "packed_indexes",
    "packed_seqlens",
    "key_values_lens",
    "packed_key_value_indexes",
)


def init_gen_context(bagel, num_images: int = 1) -> Dict[str, Any]:
    """Empty generation context (InterleaveInferencer.init_gen_context)."""
    return {
        "kv_lens": [0] * num_images,
        "ropes": [0] * num_images,
        "past_key_values": NaiveCache(bagel.config.llm_config.num_hidden_layers),
    }


@torch.no_grad()
def update_context_text(
    bagel, tokenizer, new_token_ids, device, text: str, gen_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Append a text prompt to the KV cache (InterleaveInferencer.update_context_text
    without the token-weighting path — DMD2 does not use weighted prompts).
    Updates and returns ``gen_context``."""
    past_key_values = gen_context["past_key_values"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]
    batch_size = len(kv_lens)

    generation_input, kv_lens, ropes = bagel.prepare_prompts(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        prompts=[text] * batch_size,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        token_weights=None,
    )
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)

    use_cuda = torch.device(device).type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_cuda, dtype=torch.bfloat16):
        past_key_values = bagel.forward_cache_update_text(past_key_values, **generation_input)

    gen_context["kv_lens"] = kv_lens
    gen_context["ropes"] = ropes
    gen_context["past_key_values"] = past_key_values
    return gen_context


def build_t2i_context(bagel, tokenizer, new_token_ids, device, text: str) -> Dict[str, Any]:
    """T2I cond context: the text prompt only
    (DMD2 uses cfg-embed single forward — no text/img uncond contexts)."""
    ctx = init_gen_context(bagel, num_images=1)
    return update_context_text(bagel, tokenizer, new_token_ids, device, text, ctx)


@torch.no_grad()
def update_context_image(
    bagel,
    vae_model,
    vit_transform,
    vae_transform,
    new_token_ids,
    device,
    image,
    gen_context: Dict[str, Any],
    vae: bool = True,
    vit: bool = True,
) -> Dict[str, Any]:
    """Port of InterleaveInferencer.update_context_image.

    Builds the VAE (generation) and/or VIT (understanding) image KV-cache
    contributions for a source image, mutating ``gen_context`` in place.

    Order is VAE-then-VIT, matching InterleaveInferencer.update_context_image exactly.
    This order is baked into the frozen cfg-distill teacher's KV layout; do
    NOT swap the two blocks (reversing them silently corrupts the i2i teacher
    target — the source-image token / rope positions would no longer match
    what the teacher was distilled with).
    """
    past_key_values = gen_context["past_key_values"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]

    use_cuda = torch.device(device).type == "cuda"

    if vae:
        gen_input, kv_lens, ropes = bagel.prepare_vae_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[image],
            transforms=vae_transform,
            new_token_ids=new_token_ids,
        )
        for k, v in gen_input.items():
            if isinstance(v, torch.Tensor):
                gen_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=use_cuda, dtype=torch.bfloat16):
            past_key_values = bagel.forward_cache_update_vae(
                vae_model, past_key_values, **gen_input,
            )

    if vit:
        gen_input, kv_lens, ropes = bagel.prepare_vit_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[image],
            transforms=vit_transform,
            new_token_ids=new_token_ids,
        )
        for k, v in gen_input.items():
            if isinstance(v, torch.Tensor):
                gen_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=use_cuda, dtype=torch.bfloat16):
            past_key_values = bagel.forward_cache_update_vit(past_key_values, **gen_input)

    gen_context["kv_lens"] = kv_lens
    gen_context["ropes"] = ropes
    gen_context["past_key_values"] = past_key_values
    return gen_context


def build_i2i_context(
    bagel,
    vae_model,
    tokenizer,
    vit_transform,
    vae_transform,
    new_token_ids,
    device,
    text: str,
    source_image,
) -> Dict[str, Any]:
    """I2I cond context: the source image (VAE, then VIT), then the text prompt.
    DMD2 inference branch: no text/img uncond contexts, cfg-embed only on teacher."""
    ctx = init_gen_context(bagel, num_images=1)
    ctx = update_context_image(
        bagel, vae_model, vit_transform, vae_transform, new_token_ids,
        device, source_image, ctx, vae=True, vit=True,
    )
    ctx = update_context_text(bagel, tokenizer, new_token_ids, device, text, ctx)
    return ctx


def prepare_flow_layout(bagel, cond_ctx, image_shape, new_token_ids) -> Dict[str, Any]:
    """Packed-sequence layout of the noisy VAE latent that follows the cond context."""
    return bagel.prepare_vae_latent(
        curr_kvlens=cond_ctx["kv_lens"],
        curr_rope=cond_ctx["ropes"],
        image_sizes=[image_shape],
        new_token_ids=new_token_ids,
    )


def cond_layout_fields(layout: Dict[str, Any], device) -> Dict[str, Any]:
    """Select the ``_COND_LAYOUT_KEYS`` fields of ``layout`` and move them to ``device``."""
    return {k: layout[k].to(device) for k in _COND_LAYOUT_KEYS}
