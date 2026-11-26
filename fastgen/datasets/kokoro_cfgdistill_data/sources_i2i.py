# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Ported verbatim from data/sources_i2i.py of the reference CFG-distillation post-training code.
# Only adjustment: import paths.

import random
import re

from .prompt_builders_i2i import get_i2i_prompt_builder
from .sample_utils import rgb, valid_latent


def _pack_i2i_sample(text, vae_latent, source_image):
    text = text or ""
    if not text.strip() or source_image is None or not valid_latent(vae_latent):
        return None
    return {
        "text": text,
        "vae_latent": vae_latent,
        "source_image": source_image,
        "task_type": "i2i",
    }


def decode_i2i_diff(sample, prompt_builder="i2i_diff"):
    json_data = sample["diff.json"]
    builder = get_i2i_prompt_builder(prompt_builder)
    text = builder(json_data)
    vae_latent = sample["child.pth"].squeeze(0)
    source_image = rgb(sample["parent.webp"])
    return _pack_i2i_sample(text, vae_latent, source_image)


def decode_i2i_reverse(sample, prompt_builder="i2i_banana"):
    metadata = sample["reverse.json"]
    builder = get_i2i_prompt_builder(prompt_builder)
    input_ext = metadata.get("input_ext", ".webp")
    text = builder(metadata)
    vae_latent = sample["output.pth"].squeeze(0)
    source_image = rgb(sample.get(f"input{input_ext}"))
    return _pack_i2i_sample(text, vae_latent, source_image)


def decode_i2i_group(sample, prompt_builder="i2i_banana"):
    metadata = sample["group.json"]
    task = metadata.get("task", "")
    if task in {"clothes_m", "clothes_f"}:
        return None
    num_outputs = int(metadata.get("num_outputs") or 0)
    if num_outputs <= 0:
        return None

    if num_outputs >= 2 and random.random() < 0.5:
        input_index, output_index = random.sample(range(num_outputs), 2)
        input_meta = sample.get(f"output{input_index}.json", {})
        input_ext = input_meta.get("ext", ".webp")
        source_image = rgb(sample.get(f"output{input_index}{input_ext}"))
    else:
        output_index = random.randint(0, num_outputs - 1)
        input_ext = metadata.get("input_ext", ".webp")
        source_image = rgb(sample.get(f"input{input_ext}"))

    output_meta = sample.get(f"output{output_index}.json", {})
    builder = get_i2i_prompt_builder(prompt_builder)
    text = builder(output_meta)
    vae_latent = sample[f"output{output_index}.pth"].squeeze(0)
    return _pack_i2i_sample(text, vae_latent, source_image)


def decode_i2i_story(sample, prompt_builder="i2i_story"):
    indices = sorted(
        int(match.group(1))
        for key in sample.keys()
        if (match := re.match(r"^img(\d+)\.webp$", key))
        and f"img{match.group(1)}.pth" in sample
        and f"img{match.group(1)}.json" in sample
    )
    pairs = [(a, b) for a, b in zip(indices, indices[1:]) if b == a + 1]
    if not pairs:
        return None

    source_idx, target_idx = random.choice(pairs)
    target_meta = sample.get(f"img{target_idx}.json", {})
    builder = get_i2i_prompt_builder(prompt_builder)
    text = builder(target_meta)
    source_image = rgb(sample.get(f"img{source_idx}.webp"))
    vae_latent = sample[f"img{target_idx}.pth"].squeeze(0)
    return _pack_i2i_sample(text, vae_latent, source_image)


def decode_i2i_auto(sample, prompt_builder=None):
    if "diff.json" in sample:
        return decode_i2i_diff(sample, prompt_builder or "i2i_diff")
    if "reverse.json" in sample:
        return decode_i2i_reverse(sample, prompt_builder or "i2i_banana")
    if "group.json" in sample:
        return decode_i2i_group(sample, prompt_builder or "i2i_banana")
    if "story.json" in sample:
        return decode_i2i_story(sample, prompt_builder or "i2i_story")
    return None


I2I_DECODERS = {
    "auto": decode_i2i_auto,
    "i2i_auto": decode_i2i_auto,
    "i2i_diff": decode_i2i_diff,
    "i2i_reverse": decode_i2i_reverse,
    "i2i_group": decode_i2i_group,
    "i2i_story": decode_i2i_story,
}


def get_i2i_decoder(name):
    try:
        return I2I_DECODERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(I2I_DECODERS))
        raise ValueError(f"Unknown i2i decoder '{name}'. Available: {available}") from exc
