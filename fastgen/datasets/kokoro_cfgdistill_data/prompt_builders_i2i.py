# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Ported verbatim from data/prompt_builders_i2i.py of the reference CFG-distillation post-training code.
# Only adjustment: import path for sample_utils.

import random
import re

from .sample_utils import (
    apply_tag_drop_strategy,
    as_list,
    first_text,
    join_tags,
    load_no_drop_tags,
    text_value,
)


def build_prompt_diff(json_data):
    prompts = json_data.get("prompts", {})
    if not isinstance(prompts, dict):
        return ""

    if "style" in prompts:
        style = prompts.get("prompt_template") or prompts.get("style") or ""
        tag_groups = []
        for key in ("character_tags", "copyright_tags", "general_tags", "tags"):
            tag_groups.extend(as_list(prompts.get(key)))
        tags = join_tags(tag_groups)
        return f"{style} {tags}".strip()

    if random.random() < 0.4:
        text = first_text(
            prompts,
            (
                "en_long_command",
                "en_short_command",
                "cn_long_command",
                "cn_short_command",
                "jp_long_command",
                "jp_short_command",
            ),
        )
        if text:
            return text

    tags = prompts.get("tags", "")
    if isinstance(tags, str) and tags.strip():
        tags = re.sub(r"\bremove\b", "no", tags)
        return join_tags(as_list(tags))

    return first_text(prompts, ("en_long_command", "en_short_command"))


def build_prompt_banana(metadata):
    natural = metadata.get("natural", "")
    tags = metadata.get("tag", "")
    if random.random() < 0.5 and isinstance(natural, str) and natural.strip():
        return natural.strip()
    if isinstance(tags, str) and tags.strip():
        return join_tags(as_list(tags))
    return natural.strip() if isinstance(natural, str) else ""


def build_prompt_story(img_meta):
    language_prob = random.random()
    if language_prob < 0.8:
        key = "prompt_en"
    elif language_prob < 0.9:
        key = "prompt_zh"
    else:
        key = "prompt_jp"

    prompt = text_value(img_meta.get(key))
    if not prompt:
        prompt = first_text(img_meta, ("prompt_en", "prompt_zh", "prompt_jp"))
    return prompt or "move to the next scene"


def build_prompt_story_new(img_meta):
    tags = img_meta.get("diff_tags", "")
    no_drop_set = load_no_drop_tags()
    return apply_tag_drop_strategy(tags, no_drop_set)


def build_prompt_rp_story_new(img_meta):
    prompt = text_value(img_meta.get("prompt"))
    if not prompt:
        print("Warning: prompt is empty for realperson_story sample. This sample will be filtered out.")
        prompt = ""
    return prompt


I2I_PROMPT_BUILDERS = {
    "i2i_diff": build_prompt_diff,
    "i2i_banana": build_prompt_banana,
    "i2i_story": build_prompt_story,
    "i2i_story_new": build_prompt_story_new,
    "i2i_rp_story_new": build_prompt_rp_story_new,
}


def get_i2i_prompt_builder(name):
    try:
        return I2I_PROMPT_BUILDERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(I2I_PROMPT_BUILDERS))
        raise ValueError(f"Unknown i2i prompt_builder '{name}'. Available: {available}") from exc
