import random

from .sample_utils import (
    as_list,
    danbooru_tags,
    extract_r18_tags,
    first_text,
    get_keywords_by_lang,
    get_text_by_lang,
    join_tags,
    load_r18_tags,
    sample_language,
    sample_reg_tokens,
    weighted_multilingual,
)

def build_prompt_114w_tagdict(json_data):
    prompts = json_data.get("prompts") or {}
    if not isinstance(prompts, dict):
        return ""

    tags_prompt = ""
    if isinstance(prompts.get("tags_dict"), dict):
        tags_prompt = join_tags(danbooru_tags(prompts["tags_dict"], include_copyright=True))
    if not tags_prompt:
        tags_prompt = join_tags(as_list(prompts.get("tags")))

    natural_en = first_text(prompts, ("en_instruction", "en_description"))
    natural_other = first_text(
        prompts,
        ("cn_instruction", "jp_instruction", "cn_description", "jp_description"),
    )

    choices = []
    if tags_prompt:
        choices.extend([tags_prompt] * 7)
    if natural_en:
        choices.extend([natural_en] * 2)
    if natural_other:
        choices.append(natural_other)
    return random.choice(choices) if choices else ""


def build_prompt_danbooru(json_data):
    category_number = json_data.get("category_number")

    if category_number in (7, 12):
        comic_desc = json_data.get("comic_desc", {})
        if not isinstance(comic_desc, dict):
            return ""
        if category_number == 7:
            keys = ("comic_desc_en", "comic_desc_cn", "comic_desc_jp")
        else:
            keys = ("r18_comic_desc_en", "r18_comic_desc_cn", "r18_comic_desc_jp")
        return weighted_multilingual(comic_desc, keys)

    parts = []
    add_location = False

    bg_dict = json_data.get("bg", {})
    if isinstance(bg_dict, dict):
        bg_type = bg_dict.get("bg_type_number", 1)
        if bg_type == 2:
            bg_prompt = bg_dict.get("bg_type_description", "")
        elif bg_type == 3:
            bg_prompt = weighted_multilingual(bg_dict, ("bg_desc_en", "bg_desc_cn", "bg_desc_jp"))
            add_location = True
        else:
            bg_prompt = ""
        if isinstance(bg_prompt, str) and bg_prompt.strip():
            parts.append(bg_prompt.strip())

    text_dict = json_data.get("text", {})
    if isinstance(text_dict, dict) and text_dict.get("has_text", 0) == 1:
        text_prompt = weighted_multilingual(text_dict, ("text_desc_en", "text_desc_cn", "text_desc_jp"))
        if text_prompt:
            parts.append(text_prompt)

    tags_dict = json_data.get("tags_dict", {})
    if isinstance(tags_dict, dict):
        tags = danbooru_tags(tags_dict, include_copyright=False)
        if add_location:
            tags.append("location")
        tags_prompt = join_tags(tags)
        if tags_prompt:
            parts.append(tags_prompt)

    random.shuffle(parts)
    return ". ".join(parts)


def build_prompt_simple_caption(json_data):
    for key in (
        "text",
        "caption",
        "prompt",
        "tags",
        "description",
        "instruction",
    ):
        value = json_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_prompt_auto(json_data):
    text = build_prompt_114w_tagdict(json_data)
    if text.strip():
        return text

    if isinstance(json_data.get("tags_dict"), dict):
        text = build_prompt_danbooru(json_data)
        if text.strip():
            return text

    return build_prompt_simple_caption(json_data)

# Meta tags removed from tag_meta by _get_tags_annotation_anime (set difference). The literals must
# match the reference exactly (spelling, spaces, hyphens).
_REMOVED_META_TAGS_ANIME = frozenset([
    "revision", "commentary request", "variant set", "spoilers", "partially_translated", "tagme", "clothing_request", "annotated", "making-of", "redrawn", "unconventional_media", "bmp-to-png_conversion", "twitter_sample", "self-portrait", "artstation_sample", "adoptable", "check_spoilers", "reversed", "fixed", "thumbnail_surprise", "fanfiction_available", "metadata_pun", "book_cover_redraw", "exif_thumbnail_surprise", "expression_request", "noise_reduction"
])

# Danbooru data stores its tags grouped by type; this sorts them into two groups:
#   kept_tags: tags that are always kept (character, artist, dataset type, plus 'location' when the
#     location field is True). These carry identity; dropping them would ruin the sample.
#   enable_drop_tags: tags that may be dropped at random (general, meta, year, quality, copyright, NSFW,
#     rating, score). Trimming them is the tag-count data augmentation.
# Pure-function version of the reference get_tags_annotation. String fields (year / quality_tag /
# tag_nsfw / rating / score_tag) are wrapped as one-element lists before concatenation. 'location' is
# added by rebinding rather than .append, so no list shared with json_data is mutated (defensive, as in
# the reference). The caller passes enable_drop_tags through extract_r18_tags to split out R18 tags.
def _get_tags_annotation_anime(json_data):
    character_tags = json_data.get("tag_character", [])
    artist_tags = json_data.get("tag_artist", [])
    copyright_tags = json_data.get("tag_copyright", [])
    general_tags = json_data.get("tag_general", [])
    meta_tags = json_data.get("tag_meta", [])
    meta_tags = list(set(meta_tags) - _REMOVED_META_TAGS_ANIME)
    years_tags = json_data.get("year", "")
    quality_tags = json_data.get("quality_tag", "")
    dataset_type = json_data["dataset_type"]
    tag_nsfw = json_data["tag_nsfw"]
    rating = json_data["rating"]
    score_tag = json_data["score_tag"]
    enable_drop_tags = (
        general_tags + meta_tags
        + [years_tags] + [quality_tags]
        + copyright_tags
        + [tag_nsfw] + [rating] + [score_tag]
    )
    kept_tags = character_tags + artist_tags + dataset_type
    if json_data["location"] == True:
        kept_tags = kept_tags + ['location']
    return kept_tags, enable_drop_tags

# Core prompt builder for anime t2i, registered as T2I_PROMPT_BUILDERS["t2i_danbooru_yande_posttraining"]
# and called from dataset._decode_t2i. The steps below follow the reference exactly and must not be
# reordered, since the order of the random.* calls determines reproducibility:
#   (a) read the category number and sample_lang = sample_language(keys of scene / text_desc /
#       image_description / comic_desc / r18_comic_desc);
#   (b) category 7 or 12: return the comic_desc / r18_comic_desc text directly (get_text_by_lang);
#   (c) otherwise build whole_prompt in the order bg_type -> text_category -> use_tag -> six-way
#       tag_strategy, then random.shuffle(whole_prompt) and ". ".join(...);
#   (d) bg_type 1 adds no background text, 2 / 3 add the scene text, anything else adds the scene text
#       with a warning;
#   (e) when tag_strategy >= 0.5, sample_reg_tokens() is appended to enable_drop_tags.
# May return ""; _decode_t2i drops such samples (`if not text.strip(): return None`) and catches all
# exceptions. The returned string goes to the tokenizer as is.
def build_prompt_danbooru_yande_posttraining(json_data):
    r18_tags = load_r18_tags()
    category_number = json_data["image_category_number"]
    sample_lang = sample_language([
        list(json_data.get("scene", {}).keys()),
        list(json_data.get("text_desc", {}).keys()),
        list(json_data.get("image_description", {}).keys()),
        list(json_data.get("comic_desc", {}).keys()),
        list(json_data.get("r18_comic_desc", {}).keys()),
    ])

    if category_number == 7 or category_number == 12:
        if category_number == 7:
            prompt = json_data.get("comic_desc", {})
        elif category_number == 12:
            prompt = json_data.get("r18_comic_desc", {})
        return get_text_by_lang(prompt, sample_lang)

    whole_prompt = []
    bg_type_raw = json_data.get("bg_type_number", 1)
    if bg_type_raw is None:
        bg_type_raw = 1
    bg_type = int(bg_type_raw)
    if bg_type == 1:
        bg_prompt = ""
    elif bg_type == 2 or bg_type == 3:
        scene = json_data.get("scene", "")
        bg_prompt = get_text_by_lang(scene, sample_lang)
    else:
        print("bg type number = 4, unknown bg type")
        bg_prompt = get_text_by_lang(json_data.get("scene", {}), sample_lang)
    whole_prompt.append(bg_prompt)

    text_category = str(json_data.get("text_category", "0"))
    if text_category == "0":
        text_prompt = ""
    elif text_category == "1" or text_category == "2":
        text_prompt = get_text_by_lang(json_data.get("text_desc", {}), sample_lang)
    else:
        print(f"Unknown text_category: {text_category}")
        text_prompt = ""
    whole_prompt.append(text_prompt)

    kept_tags, enable_drop_tags = _get_tags_annotation_anime(json_data)
    enable_drop_tags, r18_sampled = extract_r18_tags(enable_drop_tags, r18_tags)
    use_tag = random.random() > 0.5
    if use_tag:
        tag_strategy = random.random()
        if tag_strategy < 0.5:
            tags_prompt = kept_tags + enable_drop_tags + r18_sampled
        else:
            length = len(enable_drop_tags)
            if tag_strategy < 0.6:
                enable_drop_tags = random.sample(enable_drop_tags, max(int(0.3 * length), min(10, length)))
            elif tag_strategy < 0.75:
                enable_drop_tags = random.sample(enable_drop_tags, max(int(0.4 * length), min(15, length)))
            elif tag_strategy < 0.85:
                enable_drop_tags = random.sample(enable_drop_tags, min(6, length))
            elif tag_strategy < 0.95:
                enable_drop_tags = random.sample(enable_drop_tags, min(4, length))
            else:
                enable_drop_tags = []
            enable_drop_tags.extend(sample_reg_tokens())
            tags_prompt = kept_tags + enable_drop_tags + r18_sampled
        random.shuffle(tags_prompt)
        tags_prompt = ", ".join(tags_prompt)
        whole_prompt.append(tags_prompt)
    else:
        natural_prompt = get_text_by_lang(json_data.get("image_description", {}), sample_lang)
        whole_prompt.append(natural_prompt)
        random.shuffle(kept_tags)
        whole_prompt.append(", ".join(kept_tags))

    random.shuffle(whole_prompt)
    return ". ".join(whole_prompt)

# Real-people images (the reference DanbooruWebDatasetT2IRealNew_Posttraning dataset) use a different tag
# layout. Returns three lists (one more than the anime version: h_tag) so that the caller can treat the
# R18 tags separately. Pure-function version of the reference get_tags_annotation.
# h_tag is returned as is (tag_dict.get("h_tag", [])). Unlike the anime extract_r18_tags, the 3/4
# subsampling happens in the builder's tag_strategy >= 0.5 branch; subsampling here would hand the
# tag_strategy < 0.5 branch the wrong R18 list.
# Returns (kept_tags = model_name + channel_tag + dataset_type, general_tags = tagger_tag + post_tag +
# extra_tag, h_tag).
def _get_tags_annotation_realpeople(tag_dict, extra_meta):
    tagger_tag = tag_dict.get("tagger_tag", [])
    htag = tag_dict.get("h_tag", [])
    post_tag = tag_dict.get("post_tag", [])
    extra_tag = tag_dict.get("extra_tag", [])
    dataset_type = tag_dict.get("dataset_type", [])
    channel_tag = extra_meta.get("channel_tag", [])
    model_name = extra_meta.get("model_name", [])

    kept_tags = model_name + channel_tag + dataset_type
    general_tags = tagger_tag + post_tag + extra_tag
    return kept_tags, general_tags, htag

# Core prompt builder for real-people t2i. The RNG call order below follows the reference literally and
# must not be unified with the anime builder:
#   (1) tag_strategy = random.random() comes first (the anime builder draws it only when use_tag is True);
#   (2) read tags_dict / extra_meta / text_fields; is_japanese = "everidol" or "gravia" in image_name;
#   (3) sample_lang from the four lang-key lists of text_fields;
#   (4) append ", ".join(get_keywords_by_lang(text_fields["keywords"], sample_lang));
#   (5) append get_text_by_lang(text_fields["camera_shot"], sample_lang);
#   (6) use_tag = random.random() < 0.5;
#   (7) kept_tags, enable_drop_tags, r18_sampled = _get_tags_annotation_realpeople(tags_dict, extra_meta);
#   (8) the six-way tag_strategy branches build tags_list (the >= 0.5 branches keep 3/4 of the R18 tags
#       and inject REG tokens);
#   (9) if is_japanese: kept_tags.append("japanese"), after (8) and before (10);
#   (10) use_tag: shuffle and join tags_list; otherwise the text_fields["character"] description plus
#        kept_tags (which now include "japanese");
#   (11) append text_fields["scene"] as bg_prompt;
#   (12) random.shuffle(whole_prompt); ". ".join(...).
# The input is the sample's json dict; dataset.py catches all exceptions.
#
# Why step (9) must sit exactly there: tags_list = kept_tags + enable_drop_tags + r18_* has already copied
# kept_tags into a new list, so the later append does not reach tags_list and the use_tag=True output has
# no "japanese", while the use_tag=False branch shuffles and joins the mutated kept_tags and does include
# it. Moving the append before (8) would leak "japanese" into the use_tag=True path (a silent shift of the
# data distribution); moving it after `if use_tag:` would lose the tag in the use_tag=False branch.
def build_prompt_realpeople_post_training(json_data):
    tag_strategy = random.random()
    tags_dict = json_data.get("tags_dict", {})
    extra_meta = json_data.get("extra_meta", {})
    text_fields = json_data.get("text_fields", {})
    image_name = json_data["image_name"]
    is_japanese = "everidol" in image_name or "gravia" in image_name

    whole_prompt = []
    sample_lang = sample_language([
        text_fields.get("keywords_lang", []),
        text_fields.get("camera_shot_lang", []),
        text_fields.get("character_lang", []),
        text_fields.get("scene_lang", []),
    ])

    camera_keywords = text_fields.get("keywords", "")
    keywords_prompt = get_keywords_by_lang(camera_keywords, sample_lang)
    whole_prompt.append(", ".join(keywords_prompt))
    whole_prompt.append(get_text_by_lang(text_fields.get("camera_shot", {}), sample_lang))

    use_tag = random.random() < 0.5
    kept_tags, enable_drop_tags, r18_sampled = _get_tags_annotation_realpeople(tags_dict, extra_meta)
    if tag_strategy < 0.5:
        tags_list = kept_tags + enable_drop_tags + r18_sampled
    else:
        length = len(enable_drop_tags)
        if tag_strategy < 0.6:
            enable_drop_tags = random.sample(enable_drop_tags, max(int(0.3 * length), min(10, length)))
        elif tag_strategy < 0.75:
            enable_drop_tags = random.sample(enable_drop_tags, max(int(0.4 * length), min(15, length)))
        elif tag_strategy < 0.85:
            enable_drop_tags = random.sample(enable_drop_tags, min(6, length))
        elif tag_strategy < 0.95:
            enable_drop_tags = random.sample(enable_drop_tags, min(4, length))
        else:
            enable_drop_tags = []
        enable_drop_tags.extend(sample_reg_tokens())
        if len(r18_sampled) > 1:
            r18_sampled_new = random.sample(r18_sampled, max(1, len(r18_sampled) * 3 // 4))
        else:
            r18_sampled_new = r18_sampled
        tags_list = kept_tags + enable_drop_tags + r18_sampled_new

    # Must come after tags_list is built and before the use_tag branch; see the note above the function.
    if is_japanese:
        kept_tags.append("japanese")

    if use_tag:
        random.shuffle(tags_list)
        tags = ", ".join(tags_list)
        whole_prompt.append(tags)
    else:
        fg_prompt = text_fields.get("character", "")
        fg_prompt = get_text_by_lang(fg_prompt, sample_lang)
        whole_prompt.append(fg_prompt)
        random.shuffle(kept_tags)
        whole_prompt.append(", ".join(kept_tags))

    bg_prompt = text_fields.get("scene", "")
    bg_prompt = get_text_by_lang(bg_prompt, sample_lang)
    whole_prompt.append(bg_prompt)
    random.shuffle(whole_prompt)
    return ". ".join(whole_prompt)

# Builder registry used by get_t2i_prompt_builder; the `prompt_builder` field of a data config names one
# of these keys. The older entries (auto / t2i_auto / t2i_114w_tagdict / t2i_danbooru /
# t2i_simple_caption) are kept for existing configs.

T2I_PROMPT_BUILDERS = {
    "auto": build_prompt_auto,
    "t2i_auto": build_prompt_auto,
    "t2i_114w_tagdict": build_prompt_114w_tagdict,
    "t2i_danbooru": build_prompt_danbooru,
    "t2i_simple_caption": build_prompt_simple_caption,
    "t2i_danbooru_yande_posttraining": build_prompt_danbooru_yande_posttraining,
    "t2i_realpeople_post_training": build_prompt_realpeople_post_training,
}


def get_t2i_prompt_builder(name):
    try:
        return T2I_PROMPT_BUILDERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(T2I_PROMPT_BUILDERS))
        raise ValueError(f"Unknown t2i prompt_builder '{name}'. Available: {available}") from exc
