import os
import random
import sys
import csv

def as_list(value):
    # Normalize any input to list[str]: a list is cleaned element-wise, "a, b" is split as CSV, a single
    # str is wrapped as [str], anything else goes through str().
    # Note: None inside a list becomes the string "None" instead of being filtered out.
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]
        tag = value.strip()
        return [tag] if tag else []
    tag = str(value).strip()
    return [tag] if tag else []


def join_tags(tags, shuffle=True):
    # Join list[str] into "a, b, c"; shuffled by default (tag order carries no meaning; random.shuffle
    # works in place). The separator must stay ", " (matches the tokenizer vocabulary).
    tags = [str(tag).strip() for tag in tags]
    tags = [tag for tag in tags if tag]
    if shuffle:
        random.shuffle(tags)
    return ", ".join(tags)


def first_text(mapping, keys):
    # First field, in `keys` order, that yields non-empty text. text_value is called once per key (no
    # caching, so the RNG is not consumed more than once per key); an empty string counts as not found.
    for key in keys:
        value = mapping.get(key)
        text = text_value(value)
        if text:
            return text
    return ""


def text_value(value):
    # Normalize a single value to str: a str is stripped; a list yields one random element (consumes
    # RNG!); any other type returns "" (no str() coercion, unlike as_list).
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        texts = [str(v).strip() for v in value if str(v).strip()]
        return random.choice(texts) if texts else ""
    return ""


def weighted_multilingual(mapping, keys):
    weighted_keys = random.choices(list(keys), weights=[0.5, 0.25, 0.25], k=1)
    value = mapping.get(weighted_keys[0], "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return first_text(mapping, keys)


def danbooru_tags(tags_dict, include_copyright=False):
    parts = []
    for key in (
        "tag_string_character",
        "tag_string_artist",
        *(() if not include_copyright else ("tag_string_copyright",)),
        "tag_string_general",
        "tag_string_meta",
        "years",
        "quality_tags",
        "quality_tag",
    ):
        parts.extend(as_list(tags_dict.get(key)))
    if parts and "anime" not in parts:
        parts.append("anime")
    return parts


def rgb(image):
    if image is None:
        return None
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return image


def valid_latent(latent):
    import torch

    return latent is not None and not torch.isnan(latent).any() and not torch.isinf(latent).any()


def expand_shards(shards):
    import webdataset as wds

    if isinstance(shards, str):
        return list(wds.shardlists.expand_urls(shards))
    if isinstance(shards, (list, tuple)):
        urls = []
        for item in shards:
            urls.extend(wds.shardlists.expand_urls(item))
        return urls
    return shards

# --- AVAILABLE_LANGS, LANGUAGE_WEIGHTS ---
# Each training prompt is generated in one randomly chosen language: English 50%, Chinese 25%,
# Japanese 25%. Kept in sample_utils rather than prompt_builders_t2i so other builders can reuse the
# same table; must be defined before _normalize_langs / sample_language / get_text_by_lang /
# get_keywords_by_lang. The multilingual helpers fall back through AVAILABLE_LANGS in order, and
# LANGUAGE_WEIGHTS must have a key for every entry of AVAILABLE_LANGS.
AVAILABLE_LANGS = ["en", "cn", "jp"]
LANGUAGE_WEIGHTS = {"en": 0.5, "cn": 0.25, "jp": 0.25}

# --- _normalize_langs ---
# Clean any input (str / list[str] / None / invalid) into a list of valid language codes, a subset of
# AVAILABLE_LANGS. sample_language receives the per-field lang keys of the webdataset json, whose type
# varies.
def _normalize_langs(langs):
    # Literally identical to the reference _normalize_langs; do not change the logic (no extra filter
    # branches), it affects RNG consistency.
    if isinstance(langs, str):
        langs = [langs] 
    elif not isinstance(langs, list):
        langs = []
    return [lang for lang in langs if lang in AVAILABLE_LANGS]

# --- sample_language ---
# Pick one language supported by all of a sample's text fields (scene, text description, character
# description, ...), so that every part of the prompt uses the same language:
#   1. collect the languages each field supports (e.g. scene has en/cn, text_desc has en/jp);
#   2. intersect them; if the intersection is empty, fall back to the union;
#   3. draw one by LANGUAGE_WEIGHTS.
# Returns "en" if there is no valid language information at all; never returns "" / None.
# The input is a list with one entry per field, each a str / list[str] / invalid value. Called once per
# sample; shared by the anime and realpeople builders, which pass different fields but use the same
# merge / sampling rule. Callers then fetch every field with this language via get_text_by_lang /
# get_keywords_by_lang.
def sample_language(lang_candidates):
    normalized = [set(_normalize_langs(langs)) for langs in lang_candidates if _normalize_langs(langs)]
    if not normalized:
        return "en"
    common_langs = set.intersection(*normalized) if len(normalized) > 1 else normalized[0]
    candidate_langs = [lang for lang in AVAILABLE_LANGS if lang in common_langs]
    if not candidate_langs:
        merged_langs = set.union(*normalized)
        candidate_langs = [lang for lang in AVAILABLE_LANGS if lang in merged_langs]
    weights = [LANGUAGE_WEIGHTS[lang] for lang in candidate_langs]
    return random.choices(candidate_langs, weights=weights)[0]

# --- get_text_by_lang ---
# Text of a multilingual field in the given language, with the fallback chain
# lang -> default_lang -> any available language (in AVAILABLE_LANGS order).
# The field may be a str (passed through as is, for the old schema), a dict {lang: text}, or missing /
# of another type (returns ""). `lang` comes from sample_language. May return ""; callers handle that
# through ". ".join / `if not text`.
def get_text_by_lang(field, lang, default_lang="en"):
    if isinstance(field,str):
        return field
    if not isinstance(field, dict):
        return ""
    if lang in field and field[lang]:
        return field[lang]
    if default_lang in field and field[default_lang]:
        return field[default_lang]
    for fallback_lang in AVAILABLE_LANGS:
        if fallback_lang in field and field[fallback_lang]:
            return field[fallback_lang]
    return ""

# --- get_keywords_by_lang ---
# Like get_text_by_lang, but for list-valued fields (e.g. keyword lists); returns list[str] instead of
# str. realpeople's text_fields["keywords"] is a list[str] in the schema but occasionally a dict split
# by language, hence the tolerance for several types. Separate from get_text_by_lang because the output
# type differs: callers do ", ".join(keywords), which may give "".
def get_keywords_by_lang(field, lang, default_lang="en"):
    # list / str are normalized directly; a dict falls back lang -> default_lang -> AVAILABLE_LANGS and is
    # then normalized. The output is always list[str] without empty strings.
    if isinstance(field, list):
        return [item for item in field if item]
    if isinstance(field, str):
        return [field] if field else []
    if not isinstance(field, dict):
        return []
    values = field.get(lang)
    if not values:
        values = field.get(default_lang)
    if not values:
        for fallback_lang in AVAILABLE_LANGS:
            values = field.get(fallback_lang)
            if values:
                break
    if isinstance(values, list):
        return [item for item in values if item]
    if isinstance(values, str):
        return [values] if values else []
    return []

# --- R18 tag vocabulary ---
# Module-level cache so the file (under $KOKORO_TAG_FILES_DIR) is not reread for every sample.
# load_r18_tags is the only writer; extract_r18_tags does not read the cache and gets the set as an
# argument instead.
_R18_TAGS_CACHE = None
_R18_TAGS_PATH = os.path.join(os.environ.get("KOKORO_TAG_FILES_DIR", ""), "R18_motion_tags.txt")

# --- load_r18_tags ---
# On the first call, read R18_motion_tags.txt and cache its non-empty stripped lines as a set (O(1)
# lookups); later calls return the cache. The anime builder calls this once per sample, so only the
# first call does IO. Callers must treat the returned set as read-only.
# If the file is missing, a [FATAL] line is printed and None is returned. The anime builder then fails
# on every sample, and dataset._decode_t2i swallows exceptions (`except Exception: return None`), so the
# dataloader yields nothing and hangs with that line as the only clue. The stricter variant (print to
# stderr with flush=True, then raise) is kept commented out below.
def load_r18_tags():
    global _R18_TAGS_CACHE
    if _R18_TAGS_CACHE is None:
        try:
            with open(_R18_TAGS_PATH,"r") as f:
                _R18_TAGS_CACHE = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(_R18_TAGS_CACHE)} R18 tags")
        except FileNotFoundError as e:
            print(f"[FATAL] R18 tags file not found: {_R18_TAGS_PATH}")
            # print(f"[FATAL] R18 tags file not found: {_R18_TAGS_PATH}", file=sys.stderr, flush=True)
            # raise RuntimeError(f"R18 tags file not found: {_R18_TAGS_PATH}") from e
    return _R18_TAGS_CACHE

# --- NO_DROP tag vocabulary ---
# Module-level cache so the story builder does not reread the 13 MB+ CSV (under $KOKORO_TAG_FILES_DIR)
# for every sample. load_no_drop_tags is the only writer; apply_tag_drop_strategy gets the set as an
# argument instead of reading the module global. Deliberately not merged with the R18 loader into a
# generic vocab loader: the two are built differently (NO_DROP adds two threesome tags, R18 adds none).
_NO_DROP_TAGS_CACHE = None
_NO_DROP_TAGS_PATH = os.path.join(os.environ.get("KOKORO_TAG_FILES_DIR", ""), "Aplus-tag_deduplicated_with_counts.csv")

# --- load_no_drop_tags ---
# On the first call, read the 'tag' column of the CSV (utf-8, with header), add "MMF threesome" and
# "FFM threesome", and cache the result as a set; later calls return the cache. The story builder calls
# this once per sample, so only the first call does IO. Callers must treat the returned set as read-only.
# If the file is missing, a [FATAL] line is printed and an empty set is cached, as in the reference: every
# tag then becomes droppable, which silently shifts the prompt distribution, so watch for that line.
# A CSV without a 'tag' column raises KeyError. The stricter variant (print to stderr, then raise) is
# kept commented out below.
def load_no_drop_tags():
    global _NO_DROP_TAGS_CACHE
    if _NO_DROP_TAGS_CACHE is None:
        try:
            with open(_NO_DROP_TAGS_PATH, "r", encoding="utf-8") as f:
                csv_reader = csv.DictReader(f)
                no_drop_tags = [row["tag"] for row in csv_reader]
            no_drop_tags.extend(["MMF threesome", "FFM threesome"])
            _NO_DROP_TAGS_CACHE = set(no_drop_tags)
            print(f"Loaded {len(_NO_DROP_TAGS_CACHE)} NO_DROP tags")
            print(no_drop_tags[:10])  # preview the first 10 tags
        except FileNotFoundError as e:
            print(f"[FATAL] NO_DROP tags file not found: {_NO_DROP_TAGS_PATH}; falling back to the default sampling logic")
            _NO_DROP_TAGS_CACHE = set()
            # print(f"[FATAL] NO_DROP tags file not found: {_NO_DROP_TAGS_PATH}; falling back to the default sampling logic", file=sys.stderr, flush=True)
            # raise RuntimeError(f"NO_DROP tags file not found: {_NO_DROP_TAGS_PATH}") from e
    return _NO_DROP_TAGS_CACHE

# --- apply_tag_drop_strategy ---
# Six-way random strategy over a tag list (keep all / subsample / drop all); tags in no_drop_set are
# always kept. Used by the i2i story builder for the tag -> text step. Literal port of the body of
# get_dropped_tags in the reference Danbooru edit dataset, without its tag_frequencies output (no tag
# statistics here) and without its commented-out "anime coloring" removal branch.
# Invariant: the thresholds (0.5 / 0.6 / 0.75 / 0.85 / 0.95), the number and arguments of the
# random.sample calls in each branch and the final random.shuffle must match the reference exactly.
# They consume the global RNG, and the data order of trained checkpoints depends on that sequence; any
# "equivalent but cleaner" refactor (merging the two list comprehensions, changing the sample call,
# another RNG source, ...) would shift the data order when training resumes. Per call: one branch draw,
# at most one sample, one shuffle.
# Input: tags as list[str] or a CSV string "a, b, c" (split on ","), typically img_meta["diff_tags"];
# no_drop_set must not be None (it comes from load_no_drop_tags and is empty when the CSV is missing).
# Output: the ", "-joined tag text; may be "" (0.95 branch with no no_drop tags), in which case
# _pack_i2i_sample drops the sample.
def apply_tag_drop_strategy(tags, no_drop_set):
    # In the str branch, `tag in no_drop_set` tests the raw, unstripped split element (as the reference
    # does); strip only applies to the strings that are kept.
    if isinstance(tags, list):
        no_drop_tags = [tag.strip() for tag in tags if tag in no_drop_set]
        rest_tags = [tag.strip() for tag in tags if tag not in no_drop_set]
    elif isinstance(tags, str):
        no_drop_tags = [tag.strip() for tag in tags.split(",") if tag in no_drop_set]
        rest_tags = [tag.strip() for tag in tags.split(",") if tag not in no_drop_set]
    else:
        # Fallback: treat an unknown input type as no tags rather than raising; the empty prompt is then
        # filtered by _pack_i2i_sample, as for the other builders.
        no_drop_tags, rest_tags = [], []

    tag_strategy = random.random()
    if tag_strategy < 0.5:
        tag_text_list = no_drop_tags + rest_tags
    else:
        length = len(rest_tags)
        if tag_strategy < 0.6:
            keeped_tags = random.sample(rest_tags, max(int(0.3 * length), min(10, length)))
        elif tag_strategy < 0.75:
            keeped_tags = random.sample(rest_tags, max(int(0.4 * length), min(15, length)))
        elif tag_strategy < 0.85:
            keeped_tags = random.sample(rest_tags, min(6, length))
        elif tag_strategy < 0.95:
            keeped_tags = random.sample(rest_tags, min(4, length))
        else:
            keeped_tags = []
        tag_text_list = no_drop_tags + keeped_tags
    random.shuffle(tag_text_list)
    return ", ".join(tag_text_list)


# --- extract_r18_tags ---
# Split the R18 tags out of a tag list and keep a random 3/4 of them (no sampling with <= 1 hit).
# The anime builder applies this to enable_drop_tags, feeds `remaining` into its tag strategy and
# appends r18_found back in every branch. r18_tags_set is passed in (load_r18_tags is not called here)
# to keep the function testable; literally identical to the reference. tag.replace(" ", "_") is
# required: the R18 vocabulary uses underscores while tag fields use spaces.
def extract_r18_tags(enable_drop_tags, r18_tags_set):
    r18_tags = [tag for tag in enable_drop_tags if tag.replace(" ", "_") in r18_tags_set]
    remaining = [tag for tag in enable_drop_tags if tag.replace(" ", "_") not in r18_tags_set]
    if len(r18_tags) > 1:
        num_keep = max(1, len(r18_tags) * 3 // 4)
        r18_found = random.sample(r18_tags, num_keep)
    else:
        r18_found = r18_tags
    return remaining, r18_found

# --- REG_TOKENS ---
# When training drops part of the tags at random, these REG placeholder tokens are inserted into the
# prompt, telling the model "there was information here, but it has been hidden". Having learned this,
# the model can fill in details from the placeholders at inference even when the user's prompt is
# incomplete. sample_reg_tokens() returns a non-empty subset (1-4) of them; the anime and realpeople
# builders call it in their tag_strategy >= 0.5 branches and extend enable_drop_tags with the result.

# Fixed placeholder literals, tied to the trained checkpoint's vocabulary: do not rename, add or remove.
REG_TOKENS = ['[REG0]', '[REG1]', '[REG2]', '[REG3]'] 

def sample_reg_tokens():
    # Draw a non-empty subset of REG_TOKENS uniformly via a 4-bit mask. randint(1, 15) already excludes 0,
    # so the while True loop is redundant, but it is kept on purpose to match the reference RNG usage.
    while True:
        mask = random.randint(1, 15)  # 4-bit mask, all-zero excluded (1-15)
        chosen = [REG_TOKENS[i] for i in range(4) if (mask & (1 << i))]
        if chosen:
            break
    return chosen