"""Z-Image caption rules for the BAGEL readers of the Z-Image packs.

Snapshot of the caption assembly of the reference training pipeline, copied unchanged except that
VENDOR_DIR / SHARE now come from environment variables. Do not modify this file or the caption rules it
loads: the same json and the same (seed, epoch, key) must give a prompt byte-identical to the reference.

The prompt of every Z-Image-packed sample (full_pretrain* t2i, i2i_full_pack* / ffmgal i2i) is
assembled at read time from the sample's `source_row` exactly as Z-Image training did it: same code
(the pinned Kokoro2 caption rules, loaded from KOKORO2_CAPTION_DIR), same settings (CAPTION_ENV below,
taken from the configuration of the reference Z-Image run on these packs), same sample dict
(wds_latent._make_sample), same side tables attached in the same order (wds_latent.to_gpu_samples),
same no-NL filter (wds_latent._keep_no_nl).

What differs from Z-Image on purpose:
  * the per-sample RNG key. Z-Image seeds each sample's caption draw with (seed, epoch, key) and
    its key is the member name inside the tar ("00000123"), which repeats in every shard; here the
    key is "<pack dir>/<shard file>/<member>", unique across the pool.
  * CFG text dropout is not done here: BAGEL drops the text segment itself
    (text_cond_dropout_prob), so the assembler must not drop again.
  * the text budget (1012 tokens) is still counted with the Z-Image (Qwen3) tokenizer, as in
    Z-Image; the BAGEL tokenizer (Qwen2.5) then encodes the finished string.
"""
import os
import sys

# The pinned Kokoro2 caption package is not shipped with this repo: point KOKORO2_CAPTION_DIR at a
# directory holding `Kokoro2/` (version given in the README) and KOKORO_SHARE_DIR at the side-table root.
VENDOR_DIR = os.environ.get('KOKORO2_CAPTION_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kokoro2_caption'))
SHARE = os.environ.get('KOKORO_SHARE_DIR', '')

# Caption settings of the reference Z-Image run (from its env / launch scripts and resolved config).
# Variables that run left unset stay unset here too (the modules then use their own defaults, as they
# did there).
CAPTION_ENV = {
    'PROMPT_MODE': 'material',
    'PROMPT_RULES': 'caption0819',
    'PROMPT_MODE_PER_SOURCE': '{"multiref_pack_512":"baked"}',
    'TAG_SPACIFY_P': '1.0',
    'TEXT_MAX_TOKENS': '1024',
    'DROP_NO_NL': '1',
    'NL_SIDETABLE_DIR': f'{SHARE}/_sidetable_nl_0831',
    'POPULARITY_SIDETABLE_DIR': f'{SHARE}/_sidetable_popularity_0819',
    'DANBOORU_SIDETABLE_DIR': f'{SHARE}/_sidetable_danbooru_0819',
    'SRCMETA_SIDETABLE_DIR': f'{SHARE}/_sidetable_srcmeta_0821',
    'SRCTAGS_SIDETABLE_DIR': f'{SHARE}/_sidetable_srctags_0821',
    'BBOXNL_SIDETABLE_DIR': f'{SHARE}/_sidetable_bboxnl_0821',
    'PORNSTARS_SIDETABLE_DIR': f'{SHARE}/_sidetable_pornstars_v2_512_0827',
    # Byte-identical copy of the Z-Image tokenizer (checksums in model/Z-Image-aux/SHA256SUMS).
    'CAPTION_TOKENIZER': f'{SHARE}/model/Z-Image-aux/tokenizer',
}
# Set by the reference run's config / env but never read by the caption code: TEXT_PAD (encoder
# padding), REF_* (reference image path, see the i2i reader).

_CONFIGURED = False


def configure():
    """Export CAPTION_ENV (an explicitly set variable wins, and is reported) and import the rules.

    Called once by the loader builder in the main process, before the dataloader forks, so every
    worker inherits the same environment.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    for name, value in CAPTION_ENV.items():
        have = os.environ.get(name)
        if have is None:
            os.environ[name] = value
        elif have != value:
            print(f'[zimage_caption] {name}={have!r} from the environment overrides the pinned '
                  f'value {value!r}', flush=True)
    for name in ('NL_SIDETABLE_DIR', 'POPULARITY_SIDETABLE_DIR', 'DANBOORU_SIDETABLE_DIR',
                 'SRCMETA_SIDETABLE_DIR', 'SRCTAGS_SIDETABLE_DIR', 'BBOXNL_SIDETABLE_DIR',
                 'PORNSTARS_SIDETABLE_DIR', 'CAPTION_TOKENIZER'):
        if not os.path.isdir(os.environ[name]):
            # A missing side table is not an error in the Z-Image code (that table's tags just
            # vanish); here it would silently change the captions, so stop.
            raise FileNotFoundError(f'[zimage_caption] {name}={os.environ[name]} does not exist')
    if VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)
    import Kokoro2
    if os.path.dirname(os.path.dirname(Kokoro2.__file__)) != VENDOR_DIR:
        raise ImportError(f'[zimage_caption] Kokoro2 imported from {Kokoro2.__file__}, not the pinned copy '
                          f'in {VENDOR_DIR}; another Kokoro2 tree is on sys.path')
    from Kokoro2.data import prompt_assemble as pa
    if pa.prompt_mode() != 'material' or pa.rules_version() != 'caption0819':
        raise RuntimeError(f'[zimage_caption] rules resolved to {pa.prompt_mode()}/{pa.rules_version()}')
    _CONFIGURED = True


def make_sample(shard, member_key, meta, latent, refs, seed, epoch):
    """The CPU-side sample dict of Kokoro2/data/wds_latent.py:_make_sample (field for field) plus the
    (seed, epoch) stamp of epoch_batches. `shard` is the tar path, `member_key` its member name."""
    row = meta.get('source_row') or {}
    pack_dir = os.path.dirname(shard)
    return {
        'key': f'{os.path.basename(pack_dir)}/{os.path.basename(shard)}/{member_key}',
        'prompt': meta['prompt'],
        'latent': latent,
        'refs': refs,
        'prompt_material': row.get('prompt_material'),
        'prompt_meta': row.get('meta'),
        'prompt_image': row.get('image'),
        'cross_ref': meta.get('n_ref', len(refs)) != 0,
        'prompt_source': pack_dir or '?',
        'prompt_seed': seed,
        'prompt_epoch': epoch,
    }


# ---- Kokoro2/data/wds_latent.py of the pinned caption rules, verbatim except the logger -------------
NL_REQUIRED_CLASSES = frozenset({"A", "B", "D"})
_NL_DROPPED = {}


def _has_nl(material):
    nl = (material or {}).get("nl")
    if isinstance(nl, dict):
        if any((v or "").strip() for v in nl.values()):
            return True
    elif (nl or "").strip():
        return True
    for var in ((material or {}).get("nl_variants") or {}).values():
        if isinstance(var, dict) and any((v or "").strip() for v in var.values()):
            return True
    return False


def keep_no_nl(smp):
    """True = keep. DROP_NO_NL: t2i rows of the A/B/D classes without any natural language go;
    i2i rows and the tag-only anime classes are never dropped."""
    if os.environ.get('DROP_NO_NL', '0') != '1':
        return True
    from Kokoro2.data import caption_rules as _cr
    from Kokoro2.data import i2i_caption_rules as _ir
    from Kokoro2.data.nl_segments_schema import SOURCE_CLASS as _CLS

    if _ir.source_of_i2i(smp.get("prompt_source")) is not None:
        return True
    src = _cr.source_of(smp)
    if _CLS.get(src) not in NL_REQUIRED_CLASSES:
        return True
    material = smp.get("prompt_material")
    if material is None:
        raise ValueError(f"[zimage_caption] DROP_NO_NL=1 but sample {smp.get('key')!r} ({src}) has no prompt_material")
    if _has_nl(material):
        return True
    _NL_DROPPED[src] = _NL_DROPPED.get(src, 0) + 1
    n = _NL_DROPPED[src]
    if n == 1 or n % 10000 == 0:
        print(f'[zimage_caption] source {src}: dropped {n} samples without natural language (DROP_NO_NL=1)',
              flush=True)
    return False


def assemble(batch):
    """Prompts for a list of make_sample dicts: side tables attached batch-wise in Z-Image's order
    (wds_latent.to_gpu_samples), then resolve_prompt, then maybe_deid (identity: CAPTION_DEID unset)."""
    from Kokoro2.data.prompt_assemble import resolve_prompt
    from Kokoro2.data.caption_deid import maybe_deid
    from Kokoro2.data.nl_sidetable import attach_nl_segments
    from Kokoro2.data.side_tables import attach_side_tags
    from Kokoro2.data.src_tags import attach_src_tags
    from Kokoro2.data.bbox_nl import attach_bbox_nl
    from Kokoro2.data.pornstars import attach_pornstars
    attach_nl_segments(batch)
    attach_side_tags(batch)
    attach_src_tags(batch)
    attach_bbox_nl(batch)
    attach_pornstars(batch)
    return [maybe_deid(resolve_prompt(s)) for s in batch]


def i2i_source(prompt_source):
    """Z-Image's i2i source id for a pack directory (None = not an i2i source)."""
    from Kokoro2.data import i2i_caption_rules as _ir
    return _ir.source_of_i2i(prompt_source)
