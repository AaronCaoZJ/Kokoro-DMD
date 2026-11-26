"""Snapshot of the Z-Image pack reading logic of the reference training pipeline.

The code is copied rather than imported, so changes to the reference tree cannot silently change the
data read here. It covers:
  * the Z-Image pack reader: I2I_MODE / I2I_TOKEN_MODE (ehentai-diff / kemono-diff take the story
    token but are still sampled with the diff group), _path_key / load_heldout_keys / source_row_keys
    (held-out masking), iter_pack_shard (grouped tar reading), and the checks of _ZImagePackBase._fits /
    _target and of the reference images in ZImagePackI2I._build (as the module functions below).
    Resuming from a read position is not ported: FastGen does not store read progress, so a restart
    reads from the beginning (set another DATA_SEED for another order, see loader.py).
  * the distributed iterable dataset: the strided cross-rank / cross-worker shard split of
    set_epoch / get_data_paths_per_worker -> split_paths_for_rank / split_paths_for_worker.

Deviations from the reference, neither of which changes the results:
  1. load_heldout_keys takes an optional path (the reference only reads the env var
     KOKORO_HELDOUT_BLOCKLIST; here it comes from the yaml's heldout_blocklist and falls back to the
     env var when no path is given, as in the reference).
  2. The checks are module functions that return a skip-reason string instead of class methods that
     call self._skip; the conditions are identical.
"""
import ast
import io
import json
import os
import random
import tarfile

import torch

# ---- reference Z-Image pack reader (verbatim) ------------------------------------------------------

# i2i pack dir name -> mode token, following the earlier per-source readers: the -diff sources are what
# the danbooru_diff reader read, banana what the banana diff reader read; scale_edit is instruction
# editing of the same image. Everything else pairs two different shots of one subject (post pairs,
# video / animation frames, character references), which is what the story readers read.
I2I_MODE = {
    'danbooru-diff': 'diff', 'ehentai-diff': 'diff', 'kemono-diff': 'diff',
    'banana-cos': 'diff', 'banana-figure': 'diff', 'banana-fwd': 'diff', 'banana-socks-rev': 'diff',
    'scale_edit': 'diff',
    '1douyin': 'story', '1douyin-carousell': 'story', '1gangbang': 'story', '1ins': 'story',
    '1pornstarAB': 'story', '1pornstars': 'story', '2douyin-photographers': 'story', '2ehentai': 'story',
    '4fuck': 'story', '5fvck': 'story', 'coser': 'story', 'model': 'story', 'xiuren': 'story',
    'av': 'story', 'douyin': 'story', 'ins': 'story', 'pornpics': 'story', 'xhs': 'story',
    'youtube': 'story', 'animate-cut': 'story',
    'ffm_sr': 'story', 'galgame_sr': 'story', 'gal2_sr': 'story',
}

# Token-level overrides: ehentai-diff / kemono-diff pairs are gallery frame-to-frame changes
# (scene/camera move, pixels differ everywhere), not pixel-locked in-place edits, so they take
# the story token and story ref-CFG behavior. Group registration keeps using I2I_MODE, so these
# sources stay in the ZImageI2I_diff group and sampling is unchanged.
I2I_TOKEN_MODE = {**I2I_MODE, 'ehentai-diff': 'story', 'kemono-diff': 'story'}

# Mode token ids of the reference inferencer (the mode token id goes before encode(", " + instruction)).
# Here `text` is a string encoded by KokoroNetwork's tokenizer: "<|story_mode|>, xxx" is encoded as
# [151665] + encode(", xxx"), token-for-token identical to the reference, only if the tokenizer has
# <|story_mode|> / <|diff_mode|> registered with exactly these ids (add_special_tokens(..., split_diff_story=True)).
MODE_TOKEN_IDS = {'story': 151665, 'diff': 151666}
MODE_TOKEN_TEXT = {'story': '<|story_mode|>', 'diff': '<|diff_mode|>'}


# Held-out images: KOKORO_HELDOUT_BLOCKLIST names a file of image paths (one per line) that no
# reader trains on, neither as a target nor as an i2i reference. Matched on the last three path
# components (source dir / post / file), so the same image is caught in every pack and resolution
# (the held-out test images also appear in other packs).
def _path_key(p):
    return '/'.join(str(p).replace('\\', '/').split('/')[-3:])


def load_heldout_keys(path=None):
    # [deviation] optional path; without it, read the env var as the reference does
    if path is None:
        path = os.environ.get('KOKORO_HELDOUT_BLOCKLIST')
    if not path:
        return frozenset()
    keys = frozenset(_path_key(line.strip()) for line in open(path) if line.strip() and not line.startswith('#'))
    if not keys:
        raise ValueError(f'KOKORO_HELDOUT_BLOCKLIST={path} is empty')
    return keys


def source_row_keys(row):
    """Path keys of a source row's target image and its i2i reference images."""
    paths = [row.get('image')]
    edit = row.get('edit_image')
    if isinstance(edit, str) and edit.startswith('['):
        edit = ast.literal_eval(edit)
    paths += list(edit) if isinstance(edit, (list, tuple)) else [edit]
    return {_path_key(p) for p in paths if p}


def epoch_order(dataset_name, rank, worker_id, epoch, paths):
    """A reader's shard order in an epoch."""
    order = list(paths)
    random.Random(f'{dataset_name}-{rank}-{worker_id}-epoch{epoch}').shuffle(order)
    return order


def iter_pack_shard(path):
    """(key, {meta, latent, refs: [(j, webp bytes)]}) per sample; Kokoro2/data/wds_latent.py:iter_shard
    grouping (members of one sample are adjacent; `{key}_ref_{j}.webp` belongs to `{key}`)."""
    with tarfile.open(path, 'r') as tf:
        cur_key, cur = None, {}
        for m in tf:
            if not m.isfile():
                continue
            key, ext = os.path.splitext(os.path.basename(m.name))
            is_ref = ext == '.webp' and '_ref_' in key
            base_key = key.rsplit('_ref_', 1)[0] if is_ref else key
            if base_key != cur_key:
                if cur_key is not None and cur:
                    yield cur_key, cur
                cur_key, cur = base_key, {}
            if is_ref:
                cur.setdefault('refs', []).append((int(key.rsplit('_ref_', 1)[1]), tf.extractfile(m).read()))
            elif ext == '.json':
                cur['meta'] = json.loads(tf.extractfile(m).read())
            elif ext == '.pth':
                cur['latent'] = torch.load(io.BytesIO(tf.extractfile(m).read()), map_location='cpu')
        if cur_key is not None and cur:
            yield cur_key, cur


# ---- reference distributed iterable dataset: strided split across ranks / workers ------------------

def split_paths_for_rank(paths, rank, world_size, seed=42):
    """As DistributedIterableDataset.set_epoch(seed=42): sort -> shuffle with the seed -> rank r takes
    r, r+W, r+2W, ... Ranks get disjoint shards and every shard is read (a contiguous split would drop
    the remainder)."""
    data_paths = sorted(paths)
    rng = random.Random()
    rng.seed(seed)
    rng.shuffle(data_paths)
    return data_paths[rank::world_size]


def split_paths_for_worker(paths_rank, worker_id, num_workers):
    """As DistributedIterableDataset.get_data_paths_per_worker: a strided split of the rank's shards
    across workers, then reversed (as in the reference). num_workers=None means not inside a DataLoader
    worker (get_worker_info() is None in the reference): all of the rank's shards, not reversed."""
    if num_workers is None:
        return list(paths_rank)
    return paths_rank[worker_id::num_workers][::-1]


# ---- checks of _ZImagePackBase._fits / _target and ZImagePackI2I._build (same conditions, return a skip reason) ----

def fits(h, w, max_latent_size):
    """latent h, w (8x downsampled) within the position table (max_latent_size patches of 2)."""
    return h % 2 == 0 and w % 2 == 0 and h // 2 <= max_latent_size and w // 2 <= max_latent_size


def target_latent(lat, max_latent_size):
    """_ZImagePackBase._target: (latent [16, h, w] or None, skip reason or None)."""
    if lat.dim() != 4 or lat.shape[0] != 1 or lat.shape[1] != 16:
        return None, f'target latent shape {tuple(lat.shape)}'
    lat = lat[0]
    if not torch.isfinite(lat).all():
        return None, 'target latent not finite'
    if not fits(*lat.shape[1:], max_latent_size):
        return None, 'target larger than max_latent_size'
    return lat, None


def check_i2i_ref(ref_size, max_latent_size):
    """Reference-image check of ZImagePackI2I._build; ref_size is PIL's (w, h). Returns a skip reason or None."""
    rw, rh = ref_size
    if rh % 16 or rw % 16:
        return 'reference side not a multiple of 16'
    if not fits(rh // 8, rw // 8, max_latent_size):
        return 'reference larger than max_latent_size'     # galgame_sr / gal2_sr panoramas
    return None
