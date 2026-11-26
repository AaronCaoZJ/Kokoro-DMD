"""1024px Z-Image packs -> single samples for the FastGen inference branch.

Output (same format as the cfg-distill path kokoro_cfgdistill_data/dataset.py:_decode_t2i; the
DataLoader yields [sample] each time):
    task_type      "t2i" / "i2i"
    text           t2i: the prompt; i2i with i2i_mode_token=true: "<|story_mode|>, " / "<|diff_mode|>, " + prompt
                   (the reference training encodes [mode token id] + encode(", " + prompt); the string form
                   gives the same ids only if the tokenizer has these two tokens registered, see
                   pack_reader.MODE_TOKEN_IDS); with false, just the prompt
    vae_latent     target latent, bf16 [16, h, w] ({key}.pth from the pack with the batch dim removed, values untouched)
    source_image   i2i: {key}_ref_0.webp decoded to an RGB PIL image; t2i: None
  Plus a few read-only debug fields (unused by training): prompt (the prompt without the mode token), i2i_mode,
  key ("<pack dir>/<shard>/<member>"), epoch (which pass over that shard; part of the caption seed), source
  (e.g. "full_pretrain1024/pornpics", as data_indexes.source in the reference), image_keys (last three path
  components of the target and reference images, for held-out matching).

Reading follows the reference Z-Image pack reader (snapshotted in pack_reader.py / zimage_caption.py):
  * One yaml group = one reference dataset group: the shards of all dirs in the group are flattened, split
    across ranks first (sort -> shuffle with seed 42 -> rank r takes r, r+W, ...) and then strided across
    workers, so every rank and worker reads disjoint shards; the dirs of a group are therefore mixed in
    proportion to their sample counts (as in the reference).
  * Per sample: missing members / held-out (target or reference image in the blocklist) / n_ref mismatch ->
    skip; make_sample -> DROP_NO_NL -> prompts assembled in batches of 32 (reference caption_batch) -> target
    latent check ([1,16,h,w], finite, h/w even and h/2, w/2 <= max_latent_size) -> i2i only takes
    single-reference samples (as the reference ZImagePackI2I: multi-reference samples are skipped; every
    1024px pack dir has n_ref_hist all 1 in its dataset_info, so this never triggers in practice), with the
    reference side divisible by 16 and within max_latent_size -> shuffle buffer.
  * The i2i story / diff groups are drawn per sample by yaml weight (the reference PackedDataset draws per
    sample by group weight).
  * t2i / i2i is drawn by rank 0 each step and broadcast to all ranks (kokoro_inference_wds.MixedTaskLoader,
    reused as is).

Intentional differences from the reference:
  1. Each worker keeps open_shards shards open at once and picks one at random per sample (a reference worker
     reads one shard at a time to the end). A DMD run lasts only a few thousand iterations: with the
     reference scheme each worker would read only 1-3 shards over the whole run (an i2i shard holds ~900
     samples, a t2i shard 250-1000), so a global batch would come from only ranks x workers shards and
     sources would arrive in blocks. More open shards do not change the long-run proportions (each slot is a
     reference-style sequential reader); they only break up the short-term source blocks. open_shards=1
     reproduces the reference.
  2. The read-order RNG string includes a data seed (seed, default 42), so restarting with another DATA_SEED
     gives another order (FastGen does not store read progress; restarting with the same seed rereads the
     same shards from the start). The reference resumes from data_status; that is not ported.
  3. In the worker, reference images are only header-parsed to get their size for the checks; pixels are
     decoded when the sample leaves the buffer (the buffer holds only webp bytes).
  4. A caption batch of 32 mixes several shards (a reference batch comes from one shard); the prompt depends
     only on (seed, epoch, key) and the sample's json / side tables, not on its batch mates, so prompts match
     the reference byte for byte.
"""
from __future__ import annotations

import io
import os
import random
import traceback
from typing import Any, List, Optional

import torch.distributed as dist
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from fastgen.datasets.kokoro_cfgdistill_data import make_dataloader
from fastgen.datasets.kokoro_inference_wds import MixedTaskLoader
from fastgen.utils.logging_utils import logger

from . import zimage_caption
from .pack_reader import (
    I2I_MODE,
    I2I_TOKEN_MODE,
    MODE_TOKEN_TEXT,
    check_i2i_ref,
    epoch_order,
    fits,
    iter_pack_shard,
    load_heldout_keys,
    source_row_keys,
    split_paths_for_rank,
    split_paths_for_worker,
    target_latent,
)

CAPTION_BATCH = 32      # reference _ZImagePackBase.caption_batch (side tables are queried per batch)
SPLIT_SEED = 42         # reference DistributedIterableDataset.set_epoch(seed=42): fixed seed of the cross-rank split (must match on all ranks)


def _dist_rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


class PackGroup:
    """One yaml group (= one reference dataset group, e.g. ZImageT2I_1024). iter_worker() generates the
    samples inside each DataLoader worker; the object is copied into every worker by fork, so each worker
    keeps its own skip counts."""

    CHECK_I2I_MODE = True    # i2i pack dir names must be in I2I_MODE (user-group dirs are not pack dirs; UserEditGroup turns this off)

    def __init__(self, name: str, task_type: str, shard_paths: List[str], *, rank: int, world_size: int,
                 seed: int, caption_seed: int, heldout_keys, max_latent_size: int, shuffle_buffer: int,
                 open_shards: int, i2i_mode_token: bool):
        if task_type not in ("t2i", "i2i"):
            raise ValueError(f"{name}: task_type must be t2i / i2i, got {task_type!r}")
        if not shard_paths:
            raise ValueError(f"{name}: empty shard list")
        if len(set(shard_paths)) != len(shard_paths):
            raise ValueError(f"{name}: duplicate shards in the list")
        if task_type == "i2i" and self.CHECK_I2I_MODE:
            dirs = {os.path.basename(os.path.dirname(p)) for p in shard_paths}
            unknown = sorted(dirs - set(I2I_MODE))
            if unknown:
                raise ValueError(f"{name}: no diff/story mode for pack dirs {unknown}; add them to I2I_MODE")
        self.name = name
        self.task_type = task_type
        self.n_shards = len(shard_paths)
        self.rank = rank
        self.world_size = world_size
        self.seed = int(seed)
        self.caption_seed = int(caption_seed)
        self.heldout_keys = heldout_keys
        self.max_latent_size = int(max_latent_size)
        self.shuffle_buffer = max(1, int(shuffle_buffer))
        self.open_shards = max(1, int(open_shards))
        self.i2i_mode_token = bool(i2i_mode_token)
        self.paths_rank = split_paths_for_rank(shard_paths, rank, world_size, SPLIT_SEED)
        self.skipped = {}

    # ---- skip counting (as the reference _skip: log on the 1st and every 1000th) ----
    def _skip(self, why, worker_id):
        self.skipped[why] = self.skipped.get(why, 0) + 1
        n = self.skipped[why]
        if n == 1 or n % 1000 == 0:
            print(f"[{self.name}] rank {self.rank} worker {worker_id}: skipped {n} samples: {why}", flush=True)

    def _prefilter(self, shard, key, raw, epoch, worker_id):
        """First stage of the reference _emit (per-sample filters + make_sample + DROP_NO_NL). Returns (make_sample dict, path keys) or None."""
        if "latent" not in raw or "meta" not in raw:
            self._skip("missing .pth/.json member", worker_id)
            return None
        image_keys = source_row_keys(raw["meta"].get("source_row") or {})
        if self.heldout_keys and image_keys & self.heldout_keys:
            self._skip("held-out image (heldout_blocklist)", worker_id)
            return None
        refs = [b for _, b in sorted(raw.get("refs", []))]
        if raw["meta"].get("n_ref", len(refs)) != len(refs):
            self._skip("n_ref does not match the ref members", worker_id)
            return None
        smp = zimage_caption.make_sample(shard, key, raw["meta"], raw["latent"], refs, self.caption_seed, epoch)
        if not zimage_caption.keep_no_nl(smp):
            return None
        return smp, sorted(image_keys)

    def _assemble(self, pending):
        """A batch of make_sample dicts -> prompts (caption rules). UserEditGroup uses the json prompt directly instead."""
        return zimage_caption.assemble(pending)

    def _build(self, smp, prompt, image_keys, shard, worker_id):
        """Check part of the reference ZImagePackT2I._build / ZImagePackI2I._build; returns a lightweight buffer record or None."""
        n_ref = len(smp["refs"])
        if self.task_type == "t2i" and n_ref:
            # the reference raises ValueError here, which _emit catches, logs and skips; same result
            self._skip(f"t2i reader got a sample with {n_ref} references", worker_id)
            return None
        if self.task_type == "i2i" and n_ref != 1:
            self._skip(f"{n_ref} references (only single-reference packs are wired)", worker_id)
            return None
        lat, why = target_latent(smp["latent"], self.max_latent_size)
        if lat is None:
            self._skip(why, worker_id)
            return None
        rec = {
            "task_type": self.task_type,
            "prompt": prompt,
            "vae_latent": lat,
            "key": smp["key"],
            "epoch": smp["prompt_epoch"],
            "source": "/".join(shard.split("/")[-3:-1]),
            "image_keys": image_keys,
            "ref_bytes": None,
            "i2i_mode": None,
        }
        if self.task_type == "i2i":
            with Image.open(io.BytesIO(smp["refs"][0])) as ref:    # reads only the header (size), pixels are not decoded
                why = check_i2i_ref(ref.size, self.max_latent_size)
            if why:
                self._skip(why, worker_id)
                return None
            rec["ref_bytes"] = smp["refs"][0]
            rec["i2i_mode"] = I2I_TOKEN_MODE[os.path.basename(smp["prompt_source"])]
        return rec

    def _finalize(self, rec):
        """Decode the reference image only when the record leaves the buffer, and build the dict training expects."""
        prompt = rec["prompt"]
        if rec["task_type"] == "i2i":
            source_image = Image.open(io.BytesIO(rec["ref_bytes"])).convert("RGB")
            text = f"{MODE_TOKEN_TEXT[rec['i2i_mode']]}, {prompt}" if self.i2i_mode_token else prompt
        else:
            source_image = None
            text = prompt
        return {
            "task_type": rec["task_type"],
            "text": text,
            "vae_latent": rec["vae_latent"],
            "source_image": source_image,
            # the fields below are for logging / checks only; training does not read them
            "prompt": prompt,
            "i2i_mode": rec["i2i_mode"],
            "key": rec["key"],
            "epoch": rec["epoch"],
            "source": rec["source"],
            "image_keys": rec["image_keys"],
        }

    def iter_worker(self, worker_id: int, num_workers: Optional[int]):
        """Infinite sample stream of worker `worker_id` on this rank. num_workers=None = not in a DataLoader worker (all of the rank's shards)."""
        paths = split_paths_for_worker(self.paths_rank, worker_id, num_workers)
        if not paths:
            # as the reference _check_worker_shards
            raise ValueError(
                f"{self.name}: rank {self.rank} worker {worker_id} got 0 shards ({self.n_shards} shards split over "
                f"{self.world_size} ranks x {num_workers} workers). Lower num_workers.")
        tag = f"{self.name}-s{self.seed}"
        n_slots = min(self.open_shards, len(paths))
        rng_slot = random.Random(f"{tag}-{self.rank}-{worker_id}-slots")
        rng_buf = random.Random(f"{tag}-{self.rank}-{worker_id}-buffer")
        print(f"rank-{self.rank} worker-{worker_id} {self.name}: {len(paths)} shards, {n_slots} open at a time, "
              f"shuffle buffer {self.shuffle_buffer}, seed {self.seed}", flush=True)

        def shard_stream():
            epoch = 0
            while True:
                for p in epoch_order(tag, self.rank, worker_id, epoch, paths):
                    yield epoch, p
                epoch += 1
                print(f"{self.name} repeat in rank-{self.rank} worker-{worker_id} (epoch {epoch})", flush=True)

        stream = shard_stream()
        slots: List[Any] = [None] * n_slots     # each slot: (epoch, shard path, iter_pack_shard generator)
        pending, pending_keys, buffer = [], [], []
        n_fail = 0
        while True:
            j = rng_slot.randrange(n_slots)
            if slots[j] is None:
                epoch, path = next(stream)
                slots[j] = (epoch, path, iter_pack_shard(path))
            epoch, path, it = slots[j]
            try:
                key, raw = next(it)
            except StopIteration:
                slots[j] = None
                continue
            except Exception:
                # as the reference: log a corrupt shard once and skip the rest of it
                print(f"[{self.name}] error reading {path}, skipping the rest of it", flush=True)
                traceback.print_exc()
                slots[j] = None
                n_fail += 1
                if n_fail > 10 * len(paths):
                    raise RuntimeError(f"{self.name}: {n_fail} shard reads in a row failed on rank {self.rank} "
                                       f"worker {worker_id}; giving up")
                continue
            n_fail = 0
            # The reference catches errors of these steps in the outer try of __iter__ (dropping the rest of the
            # shard); here they are caught per sample / per batch so the worker survives
            try:
                got = self._prefilter(path, key, raw, epoch, worker_id)
            except Exception:
                print(f"[{self.name}] failed to read sample {key} of {path}", flush=True)
                traceback.print_exc()
                continue
            if got is None:
                continue
            pending.append(got[0])
            pending_keys.append((got[1], path))
            if len(pending) < CAPTION_BATCH:
                continue
            try:
                prompts = self._assemble(pending)
            except Exception:
                print(f"[{self.name}] caption assemble failed for a batch of {len(pending)}, dropping it", flush=True)
                traceback.print_exc()
                pending, pending_keys = [], []
                continue
            for smp, prompt, (image_keys, shard) in zip(pending, prompts, pending_keys):
                try:
                    rec = self._build(smp, prompt, image_keys, shard, worker_id)
                except Exception:
                    print(f"[{self.name}] failed to build {smp['key']}", flush=True)
                    traceback.print_exc()
                    continue
                if rec is None:
                    continue
                buffer.append(rec)
                if len(buffer) >= self.shuffle_buffer:
                    rec = buffer.pop(rng_buf.randrange(len(buffer)))
                    try:
                        out = self._finalize(rec)
                    except Exception:
                        print(f"[{self.name}] failed to decode the reference of {rec['key']}", flush=True)
                        traceback.print_exc()
                        continue
                    yield out
            pending, pending_keys = [], []


class UserEditGroup(PackGroup):
    """A user group (`reader: user_edit` in the yaml). Each tar entry is {rid}.json (+ {rid}_ref_0.webp for
    i2i), with no .pth. Shard splitting, slots and the shuffle buffer are inherited from PackGroup. Differences:
    the prompt is taken directly from the json (no caption rules); the target latent is all zeros,
    [16, target_h/8, target_w/8], and only fixes the size (ODE rollout / TDM start from pure noise and never
    look at the target image); held-out rows are blocked by rid / ref_key."""

    CHECK_I2I_MODE = False

    def __init__(self, name, task_type, shard_paths, *, user_blocked=frozenset(), **kw):
        kw["heldout_keys"] = frozenset()
        super().__init__(name, task_type, shard_paths, **kw)
        self.user_blocked = user_blocked
        self._check_dirs(name, task_type, shard_paths)

    @staticmethod
    def _check_dirs(name, task_type, shard_paths):
        """Each dir's tar count must equal n_shards in its dataset_info.json, and its task_type must match."""
        import json
        by_dir = {}
        for p in shard_paths:
            by_dir.setdefault(os.path.dirname(p), []).append(p)
        for d, ps in by_dir.items():
            info_path = os.path.join(d, "dataset_info.json")
            if not os.path.isfile(info_path):
                raise FileNotFoundError(f"{name}: {info_path} missing (user_edit dirs carry one)")
            with open(info_path) as f:
                info = json.load(f)
            if int(info.get("n_shards", -1)) != len(ps):
                raise ValueError(f"{name}: {d} has {len(ps)} tars but dataset_info says n_shards={info.get('n_shards')}")
            if info.get("task_type") != task_type:
                raise ValueError(f"{name}: {d} is task_type={info.get('task_type')!r}, listed under {task_type}")

    def _assemble(self, pending):
        return [smp["meta"]["prompt"] for smp in pending]

    def _prefilter(self, shard, key, raw, epoch, worker_id):
        meta = raw.get("meta")
        if meta is None:
            self._skip("missing .json member", worker_id)
            return None
        if "latent" in raw:
            self._skip("user sample with a .pth member (not a user_edit dir?)", worker_id)
            return None
        if meta.get("task_type") != self.task_type:
            self._skip(f"task_type {meta.get('task_type')!r} in a {self.task_type} group", worker_id)
            return None
        ref_key = meta.get("ref_key")
        if self.user_blocked and (meta.get("rid") in self.user_blocked or (ref_key and ref_key in self.user_blocked)):
            self._skip("held-out user row (user_heldout_blocklist)", worker_id)
            return None
        refs = [b for _, b in sorted(raw.get("refs", []))]
        if meta.get("n_ref", len(refs)) != len(refs):
            self._skip("n_ref does not match the ref members", worker_id)
            return None
        smp = {"key": f"{os.path.basename(os.path.dirname(shard))}/{os.path.basename(shard)}/{key}",
               "meta": meta, "refs": refs, "prompt_epoch": epoch}
        return smp, sorted(k for k in [ref_key] if k)

    def _build(self, smp, prompt, image_keys, shard, worker_id):
        import torch
        meta = smp["meta"]
        n_ref = len(smp["refs"])
        if self.task_type == "t2i" and n_ref:
            self._skip(f"t2i reader got a sample with {n_ref} references", worker_id)
            return None
        if self.task_type == "i2i" and n_ref != 1:
            self._skip(f"{n_ref} references (only single-reference rows are wired)", worker_id)
            return None
        tw, th = int(meta["target_w"]), int(meta["target_h"])
        if tw % 16 or th % 16:
            self._skip("user target side not a multiple of 16", worker_id)
            return None
        h, w = th // 8, tw // 8
        if not fits(h, w, self.max_latent_size):
            self._skip("user target larger than max_latent_size", worker_id)
            return None
        mode, expected = None, prompt
        if self.task_type == "i2i":
            mode = meta.get("mode")
            if mode not in MODE_TOKEN_TEXT:
                self._skip(f"user i2i row with mode {mode!r}", worker_id)
                return None
            expected = f"{MODE_TOKEN_TEXT[mode]}, {prompt}"
        if meta.get("text") != expected:
            self._skip("json text != mode token + prompt", worker_id)
            return None
        rec = {
            "task_type": self.task_type,
            "prompt": prompt,
            "vae_latent": torch.zeros(16, h, w, dtype=torch.bfloat16),
            "key": smp["key"],
            "epoch": smp["prompt_epoch"],
            "source": "/".join(shard.split("/")[-3:-1]),
            "image_keys": image_keys,
            "ref_bytes": None,
            "i2i_mode": mode,
        }
        if self.task_type == "i2i":
            with Image.open(io.BytesIO(smp["refs"][0])) as ref:
                why = check_i2i_ref(ref.size, self.max_latent_size)
            if why:
                self._skip(why, worker_id)
                return None
            rec["ref_bytes"] = smp["refs"][0]
        return rec


class GroupMixDataset(IterableDataset):
    """The groups of one task (t2i or i2i), drawn per sample by weight (as the group-weight sampling of the reference PackedDataset)."""

    def __init__(self, groups: List[PackGroup], weights: List[float], tag: str):
        if not groups or len(groups) != len(weights):
            raise ValueError("GroupMixDataset: groups / weights mismatch")
        if any(w <= 0 for w in weights):
            raise ValueError(f"GroupMixDataset: weights must be > 0, got {weights}")
        self.groups = groups
        self.weights = [float(w) for w in weights]
        self.tag = tag

    def __iter__(self):
        info = get_worker_info()
        worker_id, num_workers = (0, None) if info is None else (info.id, info.num_workers)
        its = [g.iter_worker(worker_id, num_workers) for g in self.groups]
        if len(its) == 1:
            yield from its[0]
            return
        rng = random.Random(f"{self.tag}-s{self.groups[0].seed}-{self.groups[0].rank}-{worker_id}-mix")
        idx = list(range(len(its)))
        while True:
            yield next(its[rng.choices(idx, weights=self.weights)[0]])


def _preload_nl_sidetable():
    """Build the in-process nl_sidetable singleton in the main process (before workers fork) and trim its
    parsed meta to the keys that are still read later.

    Measured on CPU: the NL side table's meta.json is 1.4 GB; json.load turns it into ~29M Python objects,
    ~5.1 GiB of private memory and ~34 s. The rule code loads it once per process on the first
    attach_nl_segments, so 2 loaders x 2 workers per GPU would cost ~20 GiB per GPU, ~160 GiB on 8 GPUs.
    The table itself is an np.memmap (shared through the page cache), and after the consistency checks at
    build time only prefetch() reads meta["files"], so the other keys are dropped here (no change to the rule
    code or to any lookup result; prompts stay byte-identical). Forked workers inherit the trimmed singleton
    instead of each loading their own copy.
    """
    import gc

    from Kokoro2.data import nl_sidetable

    table = nl_sidetable.get_table()
    if table:
        keep = ("files", "n_entries", "max_segs", "class_fields", "class_code", "format")
        table.meta = {k: table.meta[k] for k in keep if k in table.meta}
    gc.collect()


def _list_shards(pack_root: str, dirs: List[str]) -> List[str]:
    """As the reference dataset_info._zimage_pack_shards: sorted(listdir *.tar) per dir. A missing dir or a dir without tars is an error."""
    shards = []
    for d in dirs:
        full = os.path.join(pack_root, d)
        if not os.path.isdir(full):
            raise FileNotFoundError(f"pack dir {full} does not exist")
        tars = sorted(os.path.join(full, f) for f in os.listdir(full) if f.endswith(".tar"))
        if not tars:
            raise FileNotFoundError(f"pack dir {full} has no .tar shards")
        shards += tars
    return shards


def _load_yaml(data_yaml: str) -> dict:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(data_yaml), resolve=True)


def build_kokoro_mainline_pack_dataloader_for_trainer(
    data_yaml: str,
    i2i_prob: Optional[float] = None,
    num_workers: Optional[int] = None,
    shuffle_buffer: Optional[int] = None,
    open_shards: Optional[int] = None,
    max_latent_size: Optional[int] = None,
    i2i_mode_token: Optional[bool] = None,
    seed: Optional[int] = None,
    batch_size: int = 1,
    pin_memory: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    **_,
) -> MixedTaskLoader:
    """Read kokoro_ode/data_mainline_1024.yaml and return a MixedTaskLoader (.batch_size = 1; each next() gives [sample dict]).

    Arguments left as None take the yaml value; max_latent_size is the smaller of the yaml value and the
    argument (usually the network's MAX_LATENT_SIZE: data must not exceed the network's position table).
    With i2i_prob = 0 / 1 only one loader is built (the other starts no workers). num_workers is per loader
    (t2i and i2i each start that many). rank / world_size default to torch.distributed (or the env vars
    RANK / WORLD_SIZE if it is not initialized, else 0 / 1).
    """
    if batch_size != 1:
        raise ValueError(f"Kokoro inference dataloader requires batch_size=1, got {batch_size}")
    cfg = _load_yaml(data_yaml)
    caption_rules = cfg.get("caption_rules", "zimage")
    if caption_rules not in ("zimage", "zimage+user"):
        raise ValueError(f"{data_yaml}: caption_rules={caption_rules!r}; only 'zimage' / 'zimage+user' are implemented")
    pack_root = cfg["pack_root"]
    i2i_prob = float(cfg.get("i2i_prob", 0.565) if i2i_prob is None else i2i_prob)
    num_workers = int(cfg.get("num_workers", 2) if num_workers is None else num_workers)
    shuffle_buffer = int(cfg.get("shuffle_buffer", 16) if shuffle_buffer is None else shuffle_buffer)
    open_shards = int(cfg.get("open_shards", 8) if open_shards is None else open_shards)
    yaml_mls = int(cfg.get("max_latent_size", 128))
    max_latent_size = yaml_mls if max_latent_size is None else min(yaml_mls, int(max_latent_size))
    i2i_mode_token = bool(cfg.get("i2i_mode_token", True) if i2i_mode_token is None else i2i_mode_token)
    seed = int(cfg.get("seed", SPLIT_SEED) if seed is None else seed)
    caption_seed = int(cfg.get("caption_seed", 0))
    if not 0.0 <= i2i_prob <= 1.0:
        raise ValueError(f"i2i_prob must be in [0, 1], got {i2i_prob}")
    if rank is None or world_size is None:
        r, w = _dist_rank_world()
        rank = r if rank is None else rank
        world_size = w if world_size is None else world_size

    blocklist = cfg.get("heldout_blocklist") or ""
    if blocklist and not os.path.isfile(blocklist):
        raise FileNotFoundError(f"heldout_blocklist {blocklist} does not exist")
    heldout_keys = load_heldout_keys(blocklist) if blocklist else frozenset()
    user_blocklist = cfg.get("user_heldout_blocklist") or ""
    user_blocked = frozenset(line.strip() for line in open(user_blocklist)
                             if line.strip() and not line.startswith("#")) if user_blocklist else frozenset()

    # The reference sets the env vars and imports the rule code in the reader's __init__ (main process, before
    # workers fork); do the same once here in the main process
    zimage_caption.configure()
    _preload_nl_sidetable()

    def build_task(task_type: str):
        entries = cfg.get(task_type) or []
        if not entries:
            return None
        groups, weights = [], []
        for e in entries:
            shards = _list_shards(pack_root, e["dirs"])
            need = world_size * max(1, num_workers)
            if len(shards) < need:
                raise ValueError(f"{e['name']}: {len(shards)} shards < {world_size} ranks x {num_workers} workers")
            reader = e.get("reader", "pack")
            if reader not in ("pack", "user_edit"):
                raise ValueError(f"{e['name']}: reader={reader!r} is not implemented")
            if reader == "user_edit" and caption_rules != "zimage+user":
                # user groups have no .pth: read as packs, every sample would be skipped, the group would never
                # yield and multi-GPU runs would eventually time out
                raise ValueError(f"{e['name']}: reader=user_edit needs caption_rules: zimage+user")
            common = dict(rank=rank, world_size=world_size, seed=seed, caption_seed=caption_seed,
                          max_latent_size=max_latent_size, shuffle_buffer=shuffle_buffer, open_shards=open_shards,
                          i2i_mode_token=i2i_mode_token)
            if reader == "user_edit":
                # user and pack groups share one GroupMixDataset and are drawn per sample by weight
                # (as in CFG distillation: P(user | task) = user weight / total task weight)
                groups.append(UserEditGroup(e["name"], task_type, shards, user_blocked=user_blocked, **common))
            else:
                groups.append(PackGroup(e["name"], task_type, shards, heldout_keys=heldout_keys, **common))
            weights.append(float(e.get("weight", 1.0)))
            logger.info(f"[mainline pack] {task_type} group {e['name']}: {len(e['dirs'])} dirs, {len(shards)} shards "
                        f"({len(groups[-1].paths_rank)} on rank {rank}), weight {weights[-1]}")
        return make_dataloader(GroupMixDataset(groups, weights, tag=task_type), num_workers=num_workers,
                               pin_memory=pin_memory)

    t2i_loader = build_task("t2i") if i2i_prob < 1.0 else None
    i2i_loader = build_task("i2i") if i2i_prob > 0.0 else None
    logger.info(
        f"[mainline pack] {data_yaml}: rank {rank}/{world_size}, i2i_prob={i2i_prob}, i2i_mode_token={i2i_mode_token}, "
        f"num_workers={num_workers} per loader, shuffle_buffer={shuffle_buffer}, open_shards={open_shards}, "
        f"max_latent_size={max_latent_size}, seed={seed}, caption_seed={caption_seed}, "
        f"held-out keys={len(heldout_keys)}")
    # The task sequence depends only on the data seed (not the global torch RNG): runs with the same seed read
    # the same data at every iteration
    return MixedTaskLoader(t2i_loader, i2i_loader, i2i_prob, task_seed=seed + 7919)
