"""Inference-style (raw-sample) Kokoro T2I+I2I dataloader for DMD2.

Yields raw samples with ``batch_size=1``; the KV-cache / inference context is
built per-step in the trainer process by ``KokoroNetwork`` — never in a
dataloader worker (NaiveCache cannot cross worker boundaries safely).

Supports mixed T2I+I2I sampling via ``MixedTaskLoader``, which broadcasts the
task choice from rank 0 each step so all FSDP ranks operate on the same task
type (avoids non-uniform collectives).
"""

from __future__ import annotations

import glob as _glob
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import webdataset as wds

from fastgen.datasets.kokoro_cfgdistill_data import (
    make_dataloader,
    make_i2i_dataset,
    make_t2i_dataset,
)
from fastgen.utils.logging_utils import logger


def _normalize_shard_specs(shard_specs) -> List[dict]:
    """Normalize shard config to a list of dicts with at least ``path``.

    Accepts:
      - None / empty → []
      - str (single shard glob) → [{"path": str, "weight": 1.0}]
      - list of strs → [{"path": s, "weight": 1.0}, ...]
      - list of dicts (already normalized) → returned with weight defaulted to 1

    The experiment config round-trips through OmegaConf/Hydra (the
    ``override_config_with_opts`` path), so ``shard_specs`` arrives here as an
    ``omegaconf.ListConfig`` of ``DictConfig`` rather than plain list/dict.
    Convert back to native containers first so the isinstance checks below
    (and downstream ``dict``/``str`` handling) work.
    """
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(shard_specs, (ListConfig, DictConfig)):
            shard_specs = OmegaConf.to_container(shard_specs, resolve=True)
    except ImportError:
        pass

    if not shard_specs:
        return []
    if isinstance(shard_specs, str):
        return [{"path": shard_specs, "weight": 1.0}]
    if not isinstance(shard_specs, (list, tuple)):
        raise TypeError(f"shards must be str / list, got {type(shard_specs)}")

    normalized = []
    for item in shard_specs:
        if isinstance(item, str):
            normalized.append({"path": item, "weight": 1.0})
        elif isinstance(item, dict):
            path = item.get("path") or item.get("shards")
            if not path:
                raise ValueError(f"Shard entry missing path/shards: {item}")
            spec = dict(item)
            spec["path"] = path
            spec["weight"] = float(item.get("weight", 1.0))
            spec.pop("shards", None)
            normalized.append(spec)
        else:
            raise TypeError(f"Unsupported shard entry: {item!r}")
    return normalized


def _resolve_shards(shard_patterns):
    """Expand shell-glob shard patterns to an explicit, sorted file list.

    The byte-ported ``expand_shards``/``wds.shardlists.expand_urls`` only does
    brace/range expansion, NOT shell ``*`` globbing.  FastGen configs build
    shard specs as shell globs (``$DATA_ROOT/*/*.tar``); without this expansion
    WebDataset receives the literal ``*`` path, every ``open()`` fails, and
    ``handler=ignore_and_continue`` + ``resampled=True`` retry forever -> a
    SILENT infinite hang (no error, no batch).  Fail loudly on zero matches.
    """
    if isinstance(shard_patterns, str):
        shard_patterns = [shard_patterns]
    resolved: list[str] = []
    for item in shard_patterns:
        if any(c in item for c in "*?["):
            resolved.extend(sorted(_glob.glob(item)))  # glob: matches only (maybe none)
        else:
            resolved.append(item)  # explicit path / brace pattern: pass through
    if not resolved:
        raise FileNotFoundError(
            f"No WebDataset shards matched {shard_patterns!r}. Refusing to start: "
            "an empty/literal-glob shard list causes a silent infinite hang "
            "(resampled=True + ignore_and_continue). Check DATA_ROOT_DIR / WDS_SHARD_DIRS."
        )
    return resolved


def _build_t2i_loader(specs, num_workers, pin_memory):
    if not specs:
        return None
    datasets = []
    weights = []
    for spec in specs:
        datasets.append(
            make_t2i_dataset(
                _resolve_shards(spec["path"]),
                prompt_builder=spec.get("prompt_builder", "auto"),
            )
        )
        weights.append(spec["weight"])
    ds = datasets[0] if len(datasets) == 1 else wds.RandomMix(datasets, probs=weights)
    return make_dataloader(ds, num_workers=num_workers, pin_memory=pin_memory)


def _build_i2i_loader(specs, num_workers, pin_memory):
    if not specs:
        return None
    datasets = []
    weights = []
    for spec in specs:
        datasets.append(
            make_i2i_dataset(
                _resolve_shards(spec["path"]),
                decoder=spec.get("decoder", "auto"),
                prompt_builder=spec.get("prompt_builder"),
            )
        )
        weights.append(spec["weight"])
    ds = datasets[0] if len(datasets) == 1 else wds.RandomMix(datasets, probs=weights)
    return make_dataloader(ds, num_workers=num_workers, pin_memory=pin_memory)


class MixedTaskLoader:
    """Yield t2i / i2i samples per ``i2i_prob`` with cross-rank broadcast sync.

    All ranks always agree on the task type at each step (rank 0 picks, then
    broadcast).  This avoids FSDP collectives being called inconsistently when
    student forward shape varies between t2i (text-only KV) and i2i (text+image
    KV).
    """

    def __init__(
        self,
        t2i_loader: Optional[torch.utils.data.DataLoader],
        i2i_loader: Optional[torch.utils.data.DataLoader],
        i2i_prob: float,
        task_seed: Optional[int] = None,
    ):
        if t2i_loader is None and i2i_loader is None:
            raise ValueError("MixedTaskLoader requires at least one of t2i / i2i loaders")
        self.t2i_loader = t2i_loader
        self.i2i_loader = i2i_loader
        self.i2i_prob = float(i2i_prob)
        # FastGen scripts.py reads this field to compute grad_accum_rounds.
        self.batch_size = 1
        self._t2i_iter = iter(t2i_loader) if t2i_loader is not None else None
        self._i2i_iter = iter(i2i_loader) if i2i_loader is not None else None
        # task_seed given: draw t2i/i2i from a private generator so the task sequence (and with it the
        # whole data stream) depends only on the seed. The global torch RNG is also consumed by the
        # method (noise, timesteps, exit steps), differently per method, so with it two methods on the
        # same seed would see different data at the same iteration. None = draw from the global RNG.
        self._task_gen = torch.Generator().manual_seed(int(task_seed)) if task_seed is not None else None

    def __iter__(self):
        return self

    def _draw(self) -> int:
        return 1 if torch.rand((), generator=self._task_gen).item() < self.i2i_prob else 0

    def _pick_task_id(self) -> int:
        if self.t2i_loader is None:
            return 1
        if self.i2i_loader is None:
            return 0
        if dist.is_initialized():
            device = (
                torch.device(f"cuda:{torch.cuda.current_device()}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            task = torch.zeros((), dtype=torch.long, device=device)
            if dist.get_rank() == 0:
                task.fill_(self._draw())
            dist.broadcast(task, src=0)
            return int(task.item())
        return self._draw()

    def _next_from(self, which: str):
        if which == "i2i":
            try:
                return next(self._i2i_iter)
            except StopIteration:
                self._i2i_iter = iter(self.i2i_loader)
                return next(self._i2i_iter)
        try:
            return next(self._t2i_iter)
        except StopIteration:
            self._t2i_iter = iter(self.t2i_loader)
            return next(self._t2i_iter)

    def __next__(self):
        task_id = self._pick_task_id()
        return self._next_from("i2i" if task_id == 1 else "t2i")


def build_kokoro_inference_dataloader_for_trainer(
    t2i_shards: Any = None,
    i2i_shards: Any = None,
    i2i_prob: float = 0.6,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    # Legacy single-list T2I support (back-compat with older configs).
    shard_patterns: Any = None,
    prompt_builder: str = "auto",
    **_,
) -> torch.utils.data.DataLoader:
    """Build a mixed T2I + I2I inference-style DataLoader.

    Args:
        t2i_shards: T2I shard spec (str / list-of-str / list-of-dict).  Each
            dict may carry ``path``, ``weight``, ``prompt_builder``.
        i2i_shards: I2I shard spec (list-of-dict typically), each with
            ``path``, ``weight``, ``decoder``, ``prompt_builder``.
        i2i_prob: probability of sampling an i2i sample per step (only used
            when both loaders are present).
        batch_size: must be 1 (inference branch single-image invariant).
        shard_patterns: Legacy back-compat — if ``t2i_shards`` is empty and
            ``shard_patterns`` is set, treat ``shard_patterns`` as T2I shards
            with the given ``prompt_builder``.
    """
    if batch_size != 1:
        raise ValueError(
            f"Kokoro inference dataloader requires batch_size=1, got {batch_size}"
        )

    if not t2i_shards and shard_patterns:
        t2i_shards = [{"path": shard_patterns, "prompt_builder": prompt_builder}]

    t2i_specs = _normalize_shard_specs(t2i_shards)
    i2i_specs = _normalize_shard_specs(i2i_shards)

    for spec in t2i_specs:
        spec.setdefault("prompt_builder", prompt_builder)

    if not t2i_specs and not i2i_specs:
        raise ValueError("Provide at least one of t2i_shards / i2i_shards")

    t2i_loader = _build_t2i_loader(t2i_specs, num_workers, pin_memory)
    i2i_loader = _build_i2i_loader(i2i_specs, num_workers, pin_memory)

    logger.info(
        f"Kokoro inference dataloader: t2i sources={len(t2i_specs)}, "
        f"i2i sources={len(i2i_specs)}, i2i_prob={i2i_prob}, "
        f"num_workers={num_workers}"
    )

    return MixedTaskLoader(t2i_loader, i2i_loader, i2i_prob)
