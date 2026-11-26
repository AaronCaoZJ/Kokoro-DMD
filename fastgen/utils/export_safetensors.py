# SPDX-License-Identifier: Apache-2.0
"""Consolidate a sharded FSDP DCP checkpoint into a single backend-loadable .safetensors.

FSDPCheckpointer saves each model component as a sharded DCP directory
(`<prefix>.net_model/`, `<prefix>.ema_net_model/`, ...).  The KokoroNetwork state
dict inside is laid out as `bagel.<key>` (the actual BAGEL weights) + `vae.<key>`
(frozen VAE, not needed by the backend).  The inference backend loads a *flat*
`ema.safetensors` whose keys are exactly the `bagel.*` keys with the `bagel.`
prefix stripped.

This module aggregates the DCP shards in a single process (``no_dist=True`` — this
is mandatory when called from inside a running distributed job, otherwise the
rank-0-only call would hang waiting for a collective), keeps the `bagel.*`
tensors, strips the prefix, and produces that flat dict / file.  It does NOT
instantiate the 7B model — the empty template is built from the DCP `.metadata`.

Used by both ``scripts/export_student_safetensors.py`` (manual, offline) and the
optional save-time export hook in ``FSDPCheckpointer.save``.
"""

import os
from typing import Dict, Mapping, Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

# model_dict key -> output filename in the backend layout.
DEFAULT_KEY_FILENAMES: Dict[str, str] = {
    "net": "student.safetensors",
    "ema": "ema.safetensors",
    "ema_net": "ema.safetensors",
}

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def normalize_prefix(ckpt: str) -> str:
    """Accept a prefix, a ``*_model`` dir, or a ``*.pth`` file and return the prefix."""
    ckpt = ckpt.rstrip("/")
    for suffix in (".net_model", ".ema_net_model", ".ema_model", ".pth"):
        if ckpt.endswith(suffix):
            return ckpt[: -len(suffix)]
    return ckpt


def consolidate_dcp_dir(dcp_dir: str, strip_prefix: str = "bagel.",
                        cast: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Aggregate a single DCP model dir into a flat state dict (prefix stripped).

    Args:
        dcp_dir: path to the ``*_model`` DCP directory.
        strip_prefix: only keys under this prefix are kept; the prefix is removed.
        cast: optional "bf16"/"fp16"/"fp32" to cast exported tensors (default: keep).

    Returns:
        ``{stripped_key: cpu_tensor}`` ready for ``safetensors.torch.save_file``.
    """
    reader = FileSystemReader(dcp_dir)
    md = reader.read_metadata()

    template: Dict[str, torch.Tensor] = {}
    for key, meta in md.state_dict_metadata.items():
        if not key.startswith(strip_prefix):
            continue
        size = getattr(meta, "size", None)
        props = getattr(meta, "properties", None)
        if size is None or props is None:  # non-tensor (bytes/scalar) entry
            continue
        template[key] = torch.empty(tuple(size), dtype=props.dtype)
    if not template:
        raise ValueError(f"No keys with prefix '{strip_prefix}' found in {dcp_dir}")

    # no_dist=True: read all shards in this single process. REQUIRED when called
    # from a rank-0-only path inside a live distributed job (else it deadlocks).
    try:
        dcp.load(template, storage_reader=reader, no_dist=True)
    except TypeError:
        # torch < 2.6 (e.g. 2.5.1): dcp.load has no no_dist; the private loader does.
        from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

        _load_state_dict(template, storage_reader=reader, no_dist=True)

    cast_dtype = _DTYPES.get(cast) if cast else None
    n = len(strip_prefix)
    out: Dict[str, torch.Tensor] = {}
    for k, v in template.items():
        v = v.contiguous()
        if cast_dtype is not None:
            v = v.to(cast_dtype)
        out[k[n:]] = v
    return out


def export_prefix_to_dir(
    prefix: str,
    out_dir: str,
    model_keys: Optional[Mapping[str, str]] = None,
    strip_prefix: str = "bagel.",
    iteration: Optional[int] = None,
    cast: Optional[str] = None,
) -> Dict[str, str]:
    """Write the backend-layout files for every available model key under `prefix`.

    For each ``model_key -> filename`` in `model_keys`, if ``{prefix}.{model_key}_model``
    exists it is consolidated and written to ``{out_dir}/{filename}``.  Missing dirs
    are skipped (e.g. EMA when EMA is disabled).  Also writes ``step.txt`` when
    `iteration` is given.

    Returns ``{filename: written_path}`` for the files actually written.
    """
    from safetensors.torch import save_file

    prefix = normalize_prefix(prefix)
    if model_keys is None:
        model_keys = DEFAULT_KEY_FILENAMES
    os.makedirs(out_dir, exist_ok=True)

    written: Dict[str, str] = {}
    for model_key, filename in model_keys.items():
        dcp_dir = f"{prefix}.{model_key}_model"
        if not os.path.isdir(dcp_dir):
            continue
        sd = consolidate_dcp_dir(dcp_dir, strip_prefix=strip_prefix, cast=cast)
        out_path = os.path.join(out_dir, filename)
        meta = {"format": "pt", "source": os.path.basename(dcp_dir)}
        if iteration is not None:
            meta["iteration"] = str(iteration)
        save_file(sd, out_path, metadata=meta)
        written[filename] = out_path
        del sd  # free rank-0 RAM before the next (potentially 28GB) key

    if iteration is not None:
        with open(os.path.join(out_dir, "step.txt"), "w") as f:
            f.write(str(iteration))

    return written
