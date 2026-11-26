#!/usr/bin/env python
"""Export a trained Kokoro-DMD2 student checkpoint to a single .safetensors file.

Manual / offline counterpart to the save-time export hook in FSDPCheckpointer.
Both share the consolidation logic in ``fastgen.utils.export_safetensors``.

Training (FSDPCheckpointer) saves the student as a sharded DCP directory
``<prefix>.net_model/``.  Its state dict is ``bagel.*`` (the BAGEL weights) +
``vae.*`` (frozen, unused by the backend).  The backend loads a flat
``ema.safetensors`` = the ``bagel.*`` keys with the ``bagel.`` prefix stripped.
This script produces that flat file (a drop-in replacement for the teacher's
``ema.safetensors``).  It aggregates the shards CPU-only (``no_dist=True``) and
does NOT instantiate the 7B model.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/export_student_safetensors.py \
        --ckpt <run>/checkpoints/0000500 \
        --out  <run>/exported/student_0000500.safetensors

    # --ckpt may point at the prefix, the .net_model dir, or the .pth file.
    # --use-ema exports the .ema_net_model dir instead (only if EMA was enabled).
"""

import argparse
import os
import sys

from fastgen.utils.export_safetensors import consolidate_dcp_dir, normalize_prefix


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Checkpoint prefix / .net_model dir / .pth file")
    ap.add_argument("--out", required=True, help="Output .safetensors path")
    ap.add_argument("--use-ema", action="store_true", help="Export .ema_net_model instead of .net_model")
    ap.add_argument("--strip-prefix", default="bagel.", help="State-dict prefix to keep+strip (default: bagel.)")
    ap.add_argument("--dtype", default="keep", choices=["keep", "bf16", "fp16", "fp32"],
                    help="Cast exported tensors (default: keep training dtype)")
    args = ap.parse_args()

    prefix = normalize_prefix(args.ckpt)
    model_key = "ema_net" if args.use_ema else "net"
    dcp_dir = f"{prefix}.{model_key}_model"
    if not os.path.isdir(dcp_dir):
        parent = os.path.dirname(prefix) or "."
        avail = sorted(d for d in os.listdir(parent) if d.endswith("_model"))
        sys.exit(f"ERROR: DCP dir not found: {dcp_dir}\n  Available in {parent}: {avail}")

    print(f"Reading DCP checkpoint: {dcp_dir}")
    print("  aggregating shards (this loads the full tensors into CPU RAM)...")
    cast = None if args.dtype == "keep" else args.dtype
    sd = consolidate_dcp_dir(dcp_dir, strip_prefix=args.strip_prefix, cast=cast)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    from safetensors.torch import save_file
    save_file(sd, args.out, metadata={"format": "pt", "source": os.path.basename(dcp_dir), "exported_key": model_key})

    total_bytes = sum(v.numel() * v.element_size() for v in sd.values())
    print(f"\nWrote {len(sd)} tensors ({total_bytes / 1e9:.2f} GB) -> {args.out}")
    print("Sample exported keys:")
    for k in list(sd)[:5]:
        print(f"  {k}\t{tuple(sd[k].shape)}\t{sd[k].dtype}")


if __name__ == "__main__":
    main()
