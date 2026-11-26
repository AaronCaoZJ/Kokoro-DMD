#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Inference script for the original Kokoro/Bagel teacher checkpoint
# stored as a single safetensors file.

import argparse
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default=os.environ.get("TEACHER_CKPT"),
        help="Teacher checkpoint (.safetensors/.pt/.pth)",
    )
    p.add_argument(
        "--model_dir",
        default=os.environ.get("BAGEL_MODEL_DIR"),
        help="BAGEL-7B-MoT model directory",
    )
    p.add_argument("--out_dir", required=True, help="Directory to save generated images")
    p.add_argument("--prompts", nargs="+", help="One or more prompts (quoted strings)")
    p.add_argument("--prompt_file", help="Text file with one prompt per line")
    p.add_argument("--num_steps", type=int, default=4, help="Teacher denoising steps")
    p.add_argument("--cfg_scale", type=float, default=4.0, help="CFG guidance scale")
    p.add_argument("--resolution", type=int, default=1024, help="Output resolution in pixels")
    p.add_argument("--max_latent_size", type=int, default=96)
    p.add_argument(
        "--timestep_shift",
        type=float,
        default=1.0,
        help="Teacher timestep shift. The original Kokoro teacher uses 3.0.",
    )
    p.add_argument("--cfg_renorm_min", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_teacher_weights(bagel: torch.nn.Module, ckpt_path: str) -> None:
    from fastgen.networks.kokoro.kokoro_network import KokoroNetwork

    KokoroNetwork.load_bagel_weights(bagel, ckpt_path)


def main():
    from scripts.inference.kokoro_inference import (
        build_model,
        load_vae,
        get_tokenizer_and_token_ids,
        generate_x0_resample,
        prompt_to_filename,
    )

    args = parse_args()

    if args.prompt_file:
        prompts = [l.strip() for l in open(args.prompt_file) if l.strip()]
    elif args.prompts:
        prompts = args.prompts
    else:
        raise ValueError("Provide --prompts or --prompt_file")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Building Bagel from {args.model_dir} ...")
    bagel = build_model(args.model_dir, args.max_latent_size)
    print(f"Loading teacher weights from {args.ckpt} ...")
    load_teacher_weights(bagel, args.ckpt)
    bagel = bagel.to(device).eval()

    print("Loading VAE ...")
    vae = load_vae(args.model_dir).to(device).eval()

    tokenizer, new_token_ids = get_tokenizer_and_token_ids(args.model_dir)

    used_names = set()
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt}")
        img = generate_x0_resample(
            prompt=prompt,
            bagel=bagel,
            vae=vae,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            device=device,
            resolution=args.resolution,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            cfg_renorm_min=args.cfg_renorm_min,
            timestep_shift=args.timestep_shift,
        )
        base_name = prompt_to_filename(prompt, i)
        file_name = f"{base_name}.png"
        suffix = 1
        while file_name in used_names or os.path.exists(os.path.join(args.out_dir, file_name)):
            file_name = f"{base_name}_{suffix}.png"
            suffix += 1
        used_names.add(file_name)
        out_path = os.path.join(args.out_dir, file_name)
        img.save(out_path)
        print(f"  Saved -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
