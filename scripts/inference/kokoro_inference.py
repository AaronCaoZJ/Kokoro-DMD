#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Inference script for FastGen-trained Kokoro/Bagel DMD2 student checkpoints.
#
# Usage (single GPU):
#   python scripts/inference/kokoro_inference.py \
#       --ckpt      outputs/kokoro-dmd2/.../checkpoints/0001000.pth \
#       --model_dir ${BAGEL_MODEL_DIR} \
#       --prompts   "a red panda sitting on a bamboo" \
#       --out_dir   /tmp/kokoro_out
#
# Multi-prompt via file (one prompt per line):
#   python scripts/inference/kokoro_inference.py \
#       --ckpt ... --model_dir ... --prompt_file prompts.txt --out_dir /tmp/out
#
# Key flags:
#   --num_steps    Denoising steps (match student_sample_steps, default 4)
#   --cfg_scale    CFG strength (default 4.0)
#   --resolution   Output resolution in pixels (default 1024)
#   --use_ema      Load the EMA student (key "ema_net") instead of "net"
#   --max_latent_size  Must match training value (default 96)

import argparse
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image

# ── Make sure FastGen root is on the path ────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def prompt_to_filename(prompt: str, index: int, max_length: int = 120) -> str:
    """Convert a prompt into a filesystem-safe filename stem."""
    stem = re.sub(r"\s+", "_", prompt.strip())
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", stem)
    stem = stem.strip("._")
    if not stem:
        stem = f"prompt_{index:04d}"
    return stem[:max_length].rstrip("._") or f"prompt_{index:04d}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True,  help="FastGen .pth checkpoint")
    p.add_argument("--model_dir",  required=True,  help="BAGEL-7B-MoT model directory")
    p.add_argument("--out_dir",    required=True,  help="Directory to save generated images")
    p.add_argument("--prompts",    nargs="+",      help="One or more prompts (quoted strings)")
    p.add_argument("--prompt_file",               help="Text file with one prompt per line")
    p.add_argument("--num_steps",  type=int,   default=4,    help="Denoising steps")
    p.add_argument("--cfg_scale",  type=float, default=4.0,  help="CFG guidance scale")
    p.add_argument("--resolution", type=int,   default=1024, help="Output resolution (pixels)")
    p.add_argument("--max_latent_size", type=int, default=96)
    p.add_argument("--use_ema",    action="store_true", help="Load ema_net instead of net")
    p.add_argument(
        "--timestep_shift", type=float, default=1.0,
        help=(
            "Timestep shift for generate_image.  "
            "DMD-trained students use raw t∈[0,1] during training (the DMD fast path "
            "skips sigmoid/shift), so inference must use shift=1.0 (identity).  "
            "The pretrained Kokoro teacher uses 3.0 — only pass that when running "
            "the *teacher* directly, not the DMD student."
        ),
    )
    p.add_argument("--cfg_renorm_min",  type=float, default=0.0)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def build_model(model_dir: str, max_latent_size: int) -> torch.nn.Module:
    """Build Bagel architecture from checkpoint directory (no weights loaded)."""
    from fastgen.networks.kokoro.kokoro_network import KokoroNetwork
    bagel = KokoroNetwork._build_bagel(
        ckpt_dir          = model_dir,
        latent_patch_size = 2,
        max_latent_size   = max_latent_size,
        visual_gen        = True,
        visual_und        = True,
        timestep_shift    = 1.0,   # overridden per-call in generate_image
    )
    return bagel


def load_student_weights(bagel: torch.nn.Module, ckpt_prefix: str, use_ema: bool = False):
    """Load student weights from a FastGen FSDP DCP checkpoint into KokoroNetwork.

    FSDPCheckpointer saves each model component as a DCP directory:
        <prefix>.net_model/          ← student (KokoroNetwork) weights
        <prefix>.fake_score_model/   ← fake score weights
        <prefix>.pth                 ← scheduler / iteration metadata

    DCP stores tensors sharded across ranks.  We load the full (aggregated)
    state dict using FileSystemReader + dcp.load with a plain dict, which
    works fine on a single CPU/GPU without initialising torch.distributed.

    Returns:
        KokoroNetwork with weights loaded.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    # Strip .pth suffix if the user passed the full metadata path
    if ckpt_prefix.endswith(".pth"):
        ckpt_prefix = ckpt_prefix[:-4]

    # EMA is saved under the model_dict key "ema" -> "<prefix>.ema_model" (not
    # "ema_net_model"); must match config.model.use_ema=["ema"] + checkpointer naming.
    model_key = "ema" if use_ema else "net"
    dcp_dir = f"{ckpt_prefix}.{model_key}_model"
    if not os.path.isdir(dcp_dir):
        parent = os.path.dirname(ckpt_prefix)
        available = [d for d in os.listdir(parent) if d.endswith("_model")]
        raise FileNotFoundError(
            f"DCP model dir not found: {dcp_dir}\n"
            f"Available in {parent}: {available}"
        )

    print(f"Loading FSDP DCP checkpoint from: {dcp_dir}")

    # Wrap bagel in a minimal KokoroNetwork so the state_dict keys match
    # what was saved ("bagel.*", "noise_scheduler.*", etc.)
    from fastgen.networks.kokoro.kokoro_network import KokoroNetwork
    net = KokoroNetwork(
        bagel_model   = bagel,
        schedule_type = "rf",
        net_pred_type = "v",
    )
    # Get the model's regular (non-sharded) state dict as a template.
    # dcp.load will aggregate the DCP shards and fill in the correct values.
    state_dict = net.state_dict()
    dcp.load(state_dict, storage_reader=FileSystemReader(dcp_dir))
    info = net.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys : {len(info.missing_keys)}")
    print(f"  Unexpected   : {len(info.unexpected_keys)}")

    pth_path = f"{ckpt_prefix}.pth"
    if os.path.isfile(pth_path):
        meta = torch.load(pth_path, map_location="cpu", weights_only=False)
        print(f"  Checkpoint iteration: {meta.get('iteration', '?')}")
    print(f"  Loaded key: '{model_key}'")
    return net


def load_vae(model_dir: str):
    """Load the VAE decoder from the model directory."""
    from fastgen.networks.kokoro.autoencoder import load_ae
    ae_path = os.path.join(model_dir, "ae.safetensors")
    vae, _ = load_ae(local_path=ae_path)
    return vae


def get_tokenizer_and_token_ids(model_dir: str):
    """Load tokenizer and build new_token_ids dict expected by Bagel.prepare_prompts."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    new_token_ids = {
        "bos_token_id":          tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "eos_token_id":          tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "start_of_image":        tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_image":          tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        "cut_continue_token_id": None,
        "cut_shift_token_id":    None,
        "story_mode_token_id":   None,
        "diff_mode_token_id":    None,
    }
    return tokenizer, new_token_ids


def decode_latent(latent: torch.Tensor, resolution: int, vae, device):
    """Unpatchify and decode a single packed latent to a PIL Image."""
    patch_size = 2
    vae_scale  = 8
    h_lat = resolution // (patch_size * vae_scale)   # e.g. 1024/16 = 64
    w_lat = h_lat

    # Bagel generate_image returns latent in patchified layout:
    # (h_lat, w_lat, patch_size, patch_size, 16)
    latent = latent.reshape(1, h_lat, w_lat, patch_size, patch_size, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h_lat * patch_size, w_lat * patch_size)
    image = vae.decode(latent.to(device))
    img_np = (
        (image * 0.5 + 0.5).clamp(0, 1)[0]
        .permute(1, 2, 0)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(img_np)


def decode_packed_x0(
    packed_x0: torch.Tensor,
    resolution: int,
    vae,
    device,
):
    patch_size = 2
    vae_scale = 8
    h_lat = resolution // (patch_size * vae_scale)
    w_lat = h_lat

    latent = packed_x0.reshape(1, h_lat, w_lat, patch_size, patch_size, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h_lat * patch_size, w_lat * patch_size)
    image = vae.decode(latent.to(device))
    img_np = (
        (image * 0.5 + 0.5).clamp(0, 1)[0]
        .permute(1, 2, 0)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(img_np)


def generate_student(
    prompt: str,
    net: torch.nn.Module,
    vae,
    tokenizer,
    new_token_ids: dict,
    device: torch.device,
    resolution: int = 1024,
    num_steps: int = 4,
    student_sample_type: str = "sde",
    cfg_scale: float = 4.0,
    cfg_renorm_min: float = 0.0,
):
    from fastgen.networks.kokoro.bagel import NaiveCache

    bagel = net.bagel

    past_kv = NaiveCache(bagel.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]

    gen_input, newlens, new_rope = bagel.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    gen_input = {k: v.to(device) for k, v in gen_input.items() if isinstance(v, torch.Tensor)}

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        past_kv = bagel.forward_cache_update_text(past_kv, **gen_input)

    lat_input = bagel.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(resolution, resolution)],
        new_token_ids=new_token_ids,
    )
    lat_input = {k: v.to(device) for k, v in lat_input.items() if isinstance(v, torch.Tensor)}

    cfg_kv = NaiveCache(bagel.config.llm_config.num_hidden_layers)
    cfg_newlens = [0]
    cfg_new_rope = [0]
    cfg_lat = bagel.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope,
        image_sizes=[(resolution, resolution)],
    )
    cfg_lat = {k: v.to(device) for k, v in cfg_lat.items() if isinstance(v, torch.Tensor)}

    t_sched = net.noise_scheduler.get_t_list(sample_steps=num_steps, device=device)
    x = lat_input["packed_init_noises"] * t_sched[0]
    x0 = None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for t_cur, t_next in zip(t_sched[:-1], t_sched[1:]):
            timestep = torch.full((x.shape[0],), float(t_cur), device=device, dtype=x.dtype)
            v_t = bagel._forward_flow(
                x_t=x,
                timestep=timestep,
                packed_vae_token_indexes=lat_input["packed_vae_token_indexes"],
                packed_vae_position_ids=lat_input["packed_vae_position_ids"],
                packed_text_ids=lat_input["packed_text_ids"],
                packed_text_indexes=lat_input["packed_text_indexes"],
                packed_indexes=lat_input["packed_indexes"],
                packed_position_ids=lat_input["packed_position_ids"],
                packed_seqlens=lat_input["packed_seqlens"],
                key_values_lens=lat_input["key_values_lens"],
                past_key_values=past_kv,
                packed_key_value_indexes=lat_input["packed_key_value_indexes"],
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type="global",
                cfg_text_scale=cfg_scale,
                cfg_text_packed_position_ids=cfg_lat["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=cfg_lat["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=cfg_lat["cfg_key_values_lens"],
                cfg_text_past_key_values=cfg_kv,
                cfg_text_packed_key_value_indexes=cfg_lat["cfg_packed_key_value_indexes"],
            )
            x0 = x - timestep.unsqueeze(-1) * v_t

            if t_next > 0:
                if student_sample_type == "sde":
                    eps = torch.randn_like(x0)
                else:
                    eps = (x - (1.0 - timestep.unsqueeze(-1)) * x0) / (
                        timestep.unsqueeze(-1).clamp(min=1e-8)
                    )
                x = (1.0 - t_next) * x0 + t_next * eps

    return decode_packed_x0(x0.float(), resolution, vae, device)


def generate_x0_resample(
    prompt: str,
    bagel: torch.nn.Module,
    vae,
    tokenizer,
    new_token_ids: dict,
    device: torch.device,
    resolution: int = 1024,
    num_steps: int = 4,
    cfg_scale: float = 4.0,
    cfg_renorm_min: float = 0.0,
    timestep_shift: float = 1.0,
    sample_type: str = "sde",
):
    from fastgen.networks.kokoro.bagel import NaiveCache

    past_kv = NaiveCache(bagel.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]

    gen_input, newlens, new_rope = bagel.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    gen_input = {k: v.to(device) for k, v in gen_input.items() if isinstance(v, torch.Tensor)}

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        past_kv = bagel.forward_cache_update_text(past_kv, **gen_input)

    lat_input = bagel.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(resolution, resolution)],
        new_token_ids=new_token_ids,
    )
    lat_input = {k: v.to(device) for k, v in lat_input.items() if isinstance(v, torch.Tensor)}

    cfg_kv = NaiveCache(bagel.config.llm_config.num_hidden_layers)
    cfg_newlens = [0]
    cfg_new_rope = [0]
    cfg_lat = bagel.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope,
        image_sizes=[(resolution, resolution)],
    )
    cfg_lat = {k: v.to(device) for k, v in cfg_lat.items() if isinstance(v, torch.Tensor)}

    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
    timesteps = timesteps[:-1]

    x = lat_input["packed_init_noises"] * timesteps[0]
    x0 = None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, t_cur in enumerate(timesteps):
            t_scalar = float(t_cur)
            timestep = torch.full((x.shape[0],), t_scalar, device=device, dtype=x.dtype)
            v_t = bagel._forward_flow(
                x_t=x,
                timestep=timestep,
                packed_vae_token_indexes=lat_input["packed_vae_token_indexes"],
                packed_vae_position_ids=lat_input["packed_vae_position_ids"],
                packed_text_ids=lat_input["packed_text_ids"],
                packed_text_indexes=lat_input["packed_text_indexes"],
                packed_indexes=lat_input["packed_indexes"],
                packed_position_ids=lat_input["packed_position_ids"],
                packed_seqlens=lat_input["packed_seqlens"],
                key_values_lens=lat_input["key_values_lens"],
                past_key_values=past_kv,
                packed_key_value_indexes=lat_input["packed_key_value_indexes"],
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type="global",
                cfg_text_scale=cfg_scale,
                cfg_text_packed_position_ids=cfg_lat["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=cfg_lat["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=cfg_lat["cfg_key_values_lens"],
                cfg_text_past_key_values=cfg_kv,
                cfg_text_packed_key_value_indexes=cfg_lat["cfg_packed_key_value_indexes"],
            )
            x0 = x - timestep.unsqueeze(-1) * v_t

            if i + 1 < len(timesteps):
                t_next = float(timesteps[i + 1])
                if sample_type == "sde":
                    eps = torch.randn_like(x0)
                else:
                    eps = (x - (1.0 - timestep.unsqueeze(-1)) * x0) / timestep.unsqueeze(-1).clamp(min=1e-8)
                x = (1.0 - t_next) * x0 + t_next * eps

    return decode_packed_x0(x0.float(), resolution, vae, device)


def main():
    args = parse_args()

    # ── Prompts ───────────────────────────────────────────────────────────────
    if args.prompt_file:
        prompts = [l.strip() for l in open(args.prompt_file) if l.strip()]
    elif args.prompts:
        prompts = args.prompts
    else:
        raise ValueError("Provide --prompts or --prompt_file")

    # ── Setup ─────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Building Bagel from {args.model_dir} ...")
    bagel = build_model(args.model_dir, args.max_latent_size)
    net = load_student_weights(bagel, args.ckpt, use_ema=args.use_ema)
    net = net.to(device).eval()

    print("Loading VAE ...")
    vae = load_vae(args.model_dir).to(device).eval()

    tokenizer, new_token_ids = get_tokenizer_and_token_ids(args.model_dir)

    # ── Generate ──────────────────────────────────────────────────────────────
    used_names = set()
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt}")
        img = generate_student(
            prompt         = prompt,
            net            = net,
            vae            = vae,
            tokenizer      = tokenizer,
            new_token_ids  = new_token_ids,
            device         = device,
            resolution     = args.resolution,
            num_steps      = args.num_steps,
            cfg_scale      = args.cfg_scale,
            cfg_renorm_min = args.cfg_renorm_min,
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
        print(f"  Saved → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
