#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Batch evaluation / side-by-side multi-model comparison for FastGen Kokoro-DMD2 students.
#
# Unlike scripts/inference/kokoro_inference.py (single model, T2I only), this script can:
#   * compare MULTIPLE checkpoints (students or the teacher) in one run,
#   * run both T2I and I2I tasks,
#   * record the generation time of every image,
#   * write an HTML report that shows the prompt, the input image (i2i) and each
#     model's output side by side,
#   * override steps / seed / resolution / cfg per task.
#
# It reuses the inference path used inside training
# (KokoroNetwork.build_inference_condition + the role="student" forward + the
# packed multi-step sampling loop), so the images match what the model actually
# produces during DMD2 training -- including I2I, which kokoro_inference.py does not support.
#
# ──────────────────────────────────────────────────────────────────────────
# Quick T2I (compare two checkpoints):
#   python scripts/inference/batch_eval.py \
#       --model_dir ${BAGEL_MODEL_DIR} \
#       --ckpts  <run_dir>/checkpoints/0000500 <run_dir>/exported/student_0000550.safetensors \
#       --prompts "a red panda sitting on a bamboo" "ganyu (genshin impact)" \
#       --out_dir outputs/eval/cmp_500_550
#
# Full test suite (T2I + I2I) over checkpoints given as .pth/DCP prefixes:
#   python scripts/inference/batch_eval.py \
#       --model_dir ${BAGEL_MODEL_DIR} \
#       --test-json <suite.json> \
#       --ckpts <run_dir>/checkpoints/0001000 <run_dir>/checkpoints/0002000 \
#       --use_ema \
#       --out_dir outputs/eval/ckpt_compare
#
# Full per-model control (mix students and the teacher; set ema/role/label per model):
#   python scripts/inference/batch_eval.py \
#       --model_dir ${BAGEL_MODEL_DIR} \
#       --models-json <models.json> \
#       --test-json   <suite.json> \
#       --out_dir     outputs/eval/student_vs_teacher
#
# A suite JSON has the form {"name", "global_config", "tasks"}; see
# kokoro_ode/eval/make_heldout_suite.py for a generator.
#
# Accepted checkpoint path forms for each --ckpts entry:
#   <run_dir>/checkpoints/0000500        DCP prefix (loads .net_model; .ema_model with --use_ema)
#   <run_dir>/checkpoints/0000500.pth    same as above (the .pth suffix is stripped)
#   <run_dir>/exported/student_0000550.safetensors   flat consolidated student weights (same format as the teacher, interchangeable)
#   <bagel_dir>/ema.safetensors          original teacher weights (use with --role teacher)
# Append "::LABEL" to any entry to set its display name, e.g. "<run_dir>/...0000500::step500".

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path

import torch
from PIL import Image

# ── Make sure the FastGen repo root is importable ───────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# Student sampling schedule used by config_kokoro_dmd2.py (Wan 4-step). Used as
# the default t_list when num_steps == 4 and no t_list is given. Other step
# counts fall back to a uniform 1 → 0 linspace.
DEFAULT_T_LIST = [0.999, 0.937, 0.833, 0.624, 0.0]

DEFAULT_CONFIG = {
    "num_steps": 4,
    "sample_type": "ode",      # keep in sync with student_sample_type in the training config
    "t_list": None,            # explicit descending schedule (last entry 0.0)
    "seed": 42,
    "resolution": 1024,        # square fallback resolution when a t2i task gives no width/height
    "width": None,
    "height": None,
    "cfg_text_scale": 4.0,     # only used when role == "teacher"
    "cfg_img_scale": 2.0,      # only used when role == "teacher" and the task is i2i
}


# ──────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────
def _build_net(model_dir: str, max_latent_size: int,
               cfg_text_max: float, cfg_img_max: float, pdd_grid_steps: int = 0):
    """Build a KokoroNetwork (bagel + wrapper) without loading any weights."""
    from fastgen.networks.kokoro.kokoro_network import KokoroNetwork

    bagel = KokoroNetwork._build_bagel(
        ckpt_dir          = model_dir,
        latent_patch_size = 2,
        max_latent_size   = max_latent_size,
        visual_gen        = True,
        visual_und        = True,    # i2i needs the ViT understanding branch
        timestep_shift    = 1.0,     # DMD students are trained on raw t∈[0,1]
    )
    net = KokoroNetwork(
        bagel_model   = bagel,
        bagel_ckpt_dir= model_dir,   # lets build_inference_condition load tokenizer/vae/transforms
        schedule_type = "rf",
        net_pred_type = "v",
        cfg_text_max  = cfg_text_max,
        cfg_img_max   = cfg_img_max,
        pdd_grid_steps= pdd_grid_steps,   # PDD student: N parallel heads (bagel.pdd_heads)
    )
    return net


def _detect_ckpt_kind(ckpt_path: str, use_ema: bool):
    """Return ('dcp', dcp_dir) or ('flat', file_path)."""
    prefix = ckpt_path[:-4] if ckpt_path.endswith(".pth") else ckpt_path
    dcp_dir = f"{prefix}.{'ema' if use_ema else 'net'}_model"
    if os.path.isdir(dcp_dir):
        return "dcp", dcp_dir
    # Also accept a path pointing directly at a *_model directory
    if ckpt_path.rstrip("/").endswith("_model") and os.path.isdir(ckpt_path):
        return "dcp", ckpt_path.rstrip("/")
    if os.path.isfile(ckpt_path):
        return "flat", ckpt_path
    raise FileNotFoundError(
        f"Cannot resolve checkpoint '{ckpt_path}'. Tried DCP dir '{dcp_dir}' "
        f"and a flat file. (For DCP keep the .net_model/.ema_model dirs; for a consolidated file pass a "
        f".safetensors/.pt/.pth path.)"
    )


def load_model(model_dir: str, ckpt_path: str, *, use_ema: bool, role: str,
               max_latent_size: int, device: torch.device,
               cfg_text_max: float, cfg_img_max: float, pdd_grid_steps: int = 0):
    """Build the network and load weights from a DCP checkpoint or a flat consolidated safetensors/pt file.
    Returns the network in eval() mode on `device`.
    PDD students: if a flat safetensors file contains pdd_heads.weight, N is set to its row count / 64;
    for DCP, pass pdd_grid_steps explicitly."""
    from fastgen.networks.kokoro.kokoro_network import KokoroNetwork

    kind, src = _detect_ckpt_kind(ckpt_path, use_ema)
    if kind == "flat" and src.endswith(".safetensors") and not pdd_grid_steps:
        from safetensors import safe_open
        with safe_open(src, framework="pt") as f:
            if "pdd_heads.weight" in f.keys():
                pdd_grid_steps = f.get_slice("pdd_heads.weight").get_shape()[0] // 64
                print(f"        detected PDD student: pdd_grid_steps={pdd_grid_steps}")
    net = _build_net(model_dir, max_latent_size, cfg_text_max, cfg_img_max, pdd_grid_steps)

    if kind == "dcp":
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader

        print(f"  [DCP] loading {'ema' if use_ema else 'net'} shards: {src}")
        # Build the template state dict before init_vae() so that it only holds the saved
        # bagel.* (+ scheduler) keys. (Otherwise vae.* would be in the template but not in
        # storage, and dcp.load would fail.)
        state_dict = net.state_dict()
        dcp.load(state_dict, storage_reader=FileSystemReader(src))
        info = net.load_state_dict(state_dict, strict=False)
        print(f"        missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
        meta_path = (ckpt_path[:-4] if ckpt_path.endswith('.pth') else ckpt_path) + ".pth"
        if os.path.isfile(meta_path):
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            print(f"        iteration: {meta.get('iteration', '?')}")
    else:
        print(f"  [flat] loading bagel weights from flat file: {src}")
        KokoroNetwork.load_bagel_weights(net.bagel, src)

    # VAE for decoding (build_inference_condition only loads it lazily for i2i).
    net.init_vae()
    net = net.to(device).eval().requires_grad_(False)
    print(f"        role={role}  device={device}")
    return net


# ──────────────────────────────────────────────────────────────────────────
# Generation (ode uses kokoro_dmd2_ode.ode_sample, the same function as the training
# rollout; sde uses a simple bf16 loop)
# ──────────────────────────────────────────────────────────────────────────
def _resolve_t_list(cfg, device, dtype):
    raw = cfg.get("t_list")
    if raw:
        t = [float(x) for x in raw]
    elif int(cfg["num_steps"]) == len(DEFAULT_T_LIST) - 1:
        t = list(DEFAULT_T_LIST)
    else:
        n = int(cfg["num_steps"])
        t = torch.linspace(1.0, 0.0, n + 1).tolist()
    if abs(t[-1]) > 1e-8:
        t = t + [0.0]
    return torch.tensor(t, device=device, dtype=dtype)


def _round16(v: int) -> int:
    return max(16, int(round(v / 16)) * 16)


def _target_hw(net, cfg, source_image):
    """Determine the output (H, W) in pixels, rounded to multiples of 16."""
    if cfg.get("width") and cfg.get("height"):
        return _round16(int(cfg["height"])), _round16(int(cfg["width"]))
    if source_image is not None:
        # Derive the size with the same VAE transform used for the i2i context, so the
        # output follows the (cropped/aligned) source image rather than its raw size.
        net._ensure_image_transforms()
        try:
            resized = net._vae_transform.resize_transform(source_image.convert("RGB"))
            w, h = resized.size
            return _round16(h), _round16(w)
        except Exception:
            pass
    r = _round16(int(cfg["resolution"]))
    return r, r


def _to_pil(image_bchw: torch.Tensor) -> Image.Image:
    img = (image_bchw.float() * 0.5 + 0.5).clamp(0, 1)[0]
    img = img.permute(1, 2, 0).mul(255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(img)


@torch.no_grad()
def _generate_sde(net, cond, t_list, n_vae, patch_dim, model_dtype, role, device):
    """SDE sampling (fresh noise at every step), with bf16 state."""
    x = torch.randn(n_vae, patch_dim, device=device, dtype=model_dtype) * t_list[0]
    x_pred = x
    h_tok, w_tok = cond["patchified_vae_latent_shapes"][0]
    for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
        t_tok = t_cur.expand(n_vae).to(model_dtype)
        x_pred = net(x, t_tok, condition=cond, fwd_pred_type="x0", role=role)
        if float(t_next) > 0:
            x = (1.0 - t_next) * x_pred + t_next * torch.randn_like(x_pred)
    return net.decode_packed_x0_first(x_pred.float(), (h_tok, w_tok))


@torch.no_grad()
def generate(net, *, task_type, prompt, source_image, cfg, role, device):
    """Run one generation. Returns (PIL.Image, resolved metadata dict)."""
    bagel = net.bagel
    p = bagel.latent_patch_size
    c = bagel.latent_channel
    vae_ds = bagel.latent_downsample // p          # pixels per latent cell (=8)
    patch_dim = bagel.patch_latent_dim             # =64

    H, W = _target_hw(net, cfg, source_image)
    h_lat, w_lat = H // vae_ds, W // vae_ds
    vae_latent = torch.zeros(c, h_lat, w_lat, device=device)   # only carries the shape

    sample = {"task_type": task_type, "text": prompt, "vae_latent": vae_latent}
    if task_type == "i2i":
        sample["source_image"] = source_image.convert("RGB")

    model_dtype = next(bagel.parameters()).dtype
    use_cuda = device.type == "cuda"

    with torch.amp.autocast("cuda", enabled=use_cuda, dtype=torch.bfloat16):
        cond = net.build_inference_condition(
            sample, device,
            cfg_text_scale=float(cfg["cfg_text_scale"]),
            cfg_img_scale=float(cfg["cfg_img_scale"]),
        )

        h_tok, w_tok = cond["patchified_vae_latent_shapes"][0]
        n_vae = h_tok * w_tok
        t_list = _resolve_t_list(cfg, device, model_dtype)

        sample_type = cfg.get("sample_type", "ode")
        if getattr(net, "pdd_grid_steps", 0) > 0 and role == "student":
            # PDD: fine N-cell grid (same shift as training), one forward per block,
            # NFE = num_steps (same pdd_sample as training)
            from fastgen.methods.distribution_matching.kokoro_dmd2_ode import pdd_grid, pdd_sample
            grid = pdd_grid(net.pdd_grid_steps, float(cfg.get("pdd_shift", 5.0)))
            block = net.pdd_grid_steps // int(cfg["num_steps"])
            noise = torch.randn(n_vae, patch_dim, device=device, dtype=torch.float32)
            x_pred = pdd_sample(net, noise, grid, block, cond, model_dtype)
            image_bchw = net.decode_packed_x0_first(x_pred.float(), (h_tok, w_tok))
            t_list = torch.tensor(grid[0::block])
            sample_type = f"pdd(N={net.pdd_grid_steps},block={block})"
        elif sample_type == "ode":
            # Same integrator as the training rollout (KokoroDMD2ODE / ode_sample): Euler with
            # fp32 state and t quantized to network precision
            from fastgen.methods.distribution_matching.kokoro_dmd2_ode import ode_sample, quantize_t
            t_vals = [quantize_t(v, model_dtype) for v in t_list.tolist()]
            noise = torch.randn(n_vae, patch_dim, device=device, dtype=torch.float32)
            x_pred = ode_sample(net, noise, t_vals, cond, model_dtype, role=role)
            image_bchw = net.decode_packed_x0_first(x_pred.float(), (h_tok, w_tok))
            t_list = torch.tensor(t_vals)
        else:
            image_bchw = _generate_sde(net, cond, t_list, n_vae, patch_dim, model_dtype, role, device)

    meta = {
        "width": W, "height": H,
        "num_steps": int(len(t_list) - 1),
        "t_list": [round(float(v), 4) for v in t_list.tolist()],
        "sample_type": sample_type,
        "seed": int(cfg["seed"]),
    }
    if role == "teacher":
        meta["cfg_text_scale"] = float(cfg["cfg_text_scale"])
        if task_type == "i2i":
            meta["cfg_img_scale"] = float(cfg["cfg_img_scale"])
    return _to_pil(image_bchw), meta


# ──────────────────────────────────────────────────────────────────────────
# Test-suite / model-spec parsing
# ──────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prompt_to_stem(prompt: str, index: int, max_len: int = 80) -> str:
    stem = re.sub(r"\s+", "_", prompt.strip())
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", stem).strip("._")
    return (stem[:max_len].rstrip("._") or f"task_{index:03d}")


def resolve_config(global_config, task):
    cfg = {**DEFAULT_CONFIG}
    cfg.update({k: v for k, v in global_config.items() if k in DEFAULT_CONFIG})
    cfg.update({k: task[k] for k in DEFAULT_CONFIG if k in task})
    return cfg


def parse_models(args):
    """Return a list of {path, label, use_ema, role}."""
    models = []
    if args.models_json:
        with open(args.models_json) as f:
            entries = json.load(f)
        for e in entries:
            models.append({
                "path": e["path"],
                "label": e.get("label") or Path(e["path"]).stem,
                "use_ema": bool(e.get("use_ema", args.use_ema)),
                "role": e.get("role", args.role),
            })
    else:
        for spec in args.ckpts:
            path, _, label = spec.partition("::")
            models.append({
                "path": path,
                "label": label or _auto_label(path),
                "use_ema": args.use_ema,
                "role": args.role,
            })
    # De-duplicate display names
    seen = {}
    for m in models:
        base = m["label"]
        n = seen.get(base, 0)
        seen[base] = n + 1
        if n:
            m["label"] = f"{base}#{n+1}"
    return models


def _auto_label(path: str) -> str:
    parts = Path(path).parts
    stem = Path(path).stem
    # .../checkpoints/0000500(.net_model) -> 0000500 ; exported/student_0000550 -> student_0000550
    for i, seg in enumerate(parts):
        if seg in ("checkpoints", "exported") and i + 1 < len(parts):
            nxt = parts[i + 1]
            nxt = nxt.replace(".net_model", "").replace(".ema_model", "")
            nxt = re.sub(r"\.(safetensors|pth|pt)$", "", nxt)
            return nxt
    return stem


def load_suite(args):
    """Return (suite_name, global_config, tasks)."""
    if args.test_json:
        with open(args.test_json) as f:
            data = json.load(f)
        name = data.get("name", Path(args.test_json).stem)
        return name, data.get("global_config", {}), data.get("tasks", [])

    if args.prompt_file:
        prompts = [l.strip() for l in open(args.prompt_file) if l.strip()]
    elif args.prompts:
        prompts = args.prompts
    else:
        raise ValueError("Provide --test-json, or --prompts/--prompt_file for a quick T2I suite.")
    tasks = [{"type": "t2i", "prompt": p} for p in prompts]
    return "adhoc_t2i", {}, tasks


# ──────────────────────────────────────────────────────────────────────────
# HTML report
# ──────────────────────────────────────────────────────────────────────────
# Base64 thumbnail cache keyed by (absolute path, mtime), so flush() does not re-read/re-encode.
_EMBED_CACHE: dict = {}

# The report embeds THUMBNAILS rather than full-resolution images: base64-encoding the
# originals (~1.5 MB each) makes a single HTML file tens of MB, which many viewers (e.g.
# IDE built-in previews) fail to render, leaving later rows blank. JPEG thumbnails
# (≤512 px) are a few tens of KB each, so the report stays a single self-contained file
# that renders everywhere. Full-resolution images are still saved under images/.
_THUMB_MAX = 512


def _data_uri(report_path, rel):
    """Shrink an image to a ≤_THUMB_MAX JPEG thumbnail and encode it as a base64 data URI,
    keeping the report self-contained and small. Returns None on failure."""
    if not rel:
        return None
    p = Path(report_path).parent / rel
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return None
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    try:
        import io
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((_THUMB_MAX, _THUMB_MAX), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
        uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None
    _EMBED_CACHE[key] = uri
    return uri


def write_report(report_path, suite_name, models, tasks_meta, results):
    """results[(task_id, model_label)] = {output_path|None, elapsed, error}"""
    labels = [m["label"] for m in models]
    has_i2i = any(t["type"] == "i2i" for t in tasks_meta)

    def cell_img(rel, extra=""):
        uri = _data_uri(report_path, rel)
        if not uri:
            return '<div class="missing">— no image —</div>'
        # No loading="lazy": embedded thumbnails are already small, and some viewers
        # (e.g. IDE built-in HTML previews) never trigger lazy loading for off-screen
        # images, leaving the last rows blank. Eager loading shows everything.
        return f'<img src="{uri}">{extra}'

    rows = []
    for t in tasks_meta:
        tid = t["id"]
        head = (
            f'<div class="tid">{escape(tid)} '
            f'<span class="ttype {t["type"]}">{t["type"]}</span></div>'
            f'<div class="prompt">{escape(t.get("prompt",""))}</div>'
        )
        # If the suite has any i2i task the header has an "Input" column; t2i rows must
        # then emit an (empty) input cell too, or every model column shifts left by one.
        if has_i2i:
            if t["type"] == "i2i":
                src_cell = (
                    f'<td class="srccol"><div class="lab">Input</div>'
                    f'{cell_img(t.get("source_rel"))}</td>'
                )
            else:
                src_cell = '<td class="srccol"></td>'
        else:
            src_cell = ""
        model_cells = []
        for lb in labels:
            r = results.get((tid, lb), {})
            if r.get("status") == "success":
                cap = f'<div class="time">{r["elapsed"]:.2f}s · {r.get("steps","?")} steps</div>'
                model_cells.append(f'<td>{cell_img(r["output_rel"], cap)}</td>')
            else:
                err = escape((r.get("error") or "")[:300])
                model_cells.append(f'<td><div class="fail">failed</div><pre>{err}</pre></td>')
        rows.append(
            f'<tr><td class="meta">{head}</td>{src_cell}{"".join(model_cells)}</tr>'
        )

    header_cols = '<th class="meta">task / prompt</th>'
    if any(t["type"] == "i2i" for t in tasks_meta):
        header_cols += '<th>Input</th>'
    header_cols += "".join(f"<th>{escape(lb)}</th>" for lb in labels)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{escape(suite_name)} — Kokoro-DMD2 eval</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;background:#0f1115;color:#e6e6e6}}
 h1{{font-size:18px}} .sub{{color:#9aa0aa;font-size:13px;margin-bottom:16px}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #2a2f3a;padding:8px;vertical-align:top;text-align:center}}
 th{{background:#171a21;position:sticky;top:0;font-size:13px}}
 td.meta,th.meta{{text-align:left;max-width:240px}}
 img{{max-width:256px;max-height:256px;border-radius:6px;background:#000}}
 .tid{{font-weight:600}} .ttype{{font-size:11px;padding:1px 6px;border-radius:8px;margin-left:6px}}
 .ttype.t2i{{background:#1f3a5f}} .ttype.i2i{{background:#3a1f5f}}
 .prompt{{color:#b9c0cc;font-size:12px;margin-top:6px;white-space:pre-wrap}}
 .time{{color:#7fd17f;font-size:12px;margin-top:4px}} .lab{{color:#9aa0aa;font-size:11px;margin-bottom:4px}}
 .fail{{color:#e06c6c;font-weight:600}} pre{{text-align:left;font-size:10px;color:#caa;white-space:pre-wrap}}
 .missing{{color:#666;font-size:12px;padding:24px}}
</style></head><body>
<h1>{escape(suite_name)} — Kokoro-DMD2 4-step eval</h1>
<div class="sub">{datetime.now().isoformat(timespec='seconds')} · {len(tasks_meta)} tasks · {len(models)} models:
 {escape(', '.join(f"{m['label']} [{m['role']}{'/ema' if m['use_ema'] else ''}]" for m in models))}</div>
<table><thead><tr>{header_cols}</tr></thead><tbody>
{''.join(rows)}
</tbody></table></body></html>"""
    with open(report_path, "w") as f:
        f.write(html)


# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Batch evaluation / multi-model comparison for Kokoro-DMD2 students (T2I + I2I).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model_dir", required=True, help="BAGEL-7B-MoT directory (with llm_config.json, ae.safetensors, tokenizer)")
    ap.add_argument("--ckpts", nargs="+", help="Checkpoints to compare, each PATH[::LABEL]. A DCP prefix, .pth, or flat .safetensors")
    ap.add_argument("--models-json", help="JSON list of {path,label?,use_ema?,role?} for per-model control")
    ap.add_argument("--out_dir", required=True)

    # Test suite (one of the two)
    ap.add_argument("--test-json", help="Test-suite JSON (name/global_config/tasks)")
    ap.add_argument("--prompts", nargs="+", help="Prompts for a quick T2I run (used when --test-json is not given)")
    ap.add_argument("--prompt_file", help="Quick T2I: one prompt per line")

    # Global model/role/sampling defaults (tasks in a suite can override them)
    ap.add_argument("--use_ema", action="store_true", help="Load EMA student weights (.ema_model) from DCP checkpoints")
    ap.add_argument("--role", default="student", choices=["student", "teacher"],
                    help="student = no cfg knobs (DMD); teacher = cfg-embed with the cfg scales")
    ap.add_argument("--num_steps", type=int, default=DEFAULT_CONFIG["num_steps"])
    ap.add_argument("--sample_type", default=DEFAULT_CONFIG["sample_type"], choices=["ode", "sde"])
    ap.add_argument("--t_list", default="", help="Comma-separated descending schedule, e.g. 0.999,0.937,0.833,0.624,0.0")
    ap.add_argument("--resolution", type=int, default=DEFAULT_CONFIG["resolution"])
    ap.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    ap.add_argument("--cfg_scale", type=float, default=DEFAULT_CONFIG["cfg_text_scale"], help="Teacher role only")
    ap.add_argument("--cfg_img_scale", type=float, default=DEFAULT_CONFIG["cfg_img_scale"], help="Teacher role, i2i only")

    ap.add_argument("--max_latent_size", type=int, default=128, help="Must match MAX_LATENT_SIZE used in training (128)")
    ap.add_argument("--pdd_grid_steps", type=int, default=0,
                    help="Grid size N of a PDD student (auto-detected for flat safetensors; required for DCP)")
    ap.add_argument("--pdd_shift", type=float, default=5.0, help="PDD grid shift (must match PDD_SHIFT in training)")
    ap.add_argument("--cfg_text_max", type=float, default=5.0)
    ap.add_argument("--cfg_img_max", type=float, default=2.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-existing", action="store_true", help="Skip (task, model) pairs already successful in results.json")
    args = ap.parse_args()

    if not args.ckpts and not args.models_json:
        ap.error("Provide --ckpts or --models-json")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    report_path = out_dir / "report.html"

    # CLI defaults are merged into the suite's effective global_config (suite values win).
    cli_global = {
        "num_steps": args.num_steps,
        "sample_type": args.sample_type,
        "seed": args.seed,
        "resolution": args.resolution,
        "cfg_text_scale": args.cfg_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "pdd_shift": args.pdd_shift,
    }
    if args.t_list.strip():
        cli_global["t_list"] = [float(x) for x in args.t_list.split(",") if x.strip()]

    models = parse_models(args)
    suite_name, suite_global, tasks = load_suite(args)
    global_config = {**cli_global, **suite_global}

    # Assign task ids + a fixed per-task config; copy i2i source images into the output dir.
    t2i_n = i2i_n = 0
    tasks_meta = []
    for idx, task in enumerate(tasks):
        ttype = task.get("type", "t2i")
        tid = task.get("id")
        if not tid:
            if ttype == "t2i":
                t2i_n += 1; tid = f"t2i_{t2i_n:03d}"
            else:
                i2i_n += 1; tid = f"i2i_{i2i_n:03d}"
        meta = {"id": tid, "type": ttype, "prompt": task.get("prompt", ""),
                "cfg": resolve_config(global_config, task), "source_image": None, "source_rel": None}
        if ttype == "i2i":
            src = task.get("source_image", "")
            if src and os.path.isfile(src):
                meta["source_image"] = src
                dst = images_dir / f"src__{tid}{Path(src).suffix.lower() or '.png'}"
                try:
                    shutil.copy(src, dst)
                    meta["source_rel"] = f"images/{dst.name}"
                except Exception:
                    meta["source_rel"] = src
            else:
                meta["error"] = f"source image not found: {src}"
        tasks_meta.append(meta)

    print(f"Suite     : {suite_name}")
    print(f"Models    : {len(models)} -> {[m['label'] for m in models]}")
    print(f"Tasks     : {len(tasks_meta)} ({t2i_n} t2i, {i2i_n} i2i)")
    print(f"Output    : {out_dir}")

    # Resume support.
    results = {}
    if args.skip_existing and results_path.exists():
        with open(results_path) as f:
            prev = json.load(f)
        for r in prev.get("results", []):
            results[(r["task_id"], r["model_label"])] = r
        print(f"Resuming  : loaded {len(results)} previous results")

    def flush():
        flat = list(results.values())
        ok = sum(1 for r in flat if r.get("status") == "success")
        doc = {
            "suite_name": suite_name,
            "timestamp": datetime.now().isoformat(),
            "model_dir": args.model_dir,
            "models": models,
            "summary": {"total": len(flat), "success": ok, "failed": len(flat) - ok,
                        "total_time": round(sum(r.get("elapsed", 0.0) for r in flat), 2)},
            "results": flat,
        }
        with open(results_path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        write_report(report_path, suite_name, models, tasks_meta, results)

    # Outer loop = models (only one 7B model is loaded at a time, to bound GPU memory).
    for mi, m in enumerate(models):
        print(f"\n{'='*64}\n[{mi+1}/{len(models)}] model: {m['label']}  (role={m['role']}, ema={m['use_ema']})")
        print(f"  path: {m['path']}")
        try:
            net = load_model(
                args.model_dir, m["path"], use_ema=m["use_ema"], role=m["role"],
                max_latent_size=args.max_latent_size, device=device,
                cfg_text_max=args.cfg_text_max, cfg_img_max=args.cfg_img_max,
                pdd_grid_steps=args.pdd_grid_steps,
            )
        except Exception as e:
            print(f"  model loading failed: {e}")
            traceback.print_exc()
            for t in tasks_meta:
                results[(t["id"], m["label"])] = {
                    "task_id": t["id"], "task_type": t["type"], "prompt": t["prompt"],
                    "model_label": m["label"], "status": "failed", "output_rel": None,
                    "elapsed": 0.0, "error": f"model loading failed: {e}",
                }
            flush()
            continue

        for ti, t in enumerate(tasks_meta):
            tid = t["id"]
            key = (tid, m["label"])
            if args.skip_existing and results.get(key, {}).get("status") == "success":
                print(f"  [{ti+1}/{len(tasks_meta)}] SKIP {tid}")
                continue
            if t["type"] == "i2i" and t["source_image"] is None:
                print(f"  [{ti+1}/{len(tasks_meta)}] {tid} FAIL ({t.get('error')})")
                results[key] = {
                    "task_id": tid, "task_type": t["type"], "prompt": t["prompt"],
                    "model_label": m["label"], "status": "failed", "output_rel": None,
                    "elapsed": 0.0, "error": t.get("error"),
                }
                flush(); continue

            cfg = t["cfg"]
            print(f"  [{ti+1}/{len(tasks_meta)}] {tid} [{t['type']}] | ", end="", flush=True)
            source_image = Image.open(t["source_image"]).convert("RGB") if t["source_image"] else None

            set_seed(cfg["seed"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                img, gmeta = generate(
                    net, task_type=t["type"], prompt=t["prompt"], source_image=source_image,
                    cfg=cfg, role=m["role"], device=device,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                fname = f"{tid}__{prompt_to_stem(m['label'], mi)}.png"
                img.save(images_dir / fname)
                print(f"OK {elapsed:.2f}s -> {fname}")
                results[key] = {
                    "task_id": tid, "task_type": t["type"], "prompt": t["prompt"],
                    "source_image": t["source_image"], "model_label": m["label"],
                    "model_path": m["path"], "role": m["role"], "use_ema": m["use_ema"],
                    "config": gmeta, "steps": gmeta["num_steps"],
                    "status": "success", "output_rel": f"images/{fname}",
                    "elapsed": round(elapsed, 3), "error": None,
                }
            except Exception as e:
                elapsed = time.perf_counter() - t0
                err = traceback.format_exc()
                print(f"FAIL {elapsed:.2f}s: {e}")
                results[key] = {
                    "task_id": tid, "task_type": t["type"], "prompt": t["prompt"],
                    "source_image": t["source_image"], "model_label": m["label"],
                    "status": "failed", "output_rel": None,
                    "elapsed": round(elapsed, 3), "error": err,
                }
            flush()

        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    flat = list(results.values())
    ok = sum(1 for r in flat if r.get("status") == "success")
    print(f"\n{'='*64}\nDone. success={ok} failed={len(flat)-ok} "
          f"total_gen_time={sum(r.get('elapsed',0) for r in flat):.1f}s")
    print(f"Report : {report_path}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
