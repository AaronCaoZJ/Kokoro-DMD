"""Stitch several batch_eval output dirs into comparison grids and compute two dependency-free metrics.

    python kokoro_ode/eval/compare.py --root <eval_root> --dirs <teacher_dir> <model_dir_1> <model_dir_2> ... \
        --out <eval_root>/compare_xxx

- Grids: one image per heldout case (rows = seeds, columns = models; column 1 is GT, column 2 is the
  reference image for i2i), plus a seed-0 overview.
- Sharpness: variance of the Laplacian of the grayscale image (short side resized to 512), divided per
  case by the teacher's value at the same seed, reported as a geometric mean (1 = as sharp as the teacher).
- Diversity: mean pairwise pixel difference between the seeds of the same case (L1 at 256 px in [0,1]),
  averaged per model; compare with the teacher to spot mode collapse.
Directories in --dirs are relative to --root; the first one is the reference (should be the teacher).
"""

import argparse
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

HELDOUT = os.environ.get("HELDOUT_DIR", "")


def find_images(d):
    """Collect images by file name <task_id>__<label>.png, independent of the results.json layout."""
    imgs = {}
    for f in os.listdir(os.path.join(d, "images")):
        if f.endswith(".png") and "__" in f:
            tid = f.rsplit("__", 1)[0]
            imgs[tid] = os.path.join(d, "images", f)
    return imgs


def lap_var(im):
    g = np.asarray(im.convert("L"), dtype=np.float32)
    s = 512 / min(g.shape)
    if s < 1:
        g = np.asarray(im.convert("L").resize((round(g.shape[1] * s), round(g.shape[0] * s))), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def small(im, n=256):
    return np.asarray(im.convert("RGB").resize((n, n)), dtype=np.float32) / 255.0


def thumb(p, h=180):
    if not p or not os.path.exists(p):
        return Image.new("RGB", (h, h), (40, 40, 40))
    im = Image.open(p).convert("RGB")
    return im.resize((max(1, int(im.width * h / im.height)), h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("RUNS_DIR", "runs"), "kokoro_dmd_ode", "eval"))
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = [os.path.basename(d.rstrip("/")) for d in args.dirs]
    imgs = {n: find_images(os.path.join(args.root, d)) for n, d in zip(names, args.dirs)}
    cases = [json.loads(l) for l in open(os.path.join(HELDOUT, "cases.jsonl"), encoding="utf-8")]
    seeds = sorted({int(t.rsplit("_s", 1)[1]) for t in imgs[names[0]]})

    # Metrics
    sharp = defaultdict(list)
    div = defaultdict(list)
    lv_cache = {}
    for c in cases:
        for n in names:
            per_seed = []
            for s in seeds:
                p = imgs[n].get(f"{c['uid']}_s{s}")
                if p is None:
                    continue
                im = Image.open(p)
                lv_cache[(n, c["uid"], s)] = lap_var(im)
                per_seed.append(small(im))
            if len(per_seed) >= 2:
                div[n].append(np.mean([np.abs(a - b).mean() for a, b in itertools.combinations(per_seed, 2)]))
        for n in names:
            for s in seeds:
                a, b = lv_cache.get((n, c["uid"], s)), lv_cache.get((names[0], c["uid"], s))
                if a and b:
                    sharp[n].append(math.log(a / b))
    lines = ["| model | sharpness (vs ref, geo-mean) | seed diversity (L1) | images |", "|---|---|---|---|"]
    summary = {}
    for n in names:
        sh = math.exp(np.mean(sharp[n])) if sharp[n] else float("nan")
        dv = float(np.mean(div[n])) if div[n] else float("nan")
        summary[n] = {"sharpness_vs_ref": sh, "seed_diversity_l1": dv, "n_images": len(imgs[n])}
        lines.append(f"| {n} | {sh:.3f} | {dv:.4f} | {len(imgs[n])} |")
    open(os.path.join(args.out, "metrics.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(summary, open(os.path.join(args.out, "metrics.json"), "w"), ensure_ascii=False, indent=1)
    print("\n".join(lines))

    # Grids: one per case (rows = seeds), plus a seed-0 overview
    def row_for(c, s):
        cells = [thumb(os.path.join(HELDOUT, c["gt"]))]
        if c.get("ref"):
            cells.append(thumb(os.path.join(HELDOUT, c["ref"])))
        cells += [thumb(imgs[n].get(f"{c['uid']}_s{s}")) for n in names]
        w = sum(x.width for x in cells) + 6 * len(cells)
        row = Image.new("RGB", (w, 196), "white")
        x = 0
        for cell in cells:
            row.paste(cell, (x, 16))
            x += cell.width + 6
        ImageDraw.Draw(row).text((2, 2), f"{c['uid']} s{s} | GT{' | ref' if c.get('ref') else ''} | " + " | ".join(names), fill=(0, 0, 0))
        return row

    def stack(rows, path):
        sheet = Image.new("RGB", (max(r.width for r in rows), 196 * len(rows)), "white")
        for i, r in enumerate(rows):
            sheet.paste(r, (0, 196 * i))
        sheet.save(path, quality=85)

    for c in cases:
        stack([row_for(c, s) for s in seeds], os.path.join(args.out, f"{c['uid']}.jpg"))
    stack([row_for(c, seeds[0]) for c in cases], os.path.join(args.out, "overview_s0.jpg"))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
