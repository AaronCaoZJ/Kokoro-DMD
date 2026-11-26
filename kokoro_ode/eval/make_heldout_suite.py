"""Convert the heldout15 set ($HELDOUT_DIR/cases.jsonl) into a batch_eval test-suite JSON.

    HELDOUT_DIR=<heldout15_dir> python kokoro_ode/eval/make_heldout_suite.py --seeds 0,1,2,3 --out kokoro_ode/eval/heldout15_s4.json

- Resolution = the size of each case's GT image (the cases' size field is [H, W]).
- i2i text matches the training data: "<|story_mode|>, " / "<|diff_mode|>, " + prompt (mode from the case's mode field).
- The i2i source image is the case's ref image under $HELDOUT_DIR (heldout15/refs/). One task per seed
  (id = uid_s<seed>).
"""

import argparse
import json
import os

HELDOUT = os.environ.get("HELDOUT_DIR", "")
MODE_TEXT = {"story": "<|story_mode|>", "diff": "<|diff_mode|>"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(HELDOUT, "cases.jsonl"))
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = []
    for line in open(args.cases, encoding="utf-8"):
        c = json.loads(line)
        h, w = c["size"]
        is_i2i = c["kind"] == "i2i"
        prompt = f"{MODE_TEXT[c['mode']]}, {c['prompt']}" if is_i2i else c["prompt"]
        for seed in seeds:
            t = {"id": f"{c['uid']}_s{seed}", "type": "i2i" if is_i2i else "t2i", "prompt": prompt,
                 "width": w, "height": h, "seed": seed, "group": c["kind"]}
            if is_i2i:
                t["source_image"] = os.path.join(HELDOUT, c["ref"])
            tasks.append(t)
    suite = {"name": f"heldout15_s{len(seeds)}", "global_config": {}, "tasks": tasks}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(suite, f, ensure_ascii=False, indent=1)
    print(f"{len(tasks)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
