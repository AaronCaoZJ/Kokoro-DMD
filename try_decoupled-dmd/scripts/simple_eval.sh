#!/bin/bash
# Few-step sampling with a distilled SD1.5 generator (main/simple_inference_lcm.py), with CLIP score.
#   CHECKPOINT=<output_path>/<run>/checkpoint_model_XXXXXX/pytorch_model.bin OUTPUT_DIR=<dir> bash scripts/simple_eval.sh
# Optional flags for the python call: --clip_model ViT-G/14, --compute_image_reward

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd):$PYTHONPATH
export HF_HOME=${HF_HOME:-$(pwd)/ckpt}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a generator pytorch_model.bin}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/student_samples}
[ -f "$CHECKPOINT" ] || { echo "No checkpoint found at $CHECKPOINT"; exit 1; }
echo "Using checkpoint: $CHECKPOINT"

python main/simple_inference_lcm.py \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    --device cuda \
    --prompts \
        "A blue apple sitting next to a red banana on a wooden table" \
        "a beautiful landscape with mountains and a lake at sunset" \
        "a cute orange cat sitting on a windowsill" \
        "A teddy bear made of shiny chrome metal, reflecting a neon city" \
        "a portrait of a woman with flowers in her hair" \
        "a steaming cup of coffee on a wooden table" \
        "A crowd waits by along the St. Patrick's Day Parade route on 5th Avenue in 1951 New York" \
        "Ubein Bridge in Burma" \
        "The Most Romantic Hot Air Balloon Rides In The World Blog Masai Mara Nature Reserve, Kenya" \
    --compute_clip_score
