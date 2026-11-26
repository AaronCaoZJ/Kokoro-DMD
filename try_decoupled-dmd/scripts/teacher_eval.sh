#!/bin/bash
# Reference samples from the SD1.5 teacher (50-step DDIM, CFG 7.5) for the same prompts as scripts/simple_eval.sh.
#   OUTPUT_DIR=<dir> bash scripts/teacher_eval.sh

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd):$PYTHONPATH
export HF_HOME=${HF_HOME:-$(pwd)/ckpt}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}

OUTPUT_DIR=${OUTPUT_DIR:-outputs/teacher_samples}

python main/simple_inference_teacher.py \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    --device cuda \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --scheduler ddim \
    --prompts \
        "A blue apple sitting next to a red banana on a wooden table" \
        "a beautiful landscape with mountains and a lake at sunset" \
        "a cute orange cat sitting on a windowsill" \
        "A teddy bear made of shiny chrome metal, reflecting a neon city" \
        "a portrait of a woman with flowers in her hair" \
        "a steaming cup of coffee on a wooden table" \
        "A crowd waits by along the St. Patrick's Day Parade route on 5th Avenue in 1951 New York" \
        "Ubein Bridge in Burma" \
        "The Most Romantic Hot Air Balloon Rides In The World Blog Masai Mara Nature Reserve, Kenya"
