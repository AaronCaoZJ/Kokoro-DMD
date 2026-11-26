#!/bin/bash
# Print example training prompts from the LAION caption file (500k images, aesthetic score >= 6.25).
#   CHECKPOINT_PATH=<data root> bash scripts/view_laion_prompts.sh

cd "$(dirname "$0")/.."
CHECKPOINT_PATH=${CHECKPOINT_PATH:?set CHECKPOINT_PATH to the data root (see scripts/download_sdv15.sh)}
PKL=$CHECKPOINT_PATH/captions_laion_score6.25.pkl

if [ ! -f "$PKL" ]; then
    echo "Caption file not found: $PKL"
    echo "Download it first:  bash scripts/download_sdv15.sh $CHECKPOINT_PATH"
    exit 1
fi

# first 20 prompts
python main/view_prompts.py --pkl_path "$PKL" --num_samples 20

echo ""
echo "Random samples:  python main/view_prompts.py --pkl_path $PKL --num_samples 50 --random"
