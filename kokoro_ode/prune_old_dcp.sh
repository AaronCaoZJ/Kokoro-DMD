#!/bin/bash
# Keep only the DCP of the latest complete checkpoint (for resuming) and delete older DCPs; all exported/<iter>/
# student/ema.safetensors are kept (for evaluation).
# Each save of arm B/D is ≈ 240 GB of DCP (net / fake / ema + fp32 optimizer state) + 58 GB of exports, so three
# unpruned saves already take ~0.9 TB per arm.
#
#   bash kokoro_ode/prune_old_dcp.sh <run>/checkpoints [poll_seconds]
#
# "Complete" = exported/<iter>/step.txt has been written (export happens after the DCP is on disk). Only DCP
# directories / .pth files in this directory whose names start with 7 digits are deleted, and only once a newer
# complete checkpoint exists. Started in the background by launch_arm.sh and killed when training ends.
set -u
CKPT=${1:?usage: prune_old_dcp.sh <ckpt_dir> [poll_s]}
POLL=${2:-300}
while true; do
  if [ -d "$CKPT/exported" ]; then
    latest=$(ls "$CKPT/exported" 2>/dev/null | grep -E '^[0-9]{7}$' | sort | while read -r it; do
               [ -f "$CKPT/exported/$it/step.txt" ] && echo "$it"; done | tail -1)
    if [ -n "$latest" ]; then
      for p in "$CKPT"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9]*; do
        [ -e "$p" ] || continue
        it=$(basename "$p" | cut -c1-7)
        if [[ "$it" =~ ^[0-9]{7}$ ]] && [ "$it" \< "$latest" ]; then
          echo "$(date '+%F %T') prune $p (latest complete = $latest)"
          rm -rf -- "$p"
        fi
      done
    fi
  fi
  sleep "$POLL"
done
