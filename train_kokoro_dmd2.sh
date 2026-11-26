#!/bin/bash
# Launcher for Kokoro/Bagel DMD2 distillation in FastGen.
#
# Single node:
#   bash train_kokoro_dmd2.sh
#
# Multi-node:
#   NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.1 bash train_kokoro_dmd2.sh
#   NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.1 bash train_kokoro_dmd2.sh

set -euo pipefail

# ── Make this repository importable (uses the currently active Python environment) ──
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH=$(pwd)${PYTHONPATH:+:$PYTHONPATH}

# ── Distributed setup ─────────────────────────────────────────────────────────
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

echo "=== FastGen Kokoro DMD2 Training ==="
echo "Nodes: $NUM_NODES  |  Rank: $NODE_RANK  |  GPUs/node: $NPROC_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "====================================="

# ── Hyper-parameters ──────────────────────────────────────────────────────────
# All training hyper-parameters live in train_config.yaml; that is the only file to edit.
# Every YAML key is exported as an environment variable of the same name, which
# config_kokoro_dmd2.py reads via os.environ.get(). Variables already set (in the
# environment / on the command line) take precedence over the YAML values.
CONFIG_YAML=${CONFIG_YAML:-train_config.yaml}
eval "$(python - "$CONFIG_YAML" <<'PYEOF'
import sys, os, shlex, yaml

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in cfg.items()}  # ${VAR} placeholders

def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

for key, val in cfg.items():
    if key in os.environ:  # environment / command line overrides the YAML value
        continue
    print(f"export {key}={shlex.quote(fmt(val))}")
PYEOF
)"

echo "=== Hyper-parameters (from $CONFIG_YAML) ==="
echo "MAX_ITER=$MAX_ITER  BATCH_SIZE_GLOBAL=$BATCH_SIZE_GLOBAL  SEED=$SEED"
echo "NET_LR=$NET_LR  FAKE_SCORE_LR=$FAKE_SCORE_LR"
echo "LOGGING_ITER=$LOGGING_ITER  SAMPLE_LOGGING_ITER=$SAMPLE_LOGGING_ITER  SAVE_CKPT_ITER=$SAVE_CKPT_ITER"
echo "WANDB: $WANDB_PROJECT/$WANDB_GROUP/$WANDB_NAME  mode=$WANDB_MODE  run_id=$WANDB_RUN_ID"
echo "OUTPUT=$FASTGEN_OUTPUT_ROOT"
echo "GAN: weight_gen=${GAN_LOSS_WEIGHT_GEN:-?}  r1_weight=${GAN_R1_REG_WEIGHT:-?}  disc_lr=${DISC_LR:-?}  (>0 = GAN active; =0 = pure VSD+DSM)"
echo "EMA: USE_EMA=${USE_EMA:-?}  FSDP_META_INIT=${FSDP_META_INIT:-?}"
echo "================================="

# ── Host-memory safety fence ──────────────────────────────────────────────────
# DataLoader host memory can keep growing until it hits the container's cgroup memory.max;
# the kernel OOM killer then wedges the whole container (even SSH can no longer fork).
# scripts/mem_guard.py watches unreclaimable `anon` memory and, at a threshold below the cgroup
# limit, SIGKILLs *only* the training process group, so the job terminates cleanly instead of
# taking the container down.
#   MEM_GUARD=0           disable the fence
#   GUARD_LIMIT_GIB=1450  trip threshold in GiB (also capped at 90% of the cgroup limit)
MEM_GUARD=${MEM_GUARD:-1}
export GUARD_LIMIT_GIB=${GUARD_LIMIT_GIB:-1450}
export GUARD_POLL_SEC=${GUARD_POLL_SEC:-3}
PGID_FILE=$(mktemp /tmp/kokoro_train_pgid.XXXXXX)
export GUARD_PGID_FILE=$PGID_FILE

# ── Launch ────────────────────────────────────────────────────────────────────
# Run torchrun in its own session via setsid so the watchdog can kill the whole rank+worker
# tree (one process group) without touching this launcher or sshd.
# The setsid'd shell's $$ is its pgid; publish it so the watchdog can find its target.
setsid bash -c '
  p=$(ps -o pgid= -p $$); echo "${p// /}" > "'"$PGID_FILE"'"
  exec torchrun \
      --nnodes='"$NUM_NODES"' \
      --node_rank='"$NODE_RANK"' \
      --nproc_per_node='"$NPROC_PER_NODE"' \
      --master_addr='"$MASTER_ADDR"' \
      --master_port='"$MASTER_PORT"' \
      train.py \
      --config=fastgen/configs/experiments/Kokoro/config_kokoro_dmd2.py \
      '"$*"' \
  ' &
LAUNCH_PID=$!

GUARD_PID=""
if [ "$MEM_GUARD" != "0" ]; then
    python scripts/mem_guard.py &
    GUARD_PID=$!
    echo "=== mem_guard armed (limit=${GUARD_LIMIT_GIB}GiB, pid=$GUARD_PID) ==="
fi

# Wait for training to finish; capture the exit code without triggering `set -e`.
set +e
wait "$LAUNCH_PID"
TRAIN_RC=$?
set -e

# Stop the watchdog and clean up.
if [ -n "$GUARD_PID" ]; then
    kill "$GUARD_PID" 2>/dev/null || true
    wait "$GUARD_PID" 2>/dev/null || true
fi
rm -f "$PGID_FILE"

if [ -f /tmp/mem_guard_tripped ]; then
    echo "=== mem_guard TRIPPED — training was killed to protect the host: $(cat /tmp/mem_guard_tripped) ==="
fi
exit "$TRAIN_RC"
