#!/bin/bash
# Single-node launcher for one arm (method) of ODE-rollout DMD2 distillation.
#
#   ARM=B GPUS=0,1,2,3 MASTER_PORT=29610 bash kokoro_ode/launch_arm.sh
#   ARM=D GPUS=4,5,6,7 MASTER_PORT=29620 bash kokoro_ode/launch_arm.sh
#   ARM=ODEINIT GPUS=0,1,2,3 MAX_ITER=2000 bash kokoro_ode/launch_arm.sh
#   Extra arguments are passed through to train.py (Hydra overrides).
#
# Arm → switches: every arm sets all method switches explicitly, so values never leak in from the caller's
# environment. Per-arm defaults follow the reference implementations. For variants use ARM_OVERRIDES="K=V K=V";
# they are applied last and recorded in launch.log.
#   A        STUDENT_ROLLOUT=real_noised                       DMD2 baseline: student input = real image + fresh noise
#   B        STUDENT_ROLLOUT=ode                               DMD2 + training-time ODE rollout (Self-Forcing style)
#   C        B + DMD_NOISE=trajectory FAKE_INPUT=final         Hyper-Bagel DMDO: fake score uses the trajectory endpoint interpolated with x_T
#   D        DPDMD=1 VSD_NORM=xt VSD_T_SHIFT=3                 DP-DMD official settings: x_t normaliser, shift-3 τ,
#            STUDENT_UPDATE_FREQ=6 FAKE_SCORE_LR=1e-5           fake:student = 5:1, fake lr 1e-5
#   TDM      DMD_TARGET=landing STUDENT_UPDATE_FREQ=2          TDM demo settings: fake:student = 1:1, Huber c=1e-3,
#            NET_LR=4e-6 NET_BETAS=FAKE_SCORE_BETAS=0,0.95       generator lr = lr/5, β=(0,0.95), τ lower bound 0.07 (≈ demo's t=20)
#            TDM_MIN_T=0.07
#   PDD      PDD=1 STUDENT_UPDATE_FREQ=1 NET_LR=1e-5           parallel decoding distillation (data-free Alg. 3, no fake score)
#   ODEINIT  ODE_INIT=1 STUDENT_UPDATE_FREQ=1 NET_LR=2e-6      online teacher-ODE trajectory regression (CausVid lr)
#   ODEINIT_TDM / ODEINIT_PDD  same as TDM / PDD, but the student starts from ODE_INIT_STUDENT (the student.safetensors
#            exported by ODEINIT); kokoro_ode/run_odeinit_pipeline.sh runs both stages in one go
#   RECIPE=1 (default)  constant LR; VSD τ shift, Adam betas and teacher cfg_img as set per arm / in the yaml
#   RECIPE=2  VSD τ shift 3, Adam β2 0.95, cosine LR decay to 0, teacher cfg_img 1.5 (see the case below)
# All other hyper-parameters come from CONFIG_YAML (default kokoro_ode/train_ode_dpdmd.yaml); variables already in the
# environment take precedence over the yaml.
# Timestamps use TZ=Asia/Singapore.

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PYTHON:-python}
cd "$REPO"
export TZ=Asia/Singapore
export PYTHONPATH=$REPO${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}

ARM=${ARM:?set ARM=A|B|C|D|TDM|PDD|ODEINIT|ODEINIT_TDM|ODEINIT_PDD}
GPUS=${GPUS:?set GPUS, e.g. 0,1,2,3}
MASTER_PORT=${MASTER_PORT:?set MASTER_PORT, one per arm when running arms in parallel}
export CUDA_VISIBLE_DEVICES=$GPUS
# Memory fragmentation: with variable-length packed sequences and 1024-px i2i the CUDA cache grows until the GPU is full
# (arm D peaks at 139–142 of 143.8 GB); expandable segments keep fragmentation in check.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

# ── Arm switches (exported first, so same-named defaults in the yaml do not override them) ──
export STUDENT_ROLLOUT=ode ROLLOUT_EXIT=uniform FAKE_INPUT=matched DMD_NOISE=fresh DMD_TARGET=clean \
       VSD_NORM=x0 VSD_T_SHIFT=5.0 DPDMD=0 ODE_INIT=0 PDD=0 STUDENT_UPDATE_FREQ=5 \
       NET_LR=1e-5 FAKE_SCORE_LR=2e-5 NET_BETAS=0.9,0.999 FAKE_SCORE_BETAS=0.9,0.999 \
       TDM_MIN_T=0.02 TDM_MAX_MARGIN=0.01 TDM_HUBER_C=1e-3 TDM_FAKE_W_FLOOR=1.0 TDM_ANCHOR_WEIGHT=0 TDM_ANCHOR_HOLD_ITERS=-1 TDM_ANCHOR_FINAL_RATIO=1.0 PRETRAINED_STUDENT_NET_PATH=
case "$ARM" in
  A)       export STUDENT_ROLLOUT=real_noised ;;
  B)       ;;
  C)       export DMD_NOISE=trajectory FAKE_INPUT=final ;;
  D)       export DPDMD=1 VSD_NORM=xt VSD_T_SHIFT=3.0 STUDENT_UPDATE_FREQ=6 FAKE_SCORE_LR=1e-5 ;;
  TDM|ODEINIT_TDM)
           export DMD_TARGET=landing STUDENT_UPDATE_FREQ=2 NET_LR=4e-6 NET_BETAS=0,0.95 FAKE_SCORE_BETAS=0,0.95 \
                  TDM_MIN_T=0.07 ;;
  PDD|ODEINIT_PDD) export PDD=1 STUDENT_UPDATE_FREQ=1 NET_LR=1e-5 ;;
  ODEINIT) export ODE_INIT=1 STUDENT_UPDATE_FREQ=1 NET_LR=2e-6 ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
case "$ARM" in
  ODEINIT_*)
    [ -f "${ODE_INIT_STUDENT:-}" ] || { echo "ARM=$ARM needs ODE_INIT_STUDENT=<student.safetensors exported by ODEINIT>" >&2; exit 2; }
    export PRETRAINED_STUDENT_NET_PATH=$ODE_INIT_STUDENT ;;
esac
# ── Recipe 2: VSD noising τ shift 3 (same τ distribution for student and fake score), Adam β2 0.95,
#    cosine LR decay to 0 (same shape for student and fake score), teacher CFG embedding text 4 / img 1.5.
#    TDM keeps the demo's β=(0,0.95). RECIPE unset = recipe 1.
case "${RECIPE:-1}" in
  1) export LR_SCHEDULE=constant ;;
  2) export VSD_T_SHIFT=3.0 LR_SCHEDULE=cosine CFG_IMG_TEACHER=1.5
     case "$ARM" in TDM|ODEINIT_TDM) ;; *) export NET_BETAS=0.9,0.95 FAKE_SCORE_BETAS=0.9,0.95 ;; esac ;;
  *) echo "unknown RECIPE=$RECIPE (1 | 2)" >&2; exit 2 ;;
esac
for kv in ${ARM_OVERRIDES:-}; do export "$kv"; done
# ── yaml → environment variables (variables that are already set are not overwritten) ──
CONFIG_YAML=${CONFIG_YAML:-$REPO/kokoro_ode/train_ode_dpdmd.yaml}
eval "$("$PY" - "$CONFIG_YAML" <<'PYEOF'
import os, shlex, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
cfg = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in cfg.items()}  # ${VAR} placeholders
for k, v in cfg.items():
    if k in os.environ:
        continue
    v = "" if v is None else ("true" if v is True else "false" if v is False else str(v))
    print(f"export {k}={shlex.quote(v)}")
PYEOF
)"

STAMP=$(date +%m%d_%H%M)
export WANDB_NAME=${WANDB_NAME:-arm_${ARM}_${STAMP}}

RUN_DIR=$FASTGEN_OUTPUT_ROOT/$WANDB_PROJECT/$WANDB_GROUP/$WANDB_NAME
mkdir -p "$RUN_DIR"
export WANDB_DIR=$RUN_DIR
LOG=$RUN_DIR/launch.log

# ── W&B: log to project bagel_step_distill under $WANDB_ENTITY (wandb's default entity if unset); display name = run name · algorithm,
#    group = run name. The local run directory stays <output root>/WANDB_PROJECT/WANDB_GROUP/WANDB_NAME (resuming must find the same
#    directory); the upload target is set separately via WANDB_LOG_* (fastgen/callbacks/wandb.py).
case "$ARM" in
  A)           ALGO="DMD2 (baseline: student input = noised real image)"; ALGO_TAG=DMD2-baseline ;;
  B)           ALGO="DMD2 + ODE rollout"; ALGO_TAG=DMD2-rollout ;;
  C)           ALGO="DMDO (B + trajectory noise)"; ALGO_TAG=DMDO ;;
  D)           ALGO="DP-DMD"; ALGO_TAG=DP-DMD ;;
  TDM)         ALGO="TDM"; ALGO_TAG=TDM ;;
  PDD)         ALGO="PDD"; ALGO_TAG=PDD ;;
  ODEINIT)     ALGO="ODE-init (teacher trajectory regression)"; ALGO_TAG=ODE-init ;;
  ODEINIT_TDM) ALGO="ODE-init → TDM"; ALGO_TAG=TDM ;;
  ODEINIT_PDD) ALGO="ODE-init → PDD"; ALGO_TAG=PDD ;;
esac
RESUMED=""; [[ " $* " == *" trainer.resume=True "* ]] && RESUMED=" · resumed"
RECIPE_LABEL=""; [ "${RECIPE:-1}" = 2 ] && RECIPE_LABEL=" · recipe 2"
FLOOR_LABEL=""; FLOOR_TAG=""
case "$ARM" in TDM|ODEINIT_TDM) [ "$TDM_FAKE_W_FLOOR" != "1.0" ] && [ "$TDM_FAKE_W_FLOOR" != "1" ] && \
  FLOOR_LABEL=" · fake weight floor f=$TDM_FAKE_W_FLOOR" FLOOR_TAG=",tdm-fake-floor-$TDM_FAKE_W_FLOOR" ;; esac
export WANDB_LOG_PROJECT=${WANDB_LOG_PROJECT:-bagel_step_distill} \
       WANDB_LOG_GROUP=${WANDB_LOG_GROUP:-$WANDB_NAME}
export WANDB_LOG_NAME=${WANDB_LOG_NAME:-"$WANDB_NAME · $ALGO$FLOOR_LABEL$RECIPE_LABEL$RESUMED"}
export WANDB_TAGS=${WANDB_TAGS:-"$ALGO_TAG,recipe${RECIPE:-1},ARM-$ARM$FLOOR_TAG"}
export WANDB_NOTES=${WANDB_NOTES:-"Algorithm: $ALGO (ARM=$ARM, recipe ${RECIPE:-1}). Switches are listed in launch.log."}
# API key: a launcher started with env -i does not source ~/.bashrc, so read WANDB_API_KEY from it at runtime into this
# process's environment (never printed or written to disk).
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.bashrc" ]; then
  _k=$(sed -n 's/^export WANDB_API_KEY=//p' "$HOME/.bashrc" | tail -1); _k=${_k//\"/}; _k=${_k//\'/}
  [ -n "$_k" ] && export WANDB_API_KEY=$_k
  unset _k
fi
WANDB_REASON="from yaml"
if ls -d "$RUN_DIR"/wandb/offline-run-* >/dev/null 2>&1 && ! ls -d "$RUN_DIR"/wandb/run-* >/dev/null 2>&1; then
  export WANDB_MODE=offline; WANDB_REASON="run started offline, staying offline on resume (wandb sync after training)"
elif [ "${WANDB_MODE:-online}" = "online" ] && [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$REPO/credentials/wandb_api.txt" ]; then
  export WANDB_MODE=offline; WANDB_REASON="no WANDB_API_KEY (not in ~/.bashrc either)"
fi

{
  echo "=== $(date '+%F %T %Z') ARM=$ARM GPUS=$GPUS NPROC=$NPROC MASTER_PORT=$MASTER_PORT"
  echo "RUN_DIR=$RUN_DIR"
  echo "  ARM_OVERRIDES=${ARM_OVERRIDES:-}"
  echo "  wandb: mode=${WANDB_MODE:-online} ($WANDB_REASON) → ${WANDB_ENTITY}/${WANDB_LOG_PROJECT} group=${WANDB_LOG_GROUP} name=${WANDB_LOG_NAME} tags=${WANDB_TAGS}"
  for k in STUDENT_ROLLOUT ROLLOUT_EXIT FAKE_INPUT DMD_NOISE DMD_TARGET VSD_NORM VSD_T_SHIFT DPDMD DPDMD_DIV_WEIGHT \
           DPDMD_TEACHER_STEPS DPDMD_K_ANCHOR DPDMD_GRAD_STEPS ODE_INIT ODE_INIT_TEACHER_STEPS ODE_INIT_POINTS \
           TDM_MIN_T TDM_HUBER_C TDM_MAX_MARGIN TDM_FAKE_W_FLOOR TDM_ANCHOR_WEIGHT TDM_ANCHOR_HOLD_ITERS TDM_ANCHOR_FINAL_RATIO PDD PDD_GRID_STEPS PDD_SHIFT PDD_SOLVER PDD_L_MIN PDD_L_MAX \
           STUDENT_T_LIST STUDENT_UPDATE_FREQ GUIDANCE_SCALE CFG_IMG_TEACHER GAN_LOSS_WEIGHT_GEN NET_LR FAKE_SCORE_LR \
           NET_BETAS FAKE_SCORE_BETAS LR_SCHEDULE LR_SCHEDULE_ITERS NET_WARMUP_STEPS FAKE_SCORE_WARMUP_STEPS RECIPE OPTIM_FP32_MASTER EMA_FP32_SHADOW COND_KV_PER_ROLE USE_EMA \
           BATCH_SIZE_GLOBAL MAX_ITER SAVE_CKPT_ITER BAGEL_CKPT_PATH PRETRAINED_STUDENT_NET_PATH DATA_YAML I2I_PROB \
           WANDB_MODE WANDB_NAME; do
    printf '  %s=%s\n' "$k" "${!k-<unset>}"
  done
} | tee -a "$LOG"

# ── Memory fence: kills only this arm's process group (two arms may share a node; the limit applies to whole-node anon memory) ──
export GUARD_LIMIT_GIB=${GUARD_LIMIT_GIB:-1100} GUARD_POLL_SEC=${GUARD_POLL_SEC:-3}
PGID_FILE=$RUN_DIR/train.pgid
export GUARD_PGID_FILE=$PGID_FILE GUARD_SENTINEL=$RUN_DIR/mem_guard_tripped
rm -f "$PGID_FILE"

setsid bash -c '
  p=$(ps -o pgid= -p $$); echo "${p// /}" > "'"$PGID_FILE"'"
  exec "'"$PY"'" -m torch.distributed.run --nnodes=1 --nproc_per_node='"$NPROC"' \
       --master_addr=127.0.0.1 --master_port='"$MASTER_PORT"' \
       '"${ENTRY:-train.py}"' --config=fastgen/configs/experiments/Kokoro/config_kokoro_dmd2.py '"$*"'
' >> "$LOG" 2>&1 &
LAUNCH_PID=$!

GUARD_PID=""
if [ "${MEM_GUARD:-1}" != "0" ]; then
  for _ in $(seq 50); do [ -s "$PGID_FILE" ] && break; sleep 0.1; done
  "$PY" scripts/mem_guard.py >> "$RUN_DIR/mem_guard.log" 2>&1 &
  GUARD_PID=$!
fi
PRUNE_PID=""
if [ "${PRUNE_OLD_DCP:-1}" != "0" ]; then
  bash "$REPO/kokoro_ode/prune_old_dcp.sh" "$RUN_DIR/${CKPT_SUBDIR:-checkpoints}" >> "$RUN_DIR/prune.log" 2>&1 &
  PRUNE_PID=$!
fi
echo "launched: torchrun pid $LAUNCH_PID (pgid $(cat "$PGID_FILE" 2>/dev/null)), mem_guard pid ${GUARD_PID:-off}, prune pid ${PRUNE_PID:-off}, log $LOG"

set +e
wait "$LAUNCH_PID"; RC=$?
set -e
[ -n "$GUARD_PID" ] && { kill "$GUARD_PID" 2>/dev/null || true; wait "$GUARD_PID" 2>/dev/null || true; }
[ -n "$PRUNE_PID" ] && { kill "$PRUNE_PID" 2>/dev/null || true; wait "$PRUNE_PID" 2>/dev/null || true; }
[ -f "$GUARD_SENTINEL" ] && echo "=== mem_guard TRIPPED: $(cat "$GUARD_SENTINEL")" | tee -a "$LOG"
echo "=== $(date '+%F %T %Z') exit $RC" | tee -a "$LOG"
exit "$RC"
