#!/bin/bash
# Two-stage pipeline: ODE-init → TDM / PDD.
#
#   PIPE=TDM GPUS=0,1,2,3 MASTER_PORT=29610 ODE_INIT_ITERS=2000 MAX_ITER=3000 bash kokoro_ode/run_odeinit_pipeline.sh
#   PIPE=PDD GPUS=4,5,6,7 MASTER_PORT=29620 ODE_INIT_STUDENT=<existing student.safetensors> bash kokoro_ode/run_odeinit_pipeline.sh
#
# Stage 1, ARM=ODEINIT, runs ODE_INIT_ITERS iterations (the loop runs to ODE_INIT_ITERS−1, then saves and exports
#   <run>/checkpoints/exported/<ODE_INIT_ITERS>/student.safetensors). If ODE_INIT_STUDENT is given, stage 1 is skipped
#   (one ODE-init result can feed both TDM and PDD). ODE_INIT_USE_EMA=1 uses the exported ema.safetensors instead.
# Stage 2, ARM=ODEINIT_<PIPE>, starts the student from the stage-1 result (the fake score still starts from the teacher,
#   as in CausVid); MAX_ITER etc. come from the environment / yaml as usual.
# All other environment variables are passed through to launch_arm.sh for both stages.

set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
PIPE=${PIPE:?set PIPE=TDM|PDD}
case "$PIPE" in TDM|PDD) ;; *) echo "PIPE must be TDM or PDD" >&2; exit 2 ;; esac
STAMP=$(date +%m%d_%H%M)

if [ -z "${ODE_INIT_STUDENT:-}" ]; then
  ITERS=${ODE_INIT_ITERS:?set ODE_INIT_ITERS, the stage-1 iteration count, or pass ODE_INIT_STUDENT directly}
  NAME1=${ODE_INIT_WANDB_NAME:-odeinit_for_${PIPE}_${STAMP}}
  env ARM=ODEINIT WANDB_NAME="$NAME1" MAX_ITER="$ITERS" SAVE_CKPT_ITER=$((ITERS + 1)) EXPORT_SAFETENSORS=1 \
      bash "$REPO/kokoro_ode/launch_arm.sh"
  RUN1=$(${PYTHON:-python} - "${CONFIG_YAML:-$REPO/kokoro_ode/train_ode_dpdmd.yaml}" "$NAME1" <<'PY'
import os, sys, yaml
c = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
g = lambda k: os.environ.get(k) or os.path.expandvars(str(c[k]))          # as in launch_arm.sh: environment variables take precedence over the yaml
print(f"{g('FASTGEN_OUTPUT_ROOT')}/{g('WANDB_PROJECT')}/{g('WANDB_GROUP')}/{sys.argv[2]}")
PY
)
  FILE=student.safetensors; [ "${ODE_INIT_USE_EMA:-0}" = "1" ] && FILE=ema.safetensors
  ODE_INIT_STUDENT=$RUN1/${CKPT_SUBDIR:-checkpoints}/exported/$(printf '%07d' "$ITERS")/$FILE
  [ -f "$ODE_INIT_STUDENT" ] || { echo "stage 1 did not produce $ODE_INIT_STUDENT" >&2; exit 1; }
  echo "=== stage 1 done: $ODE_INIT_STUDENT"
fi

env ARM=ODEINIT_$PIPE ODE_INIT_STUDENT="$ODE_INIT_STUDENT" \
    WANDB_NAME="${WANDB_NAME:-odeinit_${PIPE}_${STAMP}}" bash "$REPO/kokoro_ode/launch_arm.sh" "$@"
