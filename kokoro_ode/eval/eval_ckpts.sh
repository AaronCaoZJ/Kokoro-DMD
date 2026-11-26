#!/bin/bash
# heldout15 × 3 seeds (42 / 100 / 1024): generate one image set for every exported checkpoint of a run
# (exported/<iter>/student|ema.safetensors).
#
#   GPU=0 RUN=<run_dir> bash kokoro_ode/eval/eval_ckpts.sh            # all checkpoints, student and ema
#   GPU=0 RUN=<run_dir> ITERS="0003000" WHICH="ema" bash kokoro_ode/eval/eval_ckpts.sh
#   GPU=0 TEACHER=1 [BASES="teacher init init_s3 teacher_ci15"] bash kokoro_ode/eval/eval_ckpts.sh   # teacher (41 steps, shift 3, cfg 4/2)
#                                                                                     # + untrained student at 4 NFE (init = Wan table, init_s3 = shift-3 table)
# The student's 4-step table / PDD grid shift is read from the run's config.yaml (images use the same table as training).
#
# Output: results/raw/eval_seeds42_100_1024/<run_name>_<iter>_<which>/ (batch_eval's images/, results.json, report.html).
# (task, model) pairs that already have results are skipped (--skip-existing). PDD students are detected by batch_eval from pdd_heads.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PYTHON:-python}
cd "$REPO"
export TZ=Asia/Singapore PYTHONPATH=$REPO${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
OUT=${EVAL_ROOT:-$REPO/results/raw/eval_seeds42_100_1024}
SUITE=${SUITE:-$REPO/kokoro_ode/eval/heldout15_seeds42_100_1024.json}
[ -f "$SUITE" ] || "$PY" kokoro_ode/eval/make_heldout_suite.py --seeds 42,100,1024 --out "$SUITE"
# EVAL_BAGEL_DIR overrides BAGEL_MODEL_DIR; EVAL_TEACHER is the teacher's ema.safetensors.
# TEACHER_TAG / INIT_TAG set the name prefixes of the baseline output folders.
BAGEL_DIR=${EVAL_BAGEL_DIR:-${BAGEL_MODEL_DIR:?set BAGEL_MODEL_DIR or EVAL_BAGEL_DIR}}
T3750=${EVAL_TEACHER:?set EVAL_TEACHER (teacher ema.safetensors)}
TEACHER_TAG=${TEACHER_TAG:-teacher}
INIT_TAG=${INIT_TAG:-init_student}
run_one() {  # $1=ckpt $2=label $3=outdir, rest = extra args
  local ck=$1 label=$2 od=$3; shift 3
  mkdir -p "$od"
  "$PY" scripts/inference/batch_eval.py --model_dir "$BAGEL_DIR" --ckpts "$ck::$label" --test-json "$SUITE" \
      --out_dir "$od" --skip-existing "$@" >> "$od.log" 2>&1
  tail -3 "$od.log"
}
if [ "${TEACHER:-0}" = "1" ]; then
  # Teacher baseline: 41 steps (num_timesteps 42), static shift 3, cfg_text 4;
  # cfg_img stays at 2 (= CFG_IMG_TEACHER, the teacher's image CFG during distillation)
  TL=$("$PY" -c "import torch;s=torch.linspace(1,0,42,dtype=torch.float64);print(','.join(f'{v:.6f}' for v in (3*s/(1+2*s)).tolist()))")
  # BASES selects which baselines to run (e.g. split them across two GPUs); default: teacher init init_s3
  [[ " ${BASES:-teacher init init_s3} " == *" teacher "* ]] && \
    run_one "$T3750" teacher_41s "$OUT/teacher_41s" --role teacher --t_list "$TL" --cfg_scale 4 --cfg_img_scale 2
  # Alternative teacher setting: 40 steps (num_timesteps 41), shift 3, cfg-embed text 4 / img 1.5; only runs when
  # teacher_ci15 is listed in BASES
  TL41=$("$PY" -c "import torch;s=torch.linspace(1,0,41,dtype=torch.float64);print(','.join(f'{v:.6f}' for v in (3*s/(1+2*s)).tolist()))")
  [[ " ${BASES:-teacher init init_s3} " == *" teacher_ci15 "* ]] && \
    run_one "$T3750" ${TEACHER_TAG}_40s_ci15 "$OUT/${TEACHER_TAG}_40s_ci15" --role teacher --t_list "$TL41" --cfg_scale ${TEACHER_CFG:-4} --cfg_img_scale 1.5
  [[ " ${BASES:-teacher init init_s3} " == *" init "* ]] && run_one "$T3750" init_student_4nfe "$OUT/init_student_4nfe" --role student \
      --t_list "0.999,0.937,0.833,0.624,0.0"
  [[ " ${BASES:-teacher init init_s3} " == *" init_s3 "* ]] && run_one "$T3750" ${INIT_TAG}_4nfe_s3 "$OUT/${INIT_TAG}_4nfe_s3" --role student \
      --t_list "0.999,0.9,0.75,0.5,0.0"
  exit 0
fi
RUN=${RUN:?set RUN=<run_dir>}
NAME=$(basename "$RUN")
read -r RUN_TL RUN_PDD_SHIFT < <("$PY" - "$RUN/config.yaml" <<'PYEOF'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["model"]
print(",".join(str(float(v)) for v in m["sample_t_cfg"]["t_list"]), float(m.get("pdd_shift", 5.0)))
PYEOF
)
[ -n "${RUN_TL:-}" ] || { echo "cannot read t_list from $RUN/config.yaml" >&2; exit 1; }
echo "$NAME: t_list=$RUN_TL pdd_shift=$RUN_PDD_SHIFT"
for d in "$RUN"/checkpoints/exported/*/; do
  it=$(basename "$d")
  [ -n "${ITERS:-}" ] && [[ " $ITERS " != *" $it "* ]] && continue
  [ -f "$d/step.txt" ] || { echo "skip $it (export incomplete)"; continue; }
  for w in ${WHICH:-student ema}; do
    [ -f "$d/$w.safetensors" ] || continue
    run_one "$d/$w.safetensors" "${NAME}_${it}_${w}" "$OUT/${NAME}_${it}_${w}" --role student \
        --t_list "$RUN_TL" --pdd_shift "$RUN_PDD_SHIFT"
  done
done
