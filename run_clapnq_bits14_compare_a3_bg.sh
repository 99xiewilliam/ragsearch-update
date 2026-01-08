#!/usr/bin/env bash
set -euo pipefail

# Compare: greedy vs tpe vs grpo_a2 vs grpo_a3 on clapnq bits=14
# - Runs in background (nohup)
# - Dumps validation generations (jsonl) for quick debugging
#
# Usage:
#   bash /home/xwh/autorag_offline_search/run_clapnq_bits14_compare_a3_bg.sh

OUT="${OUT:-/home/xwh/autorag_offline_search_runs/clapnq_bits14_compare_a3_seed42_$(date +%m%d_%H%M%S)}"
DATASET_DIR="${DATASET_DIR:-/home/xwh/three_dataset_sample_split/clapnq}"
SEED="${SEED:-42}"
SPACE_BITS="${SPACE_BITS:-14}"
TRAIN_TRIALS="${TRAIN_TRIALS:-80}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
METRICS_WEIGHTS="${METRICS_WEIGHTS:-rougeL:0.34,bertscore_f1:0.33,meteor:0.33}"
DUMP_VAL_LIMIT="${DUMP_VAL_LIMIT:-200}"

mkdir -p "$OUT"

nohup bash -lc "
set -euo pipefail
mkdir -p '$OUT'
exec > '$OUT/run.log' 2>&1

echo '[start]' \$(date -Iseconds)
echo OUT='$OUT'
echo DATASET_DIR='$DATASET_DIR'
echo ALGO=greedy,tpe,grpo_a2,grpo_a3
echo SPACE_BITS='$SPACE_BITS'
echo TRAIN_TRIALS='$TRAIN_TRIALS'
echo SEED='$SEED'
echo LLM_BASE_URL='$LLM_BASE_URL'
echo METRICS_WEIGHTS='$METRICS_WEIGHTS'
echo DUMP_VAL_LIMIT='$DUMP_VAL_LIMIT'
echo '----'

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export NO_PROXY=127.0.0.1,localhost

export PYTHONPATH=/home/xwh/autorag_offline_search
conda run -n ragrl --no-capture-output python3 -m autorag_offline_search.cli \
  --dataset_dir '$DATASET_DIR' \
  --space_bits '$SPACE_BITS' \
  --train_trials '$TRAIN_TRIALS' \
  --algo 'greedy,tpe,grpo_a2,grpo_a3' \
  --metrics_weights '$METRICS_WEIGHTS' \
  --seed '$SEED' \
  --out_dir '$OUT' \
  --llm_base_url '$LLM_BASE_URL' \
  --dump_val_generations \
  --dump_val_limit '$DUMP_VAL_LIMIT'

echo '[done]' \$(date -Iseconds)
" >/dev/null 2>&1 &

echo $! > "$OUT/pid.txt" || true
echo "started: OUT=$OUT"
echo "log: $OUT/run.log"

