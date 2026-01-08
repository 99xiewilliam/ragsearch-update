#!/usr/bin/env bash
set -euo pipefail

# Compare algos on clapnq with bits=14 (2^14 search space) in the background.
# Default metrics weights include METEOR (so it will be computed).
#
# Usage:
#   chmod +x /home/xwh/autorag_offline_search/run_clapnq_bits14_compare_bg.sh
#   /home/xwh/autorag_offline_search/run_clapnq_bits14_compare_bg.sh
#
# Optional env overrides:
#   OUT=... TRAIN_TRIALS=80 METRICS_WEIGHTS="rougeL:0.4,bertscore_f1:0.3,meteor:0.3" LLM_BASE_URL=...

OUT="${OUT:-/home/xwh/autorag_offline_search_runs/clapnq_bits14_compare_seed42_$(date +%m%d_%H%M%S)}"
DATASET_DIR="${DATASET_DIR:-/home/xwh/three_dataset_sample_split/clapnq}"
ALGO="${ALGO:-greedy,tpe,grpo,grpo_a2}"
SPACE_BITS="${SPACE_BITS:-14}"
TRAIN_TRIALS="${TRAIN_TRIALS:-80}"
SEED="${SEED:-42}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"

# By default include METEOR so it will be computed (otherwise it stays 0.0 as a placeholder).
METRICS_WEIGHTS="${METRICS_WEIGHTS:-rougeL:0.34,bertscore_f1:0.33,meteor:0.33}"
RAG_PLUGIN="${RAG_PLUGIN:-autorag_offline_search.plugins.component_rag:ComponentRag}"

# Dump some validation generations for debugging (set 0 for full dump).
DUMP_VAL_LIMIT="${DUMP_VAL_LIMIT:-200}"

mkdir -p "$OUT"

(
  exec >"$OUT/run.log" 2>&1
  echo "[start] $(date -Is)"
  echo "OUT=$OUT"
  echo "DATASET_DIR=$DATASET_DIR"
  echo "ALGO=$ALGO"
  echo "SPACE_BITS=$SPACE_BITS"
  echo "TRAIN_TRIALS=$TRAIN_TRIALS"
  echo "SEED=$SEED"
  echo "LLM_BASE_URL=$LLM_BASE_URL"
  echo "METRICS_WEIGHTS=$METRICS_WEIGHTS"
  echo "DUMP_VAL_LIMIT=$DUMP_VAL_LIMIT"
  echo "----"

  export NO_PROXY=127.0.0.1,localhost
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

  conda run -n ragrl env PYTHONPATH=/home/xwh/autorag_offline_search \
    python -m autorag_offline_search.cli \
      --dataset_dir "$DATASET_DIR" \
      --algo "$ALGO" \
      --space_bits "$SPACE_BITS" \
      --train_trials "$TRAIN_TRIALS" \
      --rag_plugin "$RAG_PLUGIN" \
      --metrics_weights "$METRICS_WEIGHTS" \
      --seed "$SEED" \
      --out_dir "$OUT" \
      --llm_base_url "$LLM_BASE_URL" \
      --dump_val_generations --dump_val_limit "$DUMP_VAL_LIMIT"

  echo "----"
  echo "[done] $(date -Is)"
) &

PID=$!
echo "$PID" >"$OUT/pid.txt"
echo "已后台启动 PID=$PID"
echo "日志: $OUT/run.log"
echo "结果: $OUT/results.json"
echo "生成dump: $OUT/val_predictions_<algo>.jsonl"


