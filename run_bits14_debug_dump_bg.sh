#!/usr/bin/env bash
set -euo pipefail

# Background runner for bits=14 (2^14) across 3 datasets with generation dumps.
#
# Requires:
# - conda env: ragrl
# - PYTHONPATH=/home/xwh/autorag_offline_search
# - vLLM/OpenAI-compatible endpoint at LLM_BASE_URL (default http://127.0.0.1:8001/v1)
#
# Outputs:
# - $OUT/run.log
# - $OUT/sweep.json, $OUT/sweep.tsv
# - Per (dataset,bits14) directory:
#     - summary.tsv, results.json
#     - val_predictions_<algo>.jsonl  (only best config on validation; includes trace if supported)

OUT="${OUT:-/home/xwh/autorag_offline_search_runs/bits14_debug_dump_seed42_$(date +%m%d_%H%M%S)}"
DATASETS_ROOT="${DATASETS_ROOT:-/home/xwh/three_dataset_sample_split}"
ALGO="${ALGO:-greedy,tpe,grpo,grpo_a2}"
SEED="${SEED:-42}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
METRICS_WEIGHTS="${METRICS_WEIGHTS:-rougeL:0.3,bertscore_f1:0.3,chrf:0.4}"

# sweep budget (matches your earlier setup)
AUTO_TRIALS_MODE="${AUTO_TRIALS_MODE:-linear}"
AUTO_TRIALS_SCALE="${AUTO_TRIALS_SCALE:-7.0}"
AUTO_TRIALS_MIN="${AUTO_TRIALS_MIN:-15}"
AUTO_TRIALS_MAX="${AUTO_TRIALS_MAX:-100}"

RAG_PLUGIN="${RAG_PLUGIN:-autorag_offline_search.plugins.component_rag:ComponentRag}"

mkdir -p "$OUT"

(
  exec >"$OUT/run.log" 2>&1
  echo "[start] $(date -Is)"
  echo "OUT=$OUT"
  echo "DATASETS_ROOT=$DATASETS_ROOT"
  echo "ALGO=$ALGO"
  echo "SEED=$SEED"
  echo "LLM_BASE_URL=$LLM_BASE_URL"
  echo "METRICS_WEIGHTS=$METRICS_WEIGHTS"
  echo "AUTO_TRIALS_MODE=$AUTO_TRIALS_MODE AUTO_TRIALS_SCALE=$AUTO_TRIALS_SCALE AUTO_TRIALS_MIN=$AUTO_TRIALS_MIN AUTO_TRIALS_MAX=$AUTO_TRIALS_MAX"
  echo "RAG_PLUGIN=$RAG_PLUGIN"
  echo "----"

  # Avoid proxy hijacking local vLLM calls (your Cursor settings has http.proxy enabled).
  export NO_PROXY=127.0.0.1,localhost
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

  conda run -n ragrl env PYTHONPATH=/home/xwh/autorag_offline_search \
    python -m autorag_offline_search.cli \
      --all_datasets_root "$DATASETS_ROOT" \
      --sweep_bits 14 \
      --algo "$ALGO" \
      --rag_plugin "$RAG_PLUGIN" \
      --metrics_weights "$METRICS_WEIGHTS" \
      --auto_trials_mode "$AUTO_TRIALS_MODE" --auto_trials_scale "$AUTO_TRIALS_SCALE" --auto_trials_min "$AUTO_TRIALS_MIN" --auto_trials_max "$AUTO_TRIALS_MAX" \
      --seed "$SEED" \
      --out_dir "$OUT" \
      --llm_base_url "$LLM_BASE_URL" \
      --dump_val_generations

  echo "----"
  echo "[done] $(date -Is)"
) &

PID=$!
echo "$PID" >"$OUT/pid.txt"
echo "已后台启动 PID=$PID"
echo "日志: $OUT/run.log"
echo "完成后可用: python /home/xwh/autorag_offline_search/analyze_bits14_runs.py --sweep_out_dir $OUT"










