#!/usr/bin/env bash
set -euo pipefail

# Smoke demo: sample 20 HotPotQA examples and run the common pipeline once.
#
# Requirements:
# - conda env: ragsearch
# - vLLM/OpenAI-compatible endpoint running, e.g. http://localhost:9000/v1

ROOT="/home/xwh/ragsearch-update"
PY="conda run -n ragsearch --no-capture-output python"

DATASET_DIR="${ROOT}/datasets/hotpotqa_distractor_20"
OUT_DIR="${ROOT}/runs/hotpotqa_common_20_smoke"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/run.log}"
CFG="${ROOT}/tmp/hotpotqa_common_smoke.yaml"

LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:9000/v1}"
DEBUG_TRACE_N="${DEBUG_TRACE_N:-0}"

mkdir -p "${ROOT}/datasets" "${ROOT}/runs" "${OUT_DIR}"

echo "[1/2] Build HotPotQA sample dataset -> ${DATASET_DIR}"
${PY} "${ROOT}/scripts/datasets/make_hotpotqa_sample_dataset.py" \
  --out_dir "${DATASET_DIR}" \
  --no_split \
  --subset distractor \
  --split validation \
  --n_train 20 \
  --n_val 0 \
  --seed 42

echo
echo "[2/2] Run common pipeline (BM25-only) -> ${OUT_DIR}"
echo "  log -> ${LOG_FILE}"
conda run -n ragsearch --no-capture-output env PYTHONPATH="${ROOT}" \
  python -m autorag_offline_search.cli \
    --dataset_dir "${DATASET_DIR}" \
    --out_dir "${OUT_DIR}" \
    --algo random \
    --train_trials 1 \
    --metrics_weights "qa_f1:1,em:1" \
    --llm_base_url "${LLM_BASE_URL}" \
    --pipeline common \
    --config "${CFG}" \
    --verbose \
    --module_logs \
    --debug_trace_n "${DEBUG_TRACE_N}" \
    --dump_val_generations \
    --dump_val_limit 20 \
  2>&1 | tee "${LOG_FILE}"

echo
echo "[DONE] See outputs under: ${OUT_DIR}"

