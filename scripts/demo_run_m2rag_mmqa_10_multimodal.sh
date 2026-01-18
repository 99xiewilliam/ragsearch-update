#!/usr/bin/env bash
set -euo pipefail

# Smoke demo: sample 10 examples from whalezzz/M2RAG (mmqa/test_data.jsonl) and run the multimodal pipeline once.
#
# Requirements:
# - conda env: ragsearch
# - vLLM/OpenAI-compatible endpoint running, e.g. http://localhost:9000/v1
#
# Optional envs:
#   N=10
#   DATASET_DIR=/abs/path/to/out_dataset
#   OUT_DIR=/abs/path/to/out_run
#   LLM_BASE_URL=http://localhost:9000/v1
#   GENERATOR_MODEL=<served model id from /v1/models>
#   DEBUG_TRACE_N=3
#   ONLY_TEXT=1  (force text-only sampling; useful if you don't have a vision model served)

ROOT="/home/xwh/ragsearch-update"
PY="conda run -n ragsearch --no-capture-output python"

N="${N:-10}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/datasets/m2rag_mmqa_test10}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/m2rag_mmqa_test10_mm_smoke}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/run.log}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:9000/v1}"
ONLY_TEXT="${ONLY_TEXT:-0}"
DEBUG_TRACE_N="${DEBUG_TRACE_N:-1}"

mkdir -p "${ROOT}/datasets" "${ROOT}/runs" "${OUT_DIR}"

# If user didn't set GENERATOR_MODEL, auto-pick the first model served by the endpoint.
if [[ -z "${GENERATOR_MODEL:-}" ]]; then
  GENERATOR_MODEL="$(
    ${PY} - <<'PY'
import os
import requests

base = os.environ.get("LLM_BASE_URL", "http://localhost:9000/v1").rstrip("/")
r = requests.get(base + "/models", timeout=10)
r.raise_for_status()
obj = r.json() or {}
models = obj.get("data") or []
print((models[0] or {}).get("id", "") if models else "")
PY
  )"
fi
if [[ -z "${GENERATOR_MODEL}" ]]; then
  echo "[ERR] Could not infer GENERATOR_MODEL from ${LLM_BASE_URL}/models. Please export GENERATOR_MODEL." >&2
  exit 1
fi

# Heuristic: if served model doesn't look vision-capable, default to only-text sampling.
if [[ "${ONLY_TEXT}" == "0" ]]; then
  if [[ "${GENERATOR_MODEL,,}" != *"vl"* && "${GENERATOR_MODEL,,}" != *"vision"* ]]; then
    ONLY_TEXT="1"
  fi
fi

EXTRA_DATASET_ARGS=()
if [[ "${ONLY_TEXT}" == "1" ]]; then
  EXTRA_DATASET_ARGS+=(--only_text)
fi

echo "[1/2] Build M2RAG-MMQA sample dataset (n=${N}) -> ${DATASET_DIR}"
${PY} "${ROOT}/scripts/datasets/make_m2rag_mmqa_sample_dataset.py" \
  --out_dir "${DATASET_DIR}" \
  --n "${N}" \
  "${EXTRA_DATASET_ARGS[@]}"

CFG_PATH="${ROOT}/tmp/m2rag_mmqa_mm_smoke.yaml"
cat > "${CFG_PATH}" <<YAML
pipeline: multimodal

# Pure BM25 over pos_text/caption for stable smoke (no embeddings download).
bm25_weight: 1.0
retriever_topk: 3
chunking_enabled: false
reranker_enabled: false
pruner_enabled: false
rewriter_enabled: false

# Use a vision-capable generator (served by your vLLM endpoint).
generator_model: "${GENERATOR_MODEL}"
generator_max_tokens: 256

# Make sure multimodal pipeline augments text with metadata (caption/ocr/etc.)
multimodal_metadata_enabled: true
YAML

echo
echo "[2/2] Run multimodal pipeline -> ${OUT_DIR}"
echo "  CFG=${CFG_PATH}"
echo "  log -> ${LOG_FILE}"

conda run -n ragsearch --no-capture-output env PYTHONPATH="${ROOT}" \
  python -m autorag_offline_search.cli \
    --dataset_dir "${DATASET_DIR}" \
    --out_dir "${OUT_DIR}" \
    --algo random \
    --train_trials 1 \
    --metrics_weights "qa_f1:1,em:1" \
    --llm_base_url "${LLM_BASE_URL}" \
    --pipeline multimodal \
    --config "${CFG_PATH}" \
    --verbose \
    --module_logs \
    --debug_trace_n "${DEBUG_TRACE_N}" \
    --dump_val_generations \
    --dump_val_limit 20 \
  2>&1 | tee "${LOG_FILE}"

echo
echo "[DONE] outputs:"
echo "  ${OUT_DIR}/results.json"
echo "  ${OUT_DIR}/val_predictions_random.jsonl"

