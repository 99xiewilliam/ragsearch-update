#!/usr/bin/env bash
set -euo pipefail

# Smoke demo: sample 30 rag-mini-wikipedia QA examples and run the common pipeline once.
#
# Requirements:
# - conda env: ragsearch
# - vLLM/OpenAI-compatible endpoint running, e.g. http://localhost:9000/v1

ROOT="/home/xwh/ragsearch-update"
PY="conda run -n ragsearch --no-capture-output python"

DATASET_DIR="${ROOT}/datasets/rag_mini_wikipedia_30"
PIPELINE_MODE="${PIPELINE_MODE:-smoke}" # smoke|full
OUT_DIR="${OUT_DIR:-${ROOT}/runs/miniwiki_common_30_smoke}"
CFG="${CFG:-${ROOT}/tmp/hotpotqa_common_smoke.yaml}"

LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:9000/v1}"
DEBUG_TRACE_N="${DEBUG_TRACE_N:-0}"

if [[ "${PIPELINE_MODE}" == "full" ]]; then
  OUT_DIR="${OUT_DIR:-${ROOT}/runs/miniwiki_common_30_full}"
  CFG="${ROOT}/tmp/miniwiki_common_full.yaml"
  # Enable all major modules (may download/embed/rerank models on first run).
  cat > "${CFG}" <<YAML
pipeline: common

# Retrieval: hybrid => enable both bm25 + embeddings
bm25_weight: 0.5
retriever_topk: 10

# Chunking + embedding
chunking_enabled: true
chunking_method: token
chunk_size: 256
chunk_overlap: 0
min_chunk_words: 8
embedder_model: "BAAI/bge-m3"

# Rewriter / reranker / pruner
rewriter_enabled: true
reranker_enabled: true
reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_topk: 5
pruner_enabled: true
pruner_mode: select

# Generator (must match your vLLM served model id/path)
generator_model: "/home/xwh/models/Qwen3-4B-Instruct-2507"
generator_max_tokens: 32768
YAML
fi

LOG_FILE="${LOG_FILE:-${OUT_DIR}/run.log}"

mkdir -p "${ROOT}/datasets" "${ROOT}/runs" "${OUT_DIR}"

echo "[1/2] Build rag-mini-wikipedia sample dataset -> ${DATASET_DIR}"
${PY} "${ROOT}/scripts/datasets/make_rag_mini_wikipedia_dataset.py" \
  --out_dir "${DATASET_DIR}" \
  --n_train 30 \
  --n_val 0 \
  --seed 42

echo
echo "[2/2] Run common pipeline (${PIPELINE_MODE}) -> ${OUT_DIR}"
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

