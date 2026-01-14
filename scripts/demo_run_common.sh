#!/usr/bin/env bash
set -euo pipefail

# Demo: run common pipeline end-to-end
#
# Usage:
#   bash scripts/demo_run_common.sh
#
# Optional envs:
#   DATASET_DIR=/abs/path/to/dataset
#   OUT_DIR=/abs/path/to/out
#   LLM_BASE_URL=http://localhost:9000/v1
#   GENERATOR_MODEL=/home/xwh/models/Qwen3-4B-Instruct-2507   (or any model served by your vLLM)
#   EMBEDDER_MODEL=BAAI/bge-m3
#   METRICS_WEIGHTS="em:1,rougeL:1,accuracy:1"

export PYTHONPATH="/home/xwh/ragsearch-update:${PYTHONPATH:-}"

# Reduce noisy chroma telemetry logs
export ANONYMIZED_TELEMETRY="False"
export CHROMA_TELEMETRY="False"

DATASET_DIR="${DATASET_DIR:-/home/xwh/ragsearch-update/tmp/mm_images_dataset}"
OUT_DIR="${OUT_DIR:-/tmp/demo_common_out}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:9000/v1}"
GENERATOR_MODEL="${GENERATOR_MODEL:-/home/xwh/models/Qwen3-VL-4B-Instruct}"
EMBEDDER_MODEL="${EMBEDDER_MODEL:-BAAI/bge-m3}"
METRICS_WEIGHTS="${METRICS_WEIGHTS:-em:1,rougeL:1,accuracy:1}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[ERR] DATASET_DIR not found: ${DATASET_DIR}" >&2
  exit 1
fi

CFG_PATH="/tmp/demo_common.yaml"
cat > "${CFG_PATH}" <<YAML
base:
  pipeline: common
  llm_base_url: ${LLM_BASE_URL}
  module_logs: true
  # keep demo stable (avoid extra calls/models)
  rewriter_enabled: false
  pruner_enabled: false
  reranker_enabled: false

  # models
  embedder_model: ${EMBEDDER_MODEL}
  generator_model: ${GENERATOR_MODEL}
  generator_max_tokens: 128

space:
  # single trial (no search), just to show how the system runs
  embedding_enabled: [true]
  retriever: [cosine]
  retriever_topk: [3]
  chunking_enabled: [false]
YAML

mkdir -p "${OUT_DIR}"

echo "[RUN] common pipeline"
echo "  DATASET_DIR=${DATASET_DIR}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  LLM_BASE_URL=${LLM_BASE_URL}"
echo "  CFG=${CFG_PATH}"
echo

conda run -n ragsearch --no-capture-output \
  python -m autorag_offline_search.cli \
    --dataset_dir "${DATASET_DIR}" \
    --out_dir "${OUT_DIR}" \
    --train_trials 1 \
    --config "${CFG_PATH}" \
    --metrics_weights "${METRICS_WEIGHTS}" \
    --llm_base_url "${LLM_BASE_URL}" \
    --algo scripts.my_algo:MyAlgo \
    --verbose \
    --module_logs

echo
echo "[DONE] outputs:"
echo "  ${OUT_DIR}/results.json"
echo "  ${OUT_DIR}/summary.tsv"

