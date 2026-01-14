#!/usr/bin/env bash
set -euo pipefail

# Demo: run multimodal pipeline end-to-end (image-aware retrieval + vision generation)
#
# Usage:
#   bash scripts/demo_run_multimodal.sh
#
# Optional envs:
#   DATASET_DIR=/abs/path/to/dataset (must contain image_path in doc.metadata for best effect)
#   OUT_DIR=/abs/path/to/out
#   LLM_BASE_URL=http://localhost:9000/v1
#   GENERATOR_MODEL=/home/xwh/models/Qwen3-VL-4B-Instruct
#   MM_EMBEDDER_MODEL=Qwen/Qwen3-VL-Embedding-2B   (or openai/clip-vit-base-patch32)
#   MM_RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-2B
#   METRICS_WEIGHTS="em:1,rougeL:1,accuracy:1"

export PYTHONPATH="/home/xwh/ragsearch-update:${PYTHONPATH:-}"

# Reduce noisy chroma telemetry logs
export ANONYMIZED_TELEMETRY="False"
export CHROMA_TELEMETRY="False"

DATASET_DIR="${DATASET_DIR:-/home/xwh/ragsearch-update/tmp/mm_images_dataset}"
OUT_DIR="${OUT_DIR:-/tmp/demo_multimodal_out}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:9000/v1}"
GENERATOR_MODEL="${GENERATOR_MODEL:-/home/xwh/models/Qwen3-VL-4B-Instruct}"
MM_EMBEDDER_MODEL="${MM_EMBEDDER_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
MM_RERANKER_MODEL="${MM_RERANKER_MODEL:-Qwen/Qwen3-VL-Reranker-2B}"
METRICS_WEIGHTS="${METRICS_WEIGHTS:-em:1,rougeL:1,accuracy:1}"

# If user didn't provide a local Qwen3-VL-Embedding directory, prefer CLIP to avoid missing helper script.
if [[ "${MM_EMBEDDER_MODEL}" == Qwen/Qwen3-VL-Embedding-* ]]; then
  if [[ -f "/home/xwh/models/Qwen3-VL-Embedding-2B/scripts/qwen3_vl_embedding.py" ]]; then
    MM_EMBEDDER_MODEL="/home/xwh/models/Qwen3-VL-Embedding-2B"
  elif [[ -f "/home/xwh/models/hf/Qwen--Qwen3-VL-Embedding-2B/scripts/qwen3_vl_embedding.py" ]]; then
    MM_EMBEDDER_MODEL="/home/xwh/models/hf/Qwen--Qwen3-VL-Embedding-2B"
  else
    MM_EMBEDDER_MODEL="openai/clip-vit-base-patch32"
  fi
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[ERR] DATASET_DIR not found: ${DATASET_DIR}" >&2
  exit 1
fi

CFG_PATH="/home/xwh/ragsearch-update/tmp/mm_cfg_space.yaml"
cat > "${CFG_PATH}" <<YAML
base:
  pipeline: multimodal
  llm_base_url: ${LLM_BASE_URL}
  module_logs: true

  # keep demo stable; you can turn these on after confirming models are cached
  rewriter_enabled: false
  pruner_enabled: false

  # models
  embedder_model: ${MM_EMBEDDER_MODEL}
  reranker_model: ${MM_RERANKER_MODEL}
  generator_model: ${GENERATOR_MODEL}
  generator_max_tokens: 128

space:
  # Keep embedding on (cosine/hybrid needs it)
  embedding_enabled: [true]

  # Make the first 5 trials meaningfully different:
  # - token chunking changes number/shape of chunks
  # - retriever/hybrid_alpha changes retrieval behavior
  # - retriever_topk changes context size
  chunking_enabled: [true]
  chunking_method: [token]
  chunk_size: [64, 128, 256]
  chunk_overlap: [0, 16]
  # Chinese strings may look like 1 'word' with whitespace-split; keep it permissive.
  min_chunk_words: [1]

  retriever: [cosine, hybrid]
  hybrid_alpha: [0.3, 0.5]
  retriever_topk: [1, 3, 5]

  # reranker can be toggled later (keep off for stable demo)
  reranker_enabled: [false]
YAML

mkdir -p "${OUT_DIR}"

echo "[RUN] multimodal pipeline"
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

