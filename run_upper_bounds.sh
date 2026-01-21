#!/bin/bash
#
# ==============================================================================
# Upper Bound 专用脚本
# - HotPotQA / MiniWiki: 使用 posthoc_bm25（近似上界，cheating，会更慢但更像 upper bound）
# - M2RAG: 使用 single_doc（按 doc_<qid>/m2rag_id 对齐；可带图片）
# ==============================================================================

set -euo pipefail

PROJ_ROOT="/home/xwh/ragsearch-update"
export PYTHONPATH="$PROJ_ROOT"
export PYTHONUNBUFFERED=1

mkdir -p "$PROJ_ROOT/runs/logs"

# 你可以通过环境变量覆盖这些默认值
LLM_URL_TEXT="${LLM_URL_TEXT:-http://localhost:9001/v1}"
LLM_URL_MM="${LLM_URL_MM:-http://localhost:9000/v1}"

MODEL_TEXT="${MODEL_TEXT:-/home/xwh/models/Qwen3-4B-Instruct-2507}"
MODEL_MM="${MODEL_MM:-/home/xwh/models/Qwen3-VL-4B-Instruct}"

ORACLE_K_TEXT="${ORACLE_K_TEXT:-10}"
ORACLE_K_HOTPOT="${ORACLE_K_HOTPOT:-$ORACLE_K_TEXT}"
ORACLE_K_MINIWIKI="${ORACLE_K_MINIWIKI:-$ORACLE_K_TEXT}"

echo "======================================================================"
echo "开始执行 Upper Bound（仅 oracle_upper_bound.py）"
echo "======================================================================"
echo "LLM_URL_TEXT=$LLM_URL_TEXT"
echo "LLM_URL_MM  =$LLM_URL_MM"
echo "MODEL_TEXT  =$MODEL_TEXT"
echo "MODEL_MM    =$MODEL_MM"
echo ""

echo "[1/3] HotPotQA Upper Bound (posthoc_bm25, k=$ORACLE_K_HOTPOT)..."
python scripts/oracle_upper_bound.py \
  --dataset_dir datasets/hotpotqa_sample \
  --llm_base_url "$LLM_URL_TEXT" \
  --generator_model "$MODEL_TEXT" \
  --metrics_weights "em:1,qa_f1:1" \
  --oracle_mode posthoc_bm25 \
  --oracle_k "$ORACLE_K_HOTPOT" \
  > runs/logs/oracle_hotpot.log 2>&1

echo "[2/3] MiniWiki Upper Bound (posthoc_bm25, k=$ORACLE_K_MINIWIKI)..."
python scripts/oracle_upper_bound.py \
  --dataset_dir datasets/miniwiki_sample \
  --llm_base_url "$LLM_URL_TEXT" \
  --generator_model "$MODEL_TEXT" \
  --metrics_weights "em:1,qa_f1:1" \
  --oracle_mode posthoc_bm25 \
  --oracle_k "$ORACLE_K_MINIWIKI" \
  > runs/logs/oracle_miniwiki.log 2>&1

echo "[3/3] M2RAG Upper Bound (single_doc + images)..."
python scripts/oracle_upper_bound.py \
  --dataset_dir datasets/m2rag_mmqa_100 \
  --llm_base_url "$LLM_URL_MM" \
  --generator_model "$MODEL_MM" \
  --metrics_weights "m2rag_overall:1" \
  --oracle_mode single_doc \
  --allow_images \
  > runs/logs/oracle_m2rag.log 2>&1

echo ""
echo "Upper Bound 已完成。日志在：runs/logs/oracle_{hotpot|miniwiki|m2rag}.log"

