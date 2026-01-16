#!/bin/bash
# Run TPE test on HotPotQA common pipeline and extract best configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="${PROJECT_ROOT}/runs/hotpotqa_common_tpe_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "TPE Test on HotPotQA Common Pipeline"
echo "=========================================="
echo "Output directory: $OUT_DIR"
echo ""

# Run TPE search
HF_HOME="/home/xwh/models/hf_cache" \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
conda run -n ragsearch --no-capture-output env PYTHONPATH="$PROJECT_ROOT" \
python -m autorag_offline_search.cli \
  --dataset_dir "$PROJECT_ROOT/datasets/hotpotqa_distractor_20" \
  --out_dir "$OUT_DIR" \
  --algo tpe \
  --train_trials 100 \
  --metrics_weights "qa_f1:1,em:1" \
  --llm_base_url "http://localhost:9000/v1" \
  --pipeline common \
  --config "$PROJECT_ROOT/tmp/hotpotqa_common_tpe_test.yaml" \
  --tpe_warmup 20 --tpe_patience 15 --tpe_min_delta 0.001 \
  --debug_trace_n 1 --debug_trace_every 20 --debug_trace_split train --debug_trace_max_chars 240 \
  --module_logs \
  --verbose --log_every 5 \
  --dump_val_generations --dump_val_limit 20 \
  2>&1 | tee "$OUT_DIR/run.log"

echo ""
echo "=========================================="
echo "Extracting Best Configuration"
echo "=========================================="

# Extract best configuration
conda run -n ragsearch --no-capture-output python3 "$PROJECT_ROOT/scripts/extract_best_config.py" "$OUT_DIR/results.json"

echo ""
echo "✅ Test completed! Results in: $OUT_DIR"
