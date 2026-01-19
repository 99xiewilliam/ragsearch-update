#!/bin/bash

# ==============================================================================
# RAG Pipeline 全自动化实验脚本
# 包含：1. 各数据集 Upper Bound (天花板) 计算
#       2. Random/TPE/GRPO 算法基准对比 (10 Trials)
# ==============================================================================

set -e  # 出错即停止

# 设置项目根目录
PROJ_ROOT="/home/xwh/ragsearch-update"
export PYTHONPATH="$PROJ_ROOT"

# 创建日志目录
mkdir -p "$PROJ_ROOT/runs/logs"

echo "======================================================================"
echo "开始执行任务二：计算 Generator 的天花板分数 (Upper Bound)"
echo "======================================================================"

# 1. HotPotQA Upper Bound
echo "[1/3] Calculating HotPotQA Upper Bound..."
python scripts/oracle_upper_bound.py \
  --dataset_dir datasets/hotpotqa_sample \
  --llm_base_url http://localhost:9001/v1 \
  --generator_model "/home/xwh/models/Qwen3-4B-Instruct-2507" \
  --metrics_weights "em:1,qa_f1:1" \
  > runs/logs/oracle_hotpot.log 2>&1

# 2. MiniWiki Upper Bound
echo "[2/3] Calculating MiniWiki Upper Bound..."
python scripts/oracle_upper_bound.py \
  --dataset_dir datasets/miniwiki_sample \
  --llm_base_url http://localhost:9001/v1 \
  --generator_model "/home/xwh/models/Qwen3-4B-Instruct-2507" \
  --metrics_weights "em:1,qa_f1:1" \
  > runs/logs/oracle_miniwiki.log 2>&1

# 3. M2RAG Upper Bound
echo "[3/3] Calculating M2RAG Upper Bound..."
python scripts/oracle_upper_bound.py \
  --dataset_dir datasets/m2rag_mmqa_100 \
  --llm_base_url http://localhost:9000/v1 \
  --generator_model "/home/xwh/models/Qwen2-VL-7B-Instruct" \
  --metrics_weights "m2rag_overall:1" \
  --allow_images \
  > runs/logs/oracle_m2rag.log 2>&1

echo "Upper Bound 计算完成！结果已保存至 runs/logs/oracle_*.log"
echo ""

echo "======================================================================"
echo "开始执行任务一：算法基准对比 (Random, TPE, GRPO)"
echo "======================================================================"

# 1. HotPotQA 对比 (9001 端口)
echo ">>> Running HotPotQA Baseline..."
for algo in random tpe grpo; do
  echo "  - Algo: $algo"
  python -m autorag_offline_search.cli \
    --config tmp/template_common.yaml \
    --dataset_dir datasets/hotpotqa_sample \
    --out_dir "./runs/hotpot_$algo" \
    --algo "$algo" \
    --train_trials 10 \
    --metrics_weights "em:1,qa_f1:1" \
    --show_trial_progress \
    --resume \
    > "runs/logs/baseline_hotpot_$algo.log" 2>&1
done

# 2. MiniWiki 对比 (9001 端口)
echo ">>> Running MiniWiki Baseline..."
for algo in random tpe grpo; do
  echo "  - Algo: $algo"
  python -m autorag_offline_search.cli \
    --config tmp/template_common.yaml \
    --dataset_dir datasets/miniwiki_sample \
    --out_dir "./runs/miniwiki_$algo" \
    --algo "$algo" \
    --train_trials 10 \
    --metrics_weights "em:1,qa_f1:1,rougeL:1" \
    --show_trial_progress \
    --resume \
    > "runs/logs/baseline_miniwiki_$algo.log" 2>&1
done

# 3. M2RAG 对比 (9000 端口)
echo ">>> Running M2RAG Baseline..."
for algo in random tpe grpo; do
  echo "  - Algo: $algo"
  python -m autorag_offline_search.cli \
    --config tmp/template_multimodal.yaml \
    --dataset_dir datasets/m2rag_mmqa_100 \
    --out_dir "./runs/m2rag_$algo" \
    --algo "$algo" \
    --train_trials 10 \
    --metrics_weights "m2rag_overall:1" \
    --show_trial_progress \
    --resume \
    > "runs/logs/baseline_m2rag_$algo.log" 2>&1
done

echo ""
echo "所有实验已完成！请在 runs/ 目录下查看结果汇总。"
