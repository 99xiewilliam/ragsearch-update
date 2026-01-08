#!/bin/bash

# 1. 显卡配置：假设你有 2 张卡，编号 0 和 1
export AUTORAG_GPUS="0,1"

# 2. 实验参数
DATASETS_ROOT="/home/xwh/three_dataset_sample_split"
OUT_DIR="/home/xwh/autorag_offline_search_runs/scaling_sweep_$(date +%m%d_%H%M)"
ALGOS="random,greedy,tpe,ucb1,ts,grpo"
BITS="4,6,8,10,12,14"

# 3. 运行命令
# 我们设置 --auto_trials_scale 7，上限 100 次
# 程序会自动根据我们提供的 GPU 列表 (0,1) 来控制并行度
python -m autorag_offline_search.cli \
  --all_datasets_root "$DATASETS_ROOT" \
  --sweep_bits "$BITS" \
  --algo "$ALGOS" \
  --gpus "$AUTORAG_GPUS" \
  --metrics_weights "rougeL:0.3,bertscore_f1:0.3,chrf:0.4" \
  --auto_trials_mode linear \
  --auto_trials_scale 7 \
  --auto_trials_min 15 \
  --auto_trials_max 100 \
  --rag_plugin autorag_offline_search.plugins.component_rag:ComponentRag \
  --llm_base_url http://localhost:9000/v1 \
  --out_dir "$OUT_DIR" \
  --verbose