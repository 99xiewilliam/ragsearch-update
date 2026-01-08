#!/usr/bin/env python3
"""
在固定 dataset/bits 的检索空间下，对比（默认只跑组合算法）：
- portfolio_grpo_ts_tpe
- portfolio_grpo_greedy
- two_stage_tpe_then_grpo

注意：
- 如果你不指定 --rag_plugin，默认使用项目的离线 baseline（TF-IDF + 抽取式生成），不会请求大模型。
- 想让评测请求本地 vLLM/大模型，请用：
    --rag_plugin autorag_offline_search.plugins.component_rag:ComponentRag
  并提供：
    --llm_base_url http://localhost:9000/v1

评估方式与现有 sweep 一致：
每个 algo 都是“train 搜索 -> 取 train best -> 在 validation 上评估一次 val_reward”。

同时支持多 seed 重复，输出每个 (dataset,bits,algo) 的 mean/std/max/min（衡量稳定性）。
"""

import argparse
import os
from dataclasses import asdict
from typing import Dict, List

import pandas as pd

from autorag_offline_search.metrics import MetricsConfig, parse_weights
from autorag_offline_search.sweep import AutoTrialsConfig, run_sweep


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets_root", required=True, help="例如 /home/xwh/three_dataset_sample_split（包含 bioasq/clapnq/2wikimultihopqa 子目录）")
    p.add_argument("--out_dir", required=True, help="输出目录（每个 seed 会放在 out_dir/seed<k>/ 下）")
    p.add_argument("--bits", default="4,6,8,10,12,14", help="bits 列表")
    p.add_argument("--seeds", default="42,43,44", help="多个 seed，用逗号分隔，用于稳定性统计")
    p.add_argument(
        "--algos",
        default="portfolio_grpo_ts_tpe,portfolio_grpo_greedy,two_stage_tpe_then_grpo",
        help="要对比的算法列表（默认只跑组合算法，避免重复跑 grpo/ts/tpe baseline）",
    )

    # 与 cli.py 保持一致的一些参数
    p.add_argument("--metrics_weights", default="rougeL:0.3,bertscore_f1:0.3,chrf:0.4")
    p.add_argument("--bertscore_model", default="microsoft/deberta-xlarge-mnli")
    p.add_argument("--auto_trials_mode", default="linear")
    p.add_argument("--auto_trials_scale", type=float, default=7.0)
    p.add_argument("--auto_trials_min", type=int, default=15)
    p.add_argument("--auto_trials_max", type=int, default=100)

    p.add_argument("--llm_base_url", default="")
    p.add_argument("--cache_dir", default="")
    p.add_argument(
        "--rag_plugin",
        default="",
        help="可选：RAG 插件（默认空=baseline，不调用大模型）。例如 autorag_offline_search.plugins.component_rag:ComponentRag",
    )
    p.add_argument("--gpus", default="", help="例如 0,1 用于并行跑不同 algo")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpus:
        os.environ["AUTORAG_GPUS"] = args.gpus

    bits_list = [int(x.strip()) for x in args.bits.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    metrics_cfg = MetricsConfig(weights=parse_weights(args.metrics_weights), bertscore_model=args.bertscore_model)
    auto_cfg = AutoTrialsConfig(
        mode=args.auto_trials_mode,
        scale=args.auto_trials_scale,
        min_trials=args.auto_trials_min,
        max_trials=args.auto_trials_max,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows: List[Dict] = []
    for seed in seeds:
        out_seed = os.path.join(args.out_dir, f"seed{seed}")
        res = run_sweep(
            datasets_root=args.datasets_root,
            out_dir=out_seed,
            bits_list=bits_list,
            algos=algos,
            metrics_cfg=metrics_cfg,
            seed=seed,
            auto_trials_cfg=auto_cfg,
            rag_plugin=args.rag_plugin,
            cache_dir=args.cache_dir,
            llm_base_url=args.llm_base_url,
            verbose=args.verbose,
        )
        for row in res.get("rows", []):
            row = dict(row)
            row["seed"] = seed
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("No rows produced.")

    # 汇总稳定性：mean/std/min/max
    grp = df.groupby(["dataset", "bits", "algo"])["val_reward"]
    summary = grp.agg(["count", "mean", "std", "min", "max"]).reset_index().sort_values(["dataset", "bits", "mean"], ascending=[True, True, False])

    out_csv = os.path.join(args.out_dir, "combo_stability_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"[OK] Saved summary: {out_csv}")
    print(summary.head(50).to_string(index=False))


if __name__ == "__main__":
    main()


