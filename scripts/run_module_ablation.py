#!/usr/bin/env python3
"""
模块消融（控制变量）实验脚本：

- 固定所有“子超参”（从 --config 的 base 读取 + CLI overrides）
- 只枚举模块开关（rewriter/chunking/reranker/pruner）
- 每个组合跑一次 evaluate_config，输出 reward/metrics 汇总
- 另外为每个组合跑一个小规模“inspect pass”（默认前 N 条 QA），把模块日志写到 log 文件，
  方便你逐模块看 pipeline 真实执行情况（rewriter/retrieve/rerank/prune/generate）。

用法示例（common pipeline）：
  PYTHONPATH=/home/xwh/ragsearch-update \\
  /home/xwh/.conda/envs/ragsearch/bin/python scripts/run_module_ablation.py \\
    --config tmp/template_common.yaml \\
    --dataset_dir datasets/hotpotqa_sample \\
    --metrics_weights "em:1,qa_f1:1" \\
    --out_dir runs/ablations/hotpot_common
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from itertools import product
from typing import Any, Dict, List, Tuple

from autorag_offline_search.data import load_split
from autorag_offline_search.eval import evaluate_config
from autorag_offline_search.metrics import MetricsConfig, parse_weights
from autorag_offline_search.plugins.rag import UnifiedRag


MODULE_KEYS = [
    "rewriter_enabled",
    "chunking_enabled",
    "reranker_enabled",
    "pruner_enabled",
]


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("YAML must be a dict/mapping")
    return dict(obj)


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return bool(v)
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _combo_name(flags: Dict[str, bool]) -> str:
    # short names: r/c/rr/p
    mp = {
        "rewriter_enabled": "rewriter",
        "chunking_enabled": "chunking",
        "reranker_enabled": "reranker",
        "pruner_enabled": "pruner",
    }
    parts = []
    for k in MODULE_KEYS:
        parts.append(f"{mp[k]}={'on' if flags[k] else 'off'}")
    return "__".join(parts)


def _merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    out.update(dict(b or {}))
    return out


def _fixed_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    o: Dict[str, Any] = {}
    # pipeline-level
    if args.pipeline:
        o["pipeline"] = str(args.pipeline)
    if args.llm_base_url:
        o["llm_base_url"] = str(args.llm_base_url)
    if args.generator_model:
        o["generator_model"] = str(args.generator_model)
    if args.generator_max_tokens and int(args.generator_max_tokens) > 0:
        o["generator_max_tokens"] = int(args.generator_max_tokens)

    # fixed knobs (control variables)
    if args.rewriter_prompt_id:
        o["rewriter_prompt_id"] = str(args.rewriter_prompt_id)
    if args.chunk_size and int(args.chunk_size) > 0:
        o["chunk_size"] = int(args.chunk_size)
    if args.embedder_model:
        o["embedder_model"] = str(args.embedder_model)
    if args.retriever_topk and int(args.retriever_topk) > 0:
        o["retriever_topk"] = int(args.retriever_topk)
    if args.bm25_weight is not None:
        o["bm25_weight"] = float(args.bm25_weight)
    if args.reranker_model:
        o["reranker_model"] = str(args.reranker_model)
    if args.rerank_topk and int(args.rerank_topk) > 0:
        o["rerank_topk"] = int(args.rerank_topk)

    # logging / execution mode: force sync so module logs are visible and ordered
    o["request_parallelism"] = "off"
    return o


def run_one_combo(
    *,
    docs: List,
    qas: List,
    metrics_cfg: MetricsConfig,
    base_cfg: Dict[str, Any],
    fixed_cfg: Dict[str, Any],
    flags: Dict[str, bool],
    out_dir: str,
    inspect_n: int,
    inspect_module_logs: bool,
    inspect_trace_n: int,
) -> Dict[str, Any]:
    cfg = _merge_dict(base_cfg, fixed_cfg)
    cfg = _merge_dict(cfg, {k: bool(flags[k]) for k in MODULE_KEYS})

    name = _combo_name(flags)
    os.makedirs(out_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 1) main eval (no module logs to avoid huge output)
    cfg_main = dict(cfg)
    cfg_main["module_logs"] = False
    cfg_main["debug_trace_n"] = 0
    res, stats = evaluate_config(
        docs,
        qas,
        cfg_main,
        metrics_cfg=metrics_cfg,
        rag_factory=UnifiedRag,
        show_progress=False,
        split_name="train",
        dump_predictions_path=None,
    )

    # 2) inspect pass (small subset) -> capture stdout/stderr to file
    inspect_path = os.path.join(logs_dir, f"{name}.log")
    if inspect_n and int(inspect_n) > 0:
        sub_qas = list(qas[: int(inspect_n)])
        cfg_inspect = dict(cfg)
        cfg_inspect["module_logs"] = bool(inspect_module_logs)
        cfg_inspect["debug_trace_n"] = max(0, int(inspect_trace_n))
        cfg_inspect["debug_trace_split"] = "train"
        cfg_inspect["debug_trace_max_chars"] = 600
        cfg_inspect["debug_trace_every"] = 0
        cfg_inspect["debug_trace_trial"] = 1

        import sys
        from contextlib import redirect_stderr, redirect_stdout

        with open(inspect_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_type": "meta", "combo": name, "flags": flags, "config": cfg_inspect}, ensure_ascii=False) + "\n")
            with redirect_stdout(f), redirect_stderr(f):
                _res2, _stats2 = evaluate_config(
                    docs,
                    sub_qas,
                    cfg_inspect,
                    metrics_cfg=metrics_cfg,
                    rag_factory=UnifiedRag,
                    show_progress=False,
                    split_name="train",
                    dump_predictions_path=None,
                )
                # keep quiet; logs already captured
                _ = (_res2, _stats2)
    else:
        # still create a placeholder file for clarity
        with open(inspect_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_type": "meta", "combo": name, "note": "inspect_n=0 (no inspect pass)"}, ensure_ascii=False) + "\n")

    return {
        "combo": name,
        "flags": dict(flags),
        "reward": float(res.weighted_reward),
        "metrics": dict(res.per_metric),
        "seconds": float(stats.seconds),
        "qas": int(stats.qas),
        "config_fixed": cfg,
        "inspect_log": inspect_path,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Module ablation runner (control variables).")
    p.add_argument("--config", type=str, required=True, help="YAML config path (uses 'base' section if present).")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train", help="train|validation (no_split dataset will ignore).")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on number of QA examples (0=no cap).")
    p.add_argument("--metrics_weights", type=str, required=True, help='e.g. "em:1,qa_f1:1"')
    p.add_argument("--out_dir", type=str, default="runs/ablations", help="Output dir for summary + logs/")

    # Optional fixed overrides (so你不用改 YAML)
    p.add_argument("--pipeline", type=str, default="", help="Force pipeline: common|multimodal (optional).")
    p.add_argument("--llm_base_url", type=str, default="", help="Override llm_base_url")
    p.add_argument("--generator_model", type=str, default="", help="Override generator_model")
    p.add_argument("--generator_max_tokens", type=int, default=0)

    p.add_argument("--rewriter_prompt_id", type=str, default="", help="Fix rewriter_prompt_id")
    p.add_argument("--chunk_size", type=int, default=0, help="Fix chunk_size")
    p.add_argument("--embedder_model", type=str, default="", help="Fix embedder_model")
    p.add_argument("--retriever_topk", type=int, default=0, help="Fix retriever_topk")
    p.add_argument("--bm25_weight", type=float, default=None, help="Fix bm25_weight in [0,1]")
    p.add_argument("--reranker_model", type=str, default="", help="Fix reranker_model")
    p.add_argument("--rerank_topk", type=int, default=0, help="Fix rerank_topk")

    # Inspect/logging controls
    p.add_argument("--inspect_n", type=int, default=3, help="How many QA examples to run for inspect logs per combo.")
    p.add_argument("--inspect_module_logs", action="store_true", help="If set, print module_logs during inspect pass (very verbose).")
    p.add_argument(
        "--inspect_trace_n",
        type=int,
        default=3,
        help="If >0, print debug trace for first N examples in inspect pass (compact, recommended).",
    )

    # Which modules to sweep
    p.add_argument(
        "--modules",
        type=str,
        default="rewriter,chunking,reranker,pruner",
        help="Comma-separated subset of modules to sweep: rewriter,chunking,reranker,pruner",
    )
    args = p.parse_args()

    y = _load_yaml(args.config)
    base_cfg = dict(y.get("base") if isinstance(y.get("base"), dict) else y)

    docs, qas = load_split(args.dataset_dir, args.split)
    if args.limit and int(args.limit) > 0:
        qas = list(qas[: int(args.limit)])

    metrics_cfg = MetricsConfig(weights=parse_weights(args.metrics_weights))
    fixed_cfg = _fixed_overrides_from_args(args)

    # Build sweep set
    wanted = {s.strip().lower() for s in str(args.modules).split(",") if s.strip()}
    keys = []
    for k in MODULE_KEYS:
        nm = k.replace("_enabled", "")
        if nm in wanted:
            keys.append(k)

    if not keys:
        raise SystemExit("No modules selected to sweep (check --modules).")

    # For modules not swept, keep their base value (or default False) fixed.
    base_flags = {k: _coerce_bool(base_cfg.get(k, False)) for k in MODULE_KEYS}

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for bits in product([False, True], repeat=len(keys)):
        flags = dict(base_flags)
        for k, b in zip(keys, bits):
            flags[k] = bool(b)
        r = run_one_combo(
            docs=docs,
            qas=qas,
            metrics_cfg=metrics_cfg,
            base_cfg=base_cfg,
            fixed_cfg=fixed_cfg,
            flags=flags,
            out_dir=out_dir,
            inspect_n=int(args.inspect_n),
            inspect_module_logs=bool(args.inspect_module_logs),
            inspect_trace_n=int(args.inspect_trace_n),
        )
        results.append(r)
        print(f"[OK] {r['combo']} reward={r['reward']:.6f} seconds={r['seconds']:.2f} inspect_log={r['inspect_log']}")

    # Write summary
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_dir": args.dataset_dir,
                "split": args.split,
                "qas": len(qas),
                "metrics_weights": metrics_cfg.weights,
                "config_base": base_cfg,
                "fixed_overrides": fixed_cfg,
                "modules_swept": keys,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # TSV for quick sorting
    tsv_path = os.path.join(out_dir, "summary.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("combo\treward\tseconds\tqas\tinspect_log\n")
        for r in sorted(results, key=lambda x: float(x.get("reward", 0.0)), reverse=True):
            f.write(f"{r['combo']}\t{r['reward']:.6f}\t{r['seconds']:.2f}\t{r['qas']}\t{r['inspect_log']}\n")

    print(f"\n[DONE] wrote summary: {summary_path}")
    print(f"[DONE] wrote tsv: {tsv_path}")
    print(f"[DONE] logs under: {os.path.join(out_dir, 'logs')}")


if __name__ == "__main__":
    main()

