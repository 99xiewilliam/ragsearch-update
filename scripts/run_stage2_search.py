#!/usr/bin/env python3
"""
第二阶段（在固定模块组合下）的小规模搜索/网格实验：

- 固定模块组合（默认：rewriter=off, chunking=off, reranker=off, pruner=on）
- 只搜索：
  - pruner_mode: select|compress
  - pruner_max_tokens: 多个候选
  - retriever_topk: 多个候选
  - bm25_weight: 多个候选（0=cosine, 1=bm25, (0,1)=hybrid）
  - （可选）pruner_enabled 固定为 true；也可通过 --modules 改回其他组合

输出：
- <out_dir>/results.jsonl：每个 config 的 reward/metrics/seconds/config
- <out_dir>/summary.tsv：按 reward 排序的简单表
- <out_dir>/logs/<config_id>.log：每个 config 的小样本 trace（默认前 3 个样本）

说明：
- 这是“控制变量 + 针对 pruner/检索侧的二阶段”。
- 默认强制 request_parallelism=off 走同步，便于 module_logs/trace 可读。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

from autorag_offline_search.data import load_split
from autorag_offline_search.eval import evaluate_config
from autorag_offline_search.metrics import MetricsConfig, parse_weights
from autorag_offline_search.plugins.rag import UnifiedRag


DEFAULT_MODULE_FLAGS = {
    "rewriter_enabled": False,
    "chunking_enabled": False,
    "reranker_enabled": False,
    "pruner_enabled": True,
}


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("YAML must be a dict/mapping")
    return dict(obj)


def _merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    out.update(dict(b or {}))
    return out


def _sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _config_id(cfg: Dict[str, Any]) -> str:
    # stable-ish id for filenames
    keys = [
        "bm25_weight",
        "retriever_topk",
        "pruner_mode",
        "pruner_max_tokens",
        "rewriter_enabled",
        "chunking_enabled",
        "reranker_enabled",
        "pruner_enabled",
    ]
    sig = {k: cfg.get(k) for k in keys if k in cfg}
    return _sha1_short(json.dumps(sig, sort_keys=True, ensure_ascii=False))


def _as_list_int(s: str) -> List[int]:
    out: List[int] = []
    for x in str(s or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(float(x)))
    return out


def _as_list_float(s: str) -> List[float]:
    out: List[float] = []
    for x in str(s or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _as_list_str(s: str) -> List[str]:
    out: List[str] = []
    for x in str(s or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.append(x)
    return out


def _load_done_set(results_path: str) -> set[str]:
    done = set()
    if not os.path.exists(results_path):
        return done
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = str(obj.get("config_id") or "").strip()
            if cid:
                done.add(cid)
    return done


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-2 search under fixed module flags.")
    p.add_argument("--config", type=str, required=True, help="YAML config path (uses 'base' section if present).")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--metrics_weights", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    # Fixed pipeline/env overrides (avoid editing YAML)
    p.add_argument("--pipeline", type=str, default="", help="common|multimodal (optional override)")
    p.add_argument("--llm_base_url", type=str, default="", help="override llm_base_url")
    p.add_argument("--generator_model", type=str, default="", help="override generator_model")
    p.add_argument("--generator_max_tokens", type=int, default=0)
    p.add_argument("--embedder_model", type=str, default="", help="override embedder_model (when vectors used)")

    # Module flags (default = best from your hotpot_common ablation)
    p.add_argument("--rewriter_enabled", type=str, default="", help="true/false (optional override)")
    p.add_argument("--chunking_enabled", type=str, default="", help="true/false (optional override)")
    p.add_argument("--reranker_enabled", type=str, default="", help="true/false (optional override)")
    p.add_argument("--pruner_enabled", type=str, default="", help="true/false (optional override)")

    # Stage-2 search knobs (comma-separated lists)
    p.add_argument("--bm25_weight_list", type=str, default="0.0,0.5,1.0")
    p.add_argument("--retriever_topk_list", type=str, default="3,5,10")
    p.add_argument("--pruner_mode_list", type=str, default="select,compress")
    p.add_argument("--pruner_max_tokens_list", type=str, default="128,256,512")

    # Fix other knobs to reduce confounds
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--rerank_topk", type=int, default=5)
    p.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--rewriter_prompt_id", type=str, default="rewrite_v1")

    # Inspect controls
    p.add_argument("--inspect_n", type=int, default=3)
    p.add_argument("--inspect_trace_n", type=int, default=3)
    p.add_argument("--inspect_module_logs", action="store_true")

    p.add_argument("--resume", action="store_true", help="If set, skip configs already in results.jsonl")
    args = p.parse_args()

    y = _load_yaml(args.config)
    base_cfg = dict(y.get("base") if isinstance(y.get("base"), dict) else y)

    # Load data
    docs, qas = load_split(args.dataset_dir, args.split)
    if args.limit and int(args.limit) > 0:
        qas = list(qas[: int(args.limit)])

    metrics_cfg = MetricsConfig(weights=parse_weights(args.metrics_weights))

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")

    done = _load_done_set(results_path) if bool(args.resume) else set()

    # Build fixed config (control variables)
    fixed: Dict[str, Any] = {}
    if args.pipeline:
        fixed["pipeline"] = str(args.pipeline)
    if args.llm_base_url:
        fixed["llm_base_url"] = str(args.llm_base_url)
    if args.generator_model:
        fixed["generator_model"] = str(args.generator_model)
    if args.generator_max_tokens and int(args.generator_max_tokens) > 0:
        fixed["generator_max_tokens"] = int(args.generator_max_tokens)
    if args.embedder_model:
        fixed["embedder_model"] = str(args.embedder_model)

    fixed["chunk_size"] = int(args.chunk_size)
    fixed["rerank_topk"] = int(args.rerank_topk)
    fixed["reranker_model"] = str(args.reranker_model)
    fixed["rewriter_prompt_id"] = str(args.rewriter_prompt_id)
    fixed["request_parallelism"] = "off"  # sync for readable logs

    # module flags
    flags = dict(DEFAULT_MODULE_FLAGS)
    for k, v in [
        ("rewriter_enabled", args.rewriter_enabled),
        ("chunking_enabled", args.chunking_enabled),
        ("reranker_enabled", args.reranker_enabled),
        ("pruner_enabled", args.pruner_enabled),
    ]:
        if str(v or "").strip():
            flags[k] = str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    # Stage-2 grids
    bm25_list = _as_list_float(args.bm25_weight_list)
    topk_list = _as_list_int(args.retriever_topk_list)
    pruner_modes = _as_list_str(args.pruner_mode_list)
    pruner_tokens = _as_list_int(args.pruner_max_tokens_list)

    # Always pin module_logs off for main eval; inspect will turn on trace/logs.
    base_main = _merge_dict(base_cfg, fixed)
    base_main = _merge_dict(base_main, flags)
    base_main["module_logs"] = False
    base_main["debug_trace_n"] = 0

    rows: List[Dict[str, Any]] = []

    with open(results_path, "a", encoding="utf-8") as f_out:
        for (bm25_w, rt, pmode, pmax) in product(bm25_list, topk_list, pruner_modes, pruner_tokens):
            cfg = dict(base_main)
            cfg["bm25_weight"] = float(bm25_w)
            cfg["retriever_topk"] = int(rt)
            cfg["pruner_mode"] = str(pmode)
            cfg["pruner_max_tokens"] = int(pmax)

            cid = _config_id(cfg)
            if cid in done:
                continue

            # main eval
            res, stats = evaluate_config(
                docs,
                qas,
                cfg,
                metrics_cfg=metrics_cfg,
                rag_factory=UnifiedRag,
                show_progress=False,
                split_name=str(args.split),
                dump_predictions_path=None,
            )

            item = {
                "config_id": cid,
                "reward": float(res.weighted_reward),
                "seconds": float(stats.seconds),
                "qas": int(stats.qas),
                "metrics": dict(res.per_metric),
                "config": {
                    # store only what we changed + key fixed flags
                    "bm25_weight": float(cfg["bm25_weight"]),
                    "retriever_topk": int(cfg["retriever_topk"]),
                    "pruner_mode": str(cfg["pruner_mode"]),
                    "pruner_max_tokens": int(cfg["pruner_max_tokens"]),
                    **{k: bool(cfg.get(k)) for k in DEFAULT_MODULE_FLAGS.keys()},
                },
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()

            # inspect log (small subset)
            inspect_path = os.path.join(logs_dir, f"{cid}.log")
            if int(args.inspect_n) > 0:
                sub_qas = list(qas[: int(args.inspect_n)])
                cfg_inspect = dict(cfg)
                cfg_inspect["module_logs"] = bool(args.inspect_module_logs)
                cfg_inspect["debug_trace_n"] = max(0, int(args.inspect_trace_n))
                cfg_inspect["debug_trace_split"] = str(args.split)
                cfg_inspect["debug_trace_max_chars"] = 600
                cfg_inspect["debug_trace_every"] = 0
                cfg_inspect["debug_trace_trial"] = 1

                from contextlib import redirect_stderr, redirect_stdout

                with open(inspect_path, "w", encoding="utf-8") as f_log:
                    f_log.write(json.dumps({"_type": "meta", "config_id": cid, "config": cfg_inspect}, ensure_ascii=False) + "\n")
                    with redirect_stdout(f_log), redirect_stderr(f_log):
                        _r2, _s2 = evaluate_config(
                            docs,
                            sub_qas,
                            cfg_inspect,
                            metrics_cfg=metrics_cfg,
                            rag_factory=UnifiedRag,
                            show_progress=False,
                            split_name=str(args.split),
                            dump_predictions_path=None,
                        )
                        _ = (_r2, _s2)
            else:
                with open(inspect_path, "w", encoding="utf-8") as f_log:
                    f_log.write(json.dumps({"_type": "meta", "note": "inspect_n=0"}, ensure_ascii=False) + "\n")

            print(f"[OK] {cid} reward={float(res.weighted_reward):.6f} bm25={bm25_w} topk={rt} pruner_mode={pmode} pmax={pmax} log={inspect_path}")

    # Build TSV summary (read back results.jsonl)
    items = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    items = [it for it in items if isinstance(it, dict) and "reward" in it]
    items.sort(key=lambda x: float(x.get("reward", 0.0)), reverse=True)

    tsv_path = os.path.join(out_dir, "summary.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("rank\treward\tbm25_weight\tretriever_topk\tpruner_mode\tpruner_max_tokens\tconfig_id\n")
        for i, it in enumerate(items, start=1):
            c = it.get("config") or {}
            f.write(
                f"{i}\t{float(it.get('reward',0.0)):.6f}\t{c.get('bm25_weight')}\t{c.get('retriever_topk')}\t{c.get('pruner_mode')}\t{c.get('pruner_max_tokens')}\t{it.get('config_id')}\n"
            )

    print(f"\n[DONE] results: {results_path}")
    print(f"[DONE] summary: {tsv_path}")
    print(f"[DONE] logs: {logs_dir}")


if __name__ == "__main__":
    main()

