#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


# Ensure repo root is importable when running as a script (sys.path[0] is scripts/).
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _write_demo_dataset(root: Path) -> None:
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "validation").mkdir(parents=True, exist_ok=True)

    corpus = pd.DataFrame(
        [
            {"doc_id": "d1", "contents": "Paris is the capital of France. It is known for the Eiffel Tower."},
            {"doc_id": "d2", "contents": "Berlin is the capital of Germany."},
            {"doc_id": "d3", "contents": "Tokyo is the capital of Japan."},
        ]
    )
    qa = pd.DataFrame(
        [
            {"qid": "q1", "query": "What is the capital of France?", "generation_gt": ["Paris"]},
            {"qid": "q2", "query": "Capital of Germany?", "generation_gt": ["Berlin"]},
        ]
    )
    for split in ("train", "validation"):
        corpus.to_parquet(root / split / "corpus.parquet", index=False)
        qa.to_parquet(root / split / "qa.parquet", index=False)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one Common RAG pipeline call (with optional demo dataset).")
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="",
        help="Dataset directory containing train/validation splits. If empty and --demo is set, creates a demo dataset.",
    )
    p.add_argument("--demo", action="store_true", help="Create and use a tiny demo dataset under /tmp/autorag_demo.")
    p.add_argument("--split", type=str, default="validation", help="Which split to draw a sample query from: train|validation")
    p.add_argument("--query", type=str, default="", help="Optional query string. If set, bypass dataset QA and use this.")

    # Required: vLLM/OpenAI-compatible endpoint
    p.add_argument("--llm_base_url", type=str, required=True, help="e.g. http://localhost:9000/v1")

    # Model names/paths served by vLLM
    p.add_argument("--generator_model", type=str, default="/home/xwh/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--generator_max_tokens", type=int, default=32)

    # Common pipeline knobs (keep small for a quick smoke run)
    p.add_argument("--rewriter_enabled", action="store_true", help="Enable rewriter. Default off for stability.")
    p.add_argument("--pruner_enabled", action="store_true", help="Enable pruner. Default off for stability.")
    p.add_argument("--reranker_enabled", action="store_true", help="Enable reranker. Default off to avoid downloading models.")
    p.add_argument(
        "--reranker_model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder reranker model name (HuggingFace).",
    )
    p.add_argument("--rerank_topk", type=int, default=5, help="Rerank top-k docs from retrieved candidates.")
    p.add_argument("--retriever", type=str, default="cosine", choices=["cosine", "bm25", "hybrid"])
    p.add_argument("--retriever_topk", type=int, default=5)
    p.add_argument("--embedder_model", type=str, default="BAAI/bge-m3")
    p.add_argument("--chunking_method", type=str, default="token", choices=["semantic", "token"])
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--chunk_overlap", type=int, default=32)
    p.add_argument("--min_chunk_words", type=int, default=4)

    p.add_argument("--cache_dir", type=str, default="/tmp/autorag_cache_demo")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.demo:
        ds = Path("/tmp/autorag_demo")
        if ds.exists():
            # keep it simple: overwrite
            import shutil

            shutil.rmtree(ds)
        _write_demo_dataset(ds)
        dataset_dir = str(ds)
    else:
        dataset_dir = str(args.dataset_dir or "").strip()

    if not dataset_dir:
        raise SystemExit("Provide --dataset_dir or use --demo.")

    # Load one split (docs + qas)
    from autorag_offline_search.data import load_split
    from autorag_offline_search.plugins.rag import UnifiedRag

    docs, qas = load_split(dataset_dir, args.split)
    if not docs:
        raise SystemExit(f"No docs loaded from {dataset_dir}/{args.split}/corpus.parquet")

    if args.query:
        query = args.query
        qid = "custom"
    else:
        if not qas:
            raise SystemExit(f"No qas loaded from {dataset_dir}/{args.split}/qa.parquet")
        query = qas[0].query
        qid = qas[0].qid

    cfg: Dict = {
        "pipeline": "common",
        "llm_base_url": args.llm_base_url,
        "verbose": bool(args.verbose),
        "cache_dir": args.cache_dir,
        "dataset_id": os.path.basename(dataset_dir.rstrip("/")) or "dataset",
        # knobs
        "rewriter_enabled": bool(args.rewriter_enabled),
        "pruner_enabled": bool(args.pruner_enabled),
        "retriever": str(args.retriever),
        "retriever_topk": int(args.retriever_topk),
        "embedder_model": str(args.embedder_model),
        "chunking_method": str(args.chunking_method),
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "min_chunk_words": int(args.min_chunk_words),
        # generator (prompt fixed)
        "generator_model": str(args.generator_model),
        "generator_max_tokens": int(args.generator_max_tokens),
        # keep it fast by default
        "reranker_enabled": bool(args.reranker_enabled),
        "reranker_model": str(args.reranker_model),
        "rerank_topk": int(args.rerank_topk),
        "embedding_enabled": True if args.retriever != "bm25" else False,
        "chunking_enabled": True,
    }

    rag = UnifiedRag(docs, cfg)

    print(f"[RUN] dataset_dir={dataset_dir} split={args.split} qid={qid}")
    print(f"[RUN] query: {query}")

    if hasattr(rag, "answer_with_trace"):
        trace = rag.answer_with_trace(query)  # type: ignore[attr-defined]
        print(json.dumps(trace, ensure_ascii=False, indent=2))
    else:
        ans = rag.answer(query)
        print(json.dumps({"query": query, "answer": ans}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

