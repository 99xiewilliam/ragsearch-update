#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Multimodal RAG smoke test: 3 images (2 distractors) + Qwen3-VL Embedding/Reranker + VL generator")
    p.add_argument("--llm_base_url", type=str, default="http://localhost:9000/v1")
    p.add_argument("--img_target", type=str, default="/home/xwh/ragsearch-update/test/15-1.jpg")
    p.add_argument("--img_d1", type=str, default="/home/xwh/ragsearch-update/test/15-5.png")
    p.add_argument("--img_d2", type=str, default="/home/xwh/ragsearch-update/test/15-8.jpg")
    p.add_argument("--embedder_model", type=str, default="/home/xwh/models/Qwen3-VL-Embedding-2B")
    p.add_argument("--reranker_model", type=str, default="/home/xwh/models/Qwen3-VL-Reranker-2B")
    p.add_argument("--generator_model", type=str, default="/home/xwh/models/Qwen3-VL-4B-Instruct")
    p.add_argument("--cache_dir", type=str, default="/tmp/autorag_mm_3img_cache")
    args = p.parse_args()

    from autorag_offline_search.plugins.rag import UnifiedRag
    from autorag_offline_search.types import Doc

    imgs = [args.img_target, args.img_d1, args.img_d2]
    for fp in imgs:
        if not Path(fp).exists():
            raise SystemExit(f"Image not found: {fp}")

    # 3 docs: 目标图 + 2 干扰
    docs = [
        Doc(doc_id="target", contents="这是一张图片，请结合图片回答问题。", metadata={"image_path": args.img_target}),
        Doc(doc_id="d1", contents="这是一张图片（干扰项）。", metadata={"image_path": args.img_d1}),
        Doc(doc_id="d2", contents="这是一张图片（干扰项）。", metadata={"image_path": args.img_d2}),
    ]

    cfg = {
        "pipeline": "multimodal",
        "llm_base_url": args.llm_base_url,
        "module_logs": True,
        # retrieval: Qwen3-VL-Embedding
        "retriever": "cosine",
        "retriever_topk": 3,
        "embedding_enabled": True,
        "embedder_model": args.embedder_model,
        # keep per-doc as single chunk to preserve image association
        "chunking_enabled": False,
        # rerank: Qwen3-VL-Reranker
        "reranker_enabled": True,
        "reranker_model": args.reranker_model,
        "rerank_topk": 3,
        # LLM modules
        "rewriter_enabled": False,
        "pruner_enabled": False,
        "generator_model": args.generator_model,
        "generator_max_tokens": 64,
        "cache_dir": args.cache_dir,
        "dataset_id": "mm_3img",
    }

    rag = UnifiedRag(docs, cfg)

    # 自动选一张图生成问题：用目标图做 VQA，问“线路号”（如果图不是电车，这个问题可能不适配；你可以改）
    query = "请只根据图片回答：窗外绿色双层电车的线路号是多少？只回答数字。"
    trace = rag.answer_with_trace(query)
    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Ensure repo root import works if executed directly.
    os.chdir(Path(__file__).resolve().parents[1])
    main()

