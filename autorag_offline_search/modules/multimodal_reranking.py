from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import numpy as np


def _looks_like_qwen_vl_reranker(model_name: str) -> bool:
    s = str(model_name or "")
    s_low = s.lower()
    return ("qwen3-vl-reranker" in s_low) or ("qwen3_vl_reranker" in s_low)


@lru_cache(maxsize=2)
def _qwen_vl_reranker(model_name_or_path: str):
    import importlib.util
    from pathlib import Path

    p = Path(str(model_name_or_path))
    script = p / "scripts" / "qwen3_vl_reranker.py"
    if script.exists():
        spec = importlib.util.spec_from_file_location("_qwen3_vl_reranker", str(script))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import: {script}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        Qwen3VLReranker = getattr(mod, "Qwen3VLReranker")
    else:
        from importlib import import_module

        mod = import_module("scripts.qwen3_vl_reranker")
        Qwen3VLReranker = getattr(mod, "Qwen3VLReranker")

    kwargs = {}
    try:
        import torch

        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
    except Exception:
        pass

    return Qwen3VLReranker(model_name_or_path=str(model_name_or_path), **kwargs)


def rerank_multimodal(
    *,
    model_name: str,
    query: str,
    doc_texts: List[str],
    doc_images: List[List[str]],
    topk: int,
    instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
) -> List[int]:
    """
    Return indices into doc_texts/doc_images sorted by descending relevance.
    """
    n = len(doc_texts)
    if n == 0:
        return []
    k = min(int(topk), n)
    if k <= 0:
        return []

    if not _looks_like_qwen_vl_reranker(model_name):
        raise ValueError("rerank_multimodal only supports Qwen3-VL-Reranker models")

    docs = []
    for i in range(n):
        imgs = doc_images[i] if i < len(doc_images) else []
        p0: Optional[str] = None
        if isinstance(imgs, list) and imgs:
            p0 = str(imgs[0] or "").strip()
        d = {}
        t = str(doc_texts[i] or "").strip()
        if t:
            d["text"] = t
        if p0:
            d["image"] = p0
        if not d:
            d["text"] = "NULL"
        docs.append(d)

    scores = _qwen_vl_reranker(model_name).process(
        {
            "instruction": instruction,
            "query": {"text": str(query or "")},
            "documents": docs,
        }
    )
    scores = np.asarray(scores, dtype=np.float32)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()

