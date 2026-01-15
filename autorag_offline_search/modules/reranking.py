from __future__ import annotations

from functools import lru_cache
import os
from typing import List

import numpy as np

from .device import auto_device
from .hf_env import configure_hf_env


@lru_cache(maxsize=8)
def _ce(model_name: str):
    from sentence_transformers import CrossEncoder

    # Best-effort: respect local HF cache + offline mode when provided via env.
    configure_hf_env(
        hf_home=os.environ.get("HF_HOME", ""),
        offline=bool(int(os.environ.get("TRANSFORMERS_OFFLINE", "0") or "0")),
    )

    # Prefer CUDA, but if model load fails (e.g. bad CUDA / OOM), fall back to CPU.
    dev = auto_device()
    if dev == "cuda":
        try:
            return CrossEncoder(model_name, device="cuda")
        except Exception:
            return CrossEncoder(model_name, device="cpu")
    return CrossEncoder(model_name, device="cpu")


def rerank(
    *,
    model_name: str,
    query: str,
    docs: List[str],
    topk: int,
) -> List[int]:
    if not docs:
        return []
    if not model_name or str(model_name).lower() in {"none", "null", "no"}:
        return list(range(min(int(topk), len(docs))))

    m = _ce(model_name)
    pairs = [(query, d) for d in docs]
    try:
        scores = m.predict(pairs)
    except Exception:
        # Runtime fallback: if rerank fails on CUDA (OOM/driver), retry on CPU.
        from sentence_transformers import CrossEncoder

        m_cpu = CrossEncoder(model_name, device="cpu")
        scores = m_cpu.predict(pairs)
    scores = np.asarray(scores, dtype=np.float32)
    k = min(int(topk), len(docs))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()

