from __future__ import annotations

from functools import lru_cache
import os
from typing import List

import numpy as np

from .device import auto_device
from .hf_env import configure_hf_env


@lru_cache(maxsize=8)
def _st_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    # Best-effort: respect local HF cache + offline mode when provided via env.
    # Users can set HF_HOME=/home/xwh/models/hf_cache and TRANSFORMERS_OFFLINE=1.
    configure_hf_env(
        hf_home=os.environ.get("HF_HOME", ""),
        offline=bool(int(os.environ.get("TRANSFORMERS_OFFLINE", "0") or "0")),
    )

    # Prefer CUDA, but if model load fails (e.g. bad CUDA / OOM), fall back to CPU.
    dev = auto_device()
    if dev == "cuda":
        try:
            return SentenceTransformer(model_name, device="cuda")
        except Exception:
            return SentenceTransformer(model_name, device="cpu")
    return SentenceTransformer(model_name, device="cpu")


def embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    m = _st_model(model_name)
    try:
        emb = m.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    except Exception:
        # Runtime fallback: if encoding fails on CUDA (OOM/driver), retry on CPU.
        from sentence_transformers import SentenceTransformer

        m_cpu = SentenceTransformer(model_name, device="cpu")
        emb = m_cpu.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def embed_query(model_name: str, query: str) -> np.ndarray:
    return embed_texts(model_name, [query])[0]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def normalize_vec(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n

