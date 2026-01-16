from __future__ import annotations

from functools import lru_cache
import os
from typing import List

import numpy as np

from .device import auto_device
from .hf_env import configure_hf_env, resolve_hf_hub_cache_root


@lru_cache(maxsize=8)
def _st_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    # Best-effort: respect local HF cache + offline mode when provided via env.
    # Users can set HF_HOME=/home/xwh/models/hf_cache and TRANSFORMERS_OFFLINE=1.
    hf_home = os.environ.get("HF_HOME", "")
    is_offline = bool(int(os.environ.get("TRANSFORMERS_OFFLINE", "0") or "0"))
    configure_hf_env(hf_home=hf_home, offline=is_offline)

    # Determine cache directory
    cache_dir = None
    if hf_home:
        # SentenceTransformer's cache_folder should point to the *hub cache root*.
        # Some environments store it directly under HF_HOME (contains models--*),
        # others under HF_HOME/hub. We auto-detect to avoid offline cache misses.
        cache_dir = resolve_hf_hub_cache_root(hf_home) or hf_home

    # Build kwargs for SentenceTransformer
    st_kwargs = {}
    if cache_dir:
        st_kwargs["cache_folder"] = cache_dir
    if is_offline:
        # SentenceTransformer doesn't directly support local_files_only, but we can
        # set it via environment and it will be picked up by transformers
        st_kwargs["local_files_only"] = True

    # Prefer CUDA, but if model load fails (e.g. bad CUDA / OOM), fall back to CPU.
    dev = auto_device()
    if dev == "cuda":
        try:
            return SentenceTransformer(model_name, device="cuda", **st_kwargs)
        except Exception:
            return SentenceTransformer(model_name, device="cpu", **st_kwargs)
    return SentenceTransformer(model_name, device="cpu", **st_kwargs)


def embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    m = _st_model(model_name)
    try:
        emb = m.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    except Exception:
        # Runtime fallback: if encoding fails on CUDA (OOM/driver), retry on CPU.
        from sentence_transformers import SentenceTransformer

        hf_home = os.environ.get("HF_HOME", "")
        is_offline = bool(int(os.environ.get("TRANSFORMERS_OFFLINE", "0") or "0"))
        st_kwargs = {}
        if hf_home:
            st_kwargs["cache_folder"] = resolve_hf_hub_cache_root(hf_home) or hf_home
        if is_offline:
            st_kwargs["local_files_only"] = True

        m_cpu = SentenceTransformer(model_name, device="cpu", **st_kwargs)
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

