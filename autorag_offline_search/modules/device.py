from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def auto_device() -> str:
    """
    Prefer CUDA if available; otherwise CPU.
    Users explicitly asked to NOT expose this as a hyper-parameter.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def env_openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY") or None

