from __future__ import annotations

import os
from typing import Optional


def configure_hf_env(
    *,
    hf_home: Optional[str] = None,
    offline: bool = False,
) -> None:
    """
    Best-effort HuggingFace offline/cache configuration.

    Why:
    - Even when models are already downloaded, HF Hub may still do network HEAD requests
      (e.g. resolve-cache) unless offline is enabled.
    - In demo/production environments with restricted egress, this causes timeouts and noisy logs.

    Environment variables used:
    - HF_HOME / TRANSFORMERS_CACHE: where transformers & sentence-transformers look for models
    - HF_HUB_CACHE: where huggingface_hub stores its cache
    - TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE: prevent any network calls
    """

    # Prefer a user-provided cache directory if present.
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))

    if offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

