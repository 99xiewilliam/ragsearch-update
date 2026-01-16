from __future__ import annotations

import os
from typing import Optional


def _looks_like_hf_hub_cache_root(path: str) -> bool:
    """
    Heuristic: HuggingFace hub cache roots typically contain dirs like:
      - models--ORG--NAME
      - datasets--ORG--NAME
    """
    try:
        for name in os.listdir(path):
            if name.startswith(("models--", "datasets--")):
                return True
    except Exception:
        return False
    return False


def resolve_hf_hub_cache_root(hf_home: str) -> str:
    """
    Resolve the correct HuggingFace hub cache root directory.

    Why:
    - Some environments store hub cache directly under HF_HOME (contains models--*).
    - Others use the default HF_HOME/hub layout.
    """
    hf_home = (hf_home or "").strip()
    if not hf_home:
        return ""

    cand1 = hf_home
    cand2 = os.path.join(hf_home, "hub")

    if os.path.isdir(cand1) and _looks_like_hf_hub_cache_root(cand1):
        return cand1
    if os.path.isdir(cand2) and _looks_like_hf_hub_cache_root(cand2):
        return cand2

    # Fall back to the standard default.
    return cand2 if os.path.isdir(cand2) else cand1


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

        # Transformers cache is separate from hub cache; keep current behavior.
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))

        # IMPORTANT: choose the correct hub cache root for this machine.
        hub_cache_root = resolve_hf_hub_cache_root(hf_home)
        if hub_cache_root:
            os.environ.setdefault("HF_HUB_CACHE", hub_cache_root)

    if offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

