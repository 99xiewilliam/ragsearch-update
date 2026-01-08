from __future__ import annotations

import importlib
from typing import Any


def load_object(spec: str) -> Any:
    """
    Load python object from "module.sub:Attr" spec.
    Example: "autorag_offline_search.plugins.rag_baseline:TfidfRag"
    """
    if not spec or ":" not in spec:
        raise ValueError(f"Invalid spec: {spec!r}. Expected format 'module.sub:Attr'")
    mod, attr = spec.split(":", 1)
    m = importlib.import_module(mod)
    obj = getattr(m, attr)
    return obj


