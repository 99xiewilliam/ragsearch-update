from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    # dataclass NormalizedConfig
    if is_dataclass(obj) and hasattr(obj, key):
        return getattr(obj, key)
    # dict-like
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def enabled(cfg: Any) -> bool:
    return bool(_get(cfg, "module_logs", False)) or bool(_get(cfg, "verbose", False))


def _short(s: Any, n: int = 400) -> str:
    t = str(s or "")
    t = t.replace("\n", "\\n")
    return (t[:n] + "...") if len(t) > n else t


def log(cfg: Any, msg: str) -> None:
    if not enabled(cfg):
        return
    print(msg)


def log_kv(cfg: Any, *, prefix: str, key: str, value: Any, limit: int = 400) -> None:
    if not enabled(cfg):
        return
    print(f"{prefix}{key}={_short(value, limit)}")

