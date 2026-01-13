from __future__ import annotations

from typing import Any, Dict, List, Optional


def extract_image_paths(meta: Optional[Dict[str, Any]]) -> List[str]:
    """
    Best-effort extraction of image paths/urls from Doc.metadata.

    Supported keys (common in datasets):
    - image_path / image
    - image_paths / images
    - image_url / image_urls
    - uri / uris (when the doc is an asset)
    """
    if not isinstance(meta, dict):
        return []

    keys = ("image_path", "image", "image_url", "uri")
    list_keys = ("image_paths", "images", "image_urls", "uris")

    out: List[str] = []
    for k in keys:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())

    for k in list_keys:
        v = meta.get(k)
        if isinstance(v, list):
            for item in v:
                s = str(item or "").strip()
                if s:
                    out.append(s)

    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def is_probably_clip_model(model_name: str) -> bool:
    s = str(model_name or "").lower()
    return "clip" in s

