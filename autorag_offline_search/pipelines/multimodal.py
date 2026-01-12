from __future__ import annotations

from typing import Dict, Sequence

from ..types import Doc
from .common import CommonRagPipeline


class MultiModalRagPipeline(CommonRagPipeline):
    """
    Multimodal RAG (minimal but functional):
    - Still uses the same retrieval/rerank/prune/generate stack
    - Augments `Doc.contents` with textual signals from `Doc.metadata` (caption/ocr/etc.)

    No new hyper-parameters are introduced (by user request).
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        # Respect explicit switch: allow "multimodal pipeline but ignore metadata text".
        from .config import normalize_config

        cfg = normalize_config(config or {})
        if not cfg.multimodal_metadata_enabled:
            super().__init__(docs, config)
            return

        aug = []
        for d in docs:
            meta = d.metadata or {}
            extra_parts = []
            for k in ("caption", "image_caption", "ocr", "alt_text", "transcript", "asr", "summary"):
                v = meta.get(k)
                if isinstance(v, str) and v.strip():
                    extra_parts.append(f"[{k}] {v.strip()}")
            contents = (d.contents or "").strip()
            if extra_parts:
                contents = (contents + "\n\n" + "\n".join(extra_parts)).strip()
            aug.append(Doc(doc_id=d.doc_id, contents=contents, metadata=d.metadata))
        super().__init__(aug, config)

