from __future__ import annotations

from typing import Dict, Sequence

from ..pipelines import CommonRagPipeline, GraphRagPipeline, MultiModalRagPipeline
from ..pipelines.config import normalize_config
from ..types import Doc


class UnifiedRag:
    """
    New default RAG plugin.
    The config must contain `pipeline` in {"common","graph","multimodal"} (default: common).
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        cfg = normalize_config(config or {})
        if cfg.pipeline == "common":
            self._pipe = CommonRagPipeline(docs, config)
        elif cfg.pipeline == "graph":
            self._pipe = GraphRagPipeline(docs, config)
        elif cfg.pipeline == "multimodal":
            self._pipe = MultiModalRagPipeline(docs, config)
        else:
            raise ValueError(f"Unknown pipeline: {cfg.pipeline}")

    def answer(self, query: str) -> str:
        return self._pipe.answer(query)

    def answer_with_trace(self, query: str) -> Dict:
        if hasattr(self._pipe, "answer_with_trace"):
            return self._pipe.answer_with_trace(query)  # type: ignore[no-any-return]
        return {"query": query, "answer": self.answer(query)}

