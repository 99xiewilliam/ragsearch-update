from __future__ import annotations

from typing import Dict, Sequence

from ..pipelines import CommonRagPipeline, MultiModalRagPipeline
from ..pipelines.config import normalize_config
from ..types import Doc


class UnifiedRag:
    """
    New default RAG plugin.
    The config must contain `pipeline` in {"common","multimodal"} (default: common).
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        cfg = normalize_config(config or {})
        if cfg.pipeline == "common":
            self._pipe = CommonRagPipeline(docs, config)
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

    async def answer_async(self, query: str, *, llm_sems: Dict[str, object] | None = None) -> str:
        if hasattr(self._pipe, "answer_async"):
            return await self._pipe.answer_async(query, llm_sems=llm_sems)  # type: ignore[no-any-return]
        # fallback to sync
        return self.answer(query)

    async def answer_with_trace_async(self, query: str, *, llm_sems: Dict[str, object] | None = None) -> Dict:
        if hasattr(self._pipe, "answer_with_trace_async"):
            return await self._pipe.answer_with_trace_async(query, llm_sems=llm_sems)  # type: ignore[no-any-return]
        return self.answer_with_trace(query)

    def timing_summary(self, *, reset: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Return aggregated timing stats from the underlying pipeline.
        """
        if hasattr(self._pipe, "timing_summary"):
            return self._pipe.timing_summary(reset=reset)  # type: ignore[no-any-return]
        return {}

