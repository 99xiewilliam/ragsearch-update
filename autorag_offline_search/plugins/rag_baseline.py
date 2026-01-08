from __future__ import annotations

from typing import Dict, List, Sequence

from ..pipeline import RagPipeline
from ..types import Doc


class TfidfRag:
    """
    Default RAG plugin: the current baseline pipeline (chunk -> TF-IDF retrieve -> simple generator).

    This exists mainly to demonstrate "pluggability":
      --rag_plugin autorag_offline_search.plugins.rag_baseline:TfidfRag
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        self._pipe = RagPipeline(docs, config)

    def answer(self, query: str) -> str:
        ans, _ = self._pipe.answer(query)
        return ans


