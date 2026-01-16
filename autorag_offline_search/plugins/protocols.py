from __future__ import annotations

from typing import Dict, List, Protocol, Sequence

from ..types import Doc


class RagSystem(Protocol):
    def answer(self, query: str) -> str:
        ...


class RagFactory(Protocol):
    """
    A RAG plugin is just a class (or callable) that can be instantiated as:
      rag = RagFactory(docs, config)
      pred = rag.answer(query)
    """

    def __call__(self, docs: Sequence[Doc], config: Dict) -> RagSystem:
        ...


class SearchSpaceLike(Protocol):
    def all_configs(self) -> List[Dict]:
        ...

    def space_dict(self) -> Dict[str, Sequence]:
        ...


