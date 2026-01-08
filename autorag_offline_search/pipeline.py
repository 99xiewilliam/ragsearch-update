from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .chunking import chunk_docs_word
from .generation import generate_answer
from .retrieval import Retrieved, TfidfRetriever
from .types import Doc, QAExample


@dataclass
class PipelineArtifacts:
    chunks_count: int


class RagPipeline:
    """
    Offline, deterministic RAG-ish pipeline:
      docs -> chunks -> TF-IDF retrieve -> simple generator
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        self.config = dict(config)
        self.chunks = chunk_docs_word(
            docs,
            chunk_size=int(self.config["chunk_size"]),
            chunk_overlap=int(self.config["chunk_overlap"]),
            min_chunk_words=int(self.config.get("min_chunk_words", 8)),
        )
        stop_words = self.config.get("stop_words", "english")
        if stop_words in {"none", "None", None, False}:
            stop_words = None
        ngram = self.config.get("ngram_range", (1, 2))
        if isinstance(ngram, str) and "," in ngram:
            a, b = ngram.split(",", 1)
            ngram = (int(a), int(b))
        self.retriever = TfidfRetriever(
            self.chunks,
            stop_words=stop_words,
            ngram_range=tuple(ngram),
            max_features=int(self.config.get("max_features", 200_000)),
            lowercase=bool(self.config.get("lowercase", True)),
        )
        self.artifacts = PipelineArtifacts(chunks_count=len(self.chunks))

    def answer(self, query: str) -> Tuple[str, List[Retrieved]]:
        topk = int(self.config["retriever_topk"])
        retrieved = self.retriever.retrieve(query, topk=topk)
        gen_mode = str(self.config["generator"])
        out = generate_answer(
            query,
            retrieved,
            mode=gen_mode,
            concat_word_limit=int(self.config.get("concat_word_limit", 120)),
        )
        return out.answer, retrieved


