from __future__ import annotations

import re
from typing import Dict, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from ..chunking import chunk_docs_word
from ..types import Doc


class CharNgramRag:
    """
    Example custom RAG plugin:
    - Word chunking (configurable)
    - Character n-gram TF-IDF retriever (often better for names/typos)
    - "evidence-first" generation: return the best-matching chunk prefix

    Use:
      --rag_plugin autorag_offline_search.plugins.rag_variant:CharNgramRag
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        self.config = dict(config)
        self.chunks = chunk_docs_word(
            docs,
            chunk_size=int(self.config["chunk_size"]),
            chunk_overlap=int(self.config["chunk_overlap"]),
            min_chunk_words=int(self.config.get("min_chunk_words", 8)),
        )

        analyzer = self.config.get("char_analyzer", "char_wb")  # 'char' or 'char_wb'
        ngram_min = int(self.config.get("char_ngram_min", 3))
        ngram_max = int(self.config.get("char_ngram_max", 5))

        self.vec = TfidfVectorizer(
            lowercase=True,
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            max_features=int(self.config.get("max_features", 200_000)),
        )
        self.X = self.vec.fit_transform([c.text for c in self.chunks])

    def answer(self, query: str) -> str:
        if not self.chunks:
            return ""
        q = self.vec.transform([query])
        scores = (self.X @ q.T).toarray().reshape(-1)
        if scores.size == 0:
            return ""
        i = int(scores.argmax())
        text = self.chunks[i].text
        # return a short prefix as "answer" (extractive baseline)
        word_limit = int(self.config.get("concat_word_limit", 120))
        return " ".join(text.split()[:word_limit]).strip()


