from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .types import Chunk


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class TfidfRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        stop_words: Optional[str] = "english",
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 200_000,
        lowercase: bool = True,
    ):
        self._chunks = list(chunks)
        self._vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            stop_words=stop_words,  # datasets are EN; safe baseline
            ngram_range=ngram_range,
            max_features=max_features,
        )
        self._X = self._vectorizer.fit_transform([c.text for c in self._chunks])

    def retrieve(self, query: str, *, topk: int) -> List[Retrieved]:
        if topk <= 0:
            return []
        qv = self._vectorizer.transform([query])
        scores = (self._X @ qv.T).toarray().reshape(-1)
        if scores.size == 0:
            return []
        k = min(topk, scores.size)
        # argpartition for speed, then sort topk
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [Retrieved(chunk=self._chunks[i], score=float(scores[i])) for i in idx]


