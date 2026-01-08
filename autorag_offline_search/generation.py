from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .retrieval import Retrieved


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # naive sentence splitter; good enough for offline baseline
    parts = _SENT_SPLIT_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


@dataclass
class GenerationOutput:
    answer: str
    used_chunks: List[str]


def generate_answer(
    query: str,
    retrieved: Sequence[Retrieved],
    *,
    mode: str,
    concat_word_limit: int = 120,
) -> GenerationOutput:
    if not retrieved:
        return GenerationOutput(answer="", used_chunks=[])

    if mode == "concat_chunks":
        words: List[str] = []
        used = []
        for r in retrieved:
            used.append(r.chunk.chunk_id)
            ws = (r.chunk.text or "").split()
            for w in ws:
                words.append(w)
                if len(words) >= concat_word_limit:
                    break
            if len(words) >= concat_word_limit:
                break
        return GenerationOutput(answer=" ".join(words).strip(), used_chunks=used)

    if mode == "best_sentence":
        # choose the sentence from top chunks that's most similar to query under tf-idf
        candidates: List[str] = []
        used = []
        for r in retrieved:
            used.append(r.chunk.chunk_id)
            candidates.extend(_sentences(r.chunk.text))
        if not candidates:
            return GenerationOutput(answer=retrieved[0].chunk.text.strip(), used_chunks=used)

        vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        X = vec.fit_transform(candidates + [query])
        qv = X[-1]
        C = X[:-1]
        scores = (C @ qv.T).toarray().reshape(-1)
        best_i = int(np.argmax(scores)) if scores.size else 0
        return GenerationOutput(answer=candidates[best_i].strip(), used_chunks=used)

    raise ValueError(f"Unknown generator mode: {mode}")


