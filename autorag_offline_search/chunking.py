from __future__ import annotations

import re
from typing import Iterable, List

from .types import Chunk, Doc


_WS_RE = re.compile(r"\s+")


def _words(text: str) -> List[str]:
    # word-ish tokenization: stable, offline, cheap.
    text = _WS_RE.sub(" ", (text or "").strip())
    return text.split(" ") if text else []


def chunk_docs_word(
    docs: Iterable[Doc],
    *,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_words: int = 8,
) -> List[Chunk]:
    """
    Word-based chunking. `start/end` are word indices.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    chunks: List[Chunk] = []
    for doc in docs:
        ws = _words(doc.contents)
        if not ws:
            continue
        i = 0
        step = max(1, chunk_size - chunk_overlap)
        n = len(ws)
        while i < n:
            j = min(n, i + chunk_size)
            if (j - i) >= min_chunk_words:
                text = " ".join(ws[i:j]).strip()
                chunk_id = f"{doc.doc_id}::w{i}-{j}"
                chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, text=text, start=i, end=j))
            if j == n:
                break
            i += step
    return chunks


