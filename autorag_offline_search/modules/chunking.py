from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence

from transformers import AutoTokenizer

from ..types import Doc


_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def _words(text: str) -> List[str]:
    text = _WS_RE.sub(" ", (text or "").strip())
    return text.split(" ") if text else []


@lru_cache(maxsize=8)
def _tokenizer(model: str):
    return AutoTokenizer.from_pretrained(model, use_fast=True)


@dataclass(frozen=True)
class ChunkingConfig:
    method: str  # "semantic" | "token"
    chunk_size: int
    chunk_overlap: int
    min_chunk_words: int
    token_model: str  # which tokenizer to use when method=="token"


def chunk_docs(
    docs: Iterable[Doc],
    *,
    cfg: ChunkingConfig,
) -> List[str]:
    if int(cfg.chunk_size) <= 0:
        raise ValueError("chunk_size must be > 0")
    if int(cfg.chunk_overlap) < 0 or int(cfg.chunk_overlap) >= int(cfg.chunk_size):
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    if int(cfg.min_chunk_words) <= 0:
        raise ValueError("min_chunk_words must be > 0")

    method = str(cfg.method)
    if method == "semantic":
        return _chunk_semantic(docs, chunk_size=int(cfg.chunk_size), min_chunk_words=int(cfg.min_chunk_words))
    if method == "token":
        return _chunk_token(
            docs,
            chunk_size=int(cfg.chunk_size),
            chunk_overlap=int(cfg.chunk_overlap),
            min_chunk_words=int(cfg.min_chunk_words),
            token_model=str(cfg.token_model),
        )
    raise ValueError(f"Unknown chunking method: {method}")


def _chunk_semantic(docs: Iterable[Doc], *, chunk_size: int, min_chunk_words: int) -> List[str]:
    """
    Sentence split, then pack into ~chunk_size words. (No overlap for semantic split.)
    """
    out: List[str] = []
    target = int(chunk_size)
    for d in docs:
        sents = [s.strip() for s in _SENT_RE.split(d.contents or "") if s.strip()]
        buf: List[str] = []
        w = 0
        for s in sents:
            ws = s.split()
            if w + len(ws) > target and buf:
                text = " ".join(buf).strip()
                if len(text.split()) >= int(min_chunk_words):
                    out.append(text)
                buf, w = [], 0
            buf.append(s)
            w += len(ws)
        if buf:
            text = " ".join(buf).strip()
            if len(text.split()) >= int(min_chunk_words):
                out.append(text)
    return out


def _chunk_token(
    docs: Iterable[Doc],
    *,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_words: int,
    token_model: str,
) -> List[str]:
    """
    Token-length chunking using a HuggingFace tokenizer.
    We keep overlap in token units; output is text decoded back.
    """
    tok = _tokenizer(token_model)
    out: List[str] = []
    step = max(1, int(chunk_size) - int(chunk_overlap))

    for d in docs:
        text = (d.contents or "").strip()
        if not text:
            continue
        ids: Sequence[int] = tok.encode(text, add_special_tokens=False)
        n = len(ids)
        i = 0
        while i < n:
            j = min(n, i + int(chunk_size))
            chunk_ids = ids[i:j]
            chunk_text = tok.decode(chunk_ids, skip_special_tokens=True).strip()
            if len(_words(chunk_text)) >= int(min_chunk_words):
                out.append(chunk_text)
            if j == n:
                break
            i += step
    return out

