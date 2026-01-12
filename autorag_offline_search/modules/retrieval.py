from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .vector_index import VectorIndex

@dataclass(frozen=True)
class RetrievalConfig:
    method: str  # "cosine" | "bm25" | "hybrid"
    topk: int
    hybrid_alpha: float = 0.5


class BM25Index:
    def __init__(self, texts: List[str]):
        from rank_bm25 import BM25Okapi

        toks = [(t or "").lower().split() for t in texts]
        self._bm25 = BM25Okapi(toks)

    def topk(self, query: str, k: int) -> List[int]:
        scores = np.asarray(self._bm25.get_scores((query or "").lower().split()), dtype=np.float32)
        if scores.size == 0:
            return []
        k = min(int(k), int(scores.shape[0]))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx.tolist()

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores((query or "").lower().split()), dtype=np.float32)


def retrieve_indices(
    *,
    cfg: RetrievalConfig,
    query: str,
    vector_index: Optional[VectorIndex],
    query_emb: Optional[np.ndarray],
    bm25: Optional[BM25Index],
) -> List[int]:
    m = str(cfg.method)
    k = int(cfg.topk)
    if k <= 0:
        return []

    if m == "cosine":
        if vector_index is None or query_emb is None:
            raise ValueError("cosine retrieval requires embeddings")
        pairs = vector_index.query(query_emb, topk=k)
        return [i for (i, _s) in pairs]

    if m == "bm25":
        if bm25 is None:
            raise ValueError("bm25 retrieval requires BM25Index")
        return bm25.topk(query, k)

    if m == "hybrid":
        if bm25 is None or vector_index is None or query_emb is None:
            raise ValueError("hybrid retrieval requires both BM25Index and embeddings")
        alpha = float(cfg.hybrid_alpha)
        # Efficient hybrid for vector DB: merge candidates from BM25 and vector, then combine scores.
        bm25_scores = bm25.scores(query)
        if bm25_scores.size == 0:
            return []
        # Candidate pool:
        b_k = min(int(bm25_scores.shape[0]), max(int(k), int(k) * 5))
        bm25_idx = np.argpartition(-bm25_scores, kth=b_k - 1)[:b_k]
        bm25_idx = bm25_idx[np.argsort(-bm25_scores[bm25_idx])]

        v_pairs: List[Tuple[int, float]] = vector_index.query(query_emb, topk=max(int(k), int(k) * 5))
        v_idx = [i for (i, _s) in v_pairs]
        v_map = {int(i): float(s) for (i, s) in v_pairs}

        cand = []
        seen = set()
        for i in bm25_idx.tolist():
            ii = int(i)
            if ii not in seen:
                seen.add(ii)
                cand.append(ii)
        for ii in v_idx:
            if ii not in seen:
                seen.add(ii)
                cand.append(int(ii))

        # Normalize BM25 within candidate set
        b = bm25_scores[cand].astype(np.float32)
        b_norm = (b - float(b.min())) / (float(np.ptp(b)) + 1e-12)

        # Vector sims within candidate set (missing => 0)
        v = np.asarray([v_map.get(int(i), 0.0) for i in cand], dtype=np.float32)
        # Normalize sims to [0,1] within cand for stability
        v_norm = (v - float(v.min())) / (float(np.ptp(v)) + 1e-12) if cand else v

        score = alpha * b_norm + (1.0 - alpha) * v_norm
        order = np.argsort(-score)[: min(int(k), len(cand))]
        return [int(cand[int(j)]) for j in order.tolist()]

    raise ValueError(f"Unknown retriever method: {m}")

