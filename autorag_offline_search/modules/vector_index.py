from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np


class VectorIndex(Protocol):
    def query(self, query_emb: np.ndarray, *, topk: int) -> List[Tuple[int, float]]:
        """
        Return list of (chunk_index, similarity_score) sorted desc by similarity.
        similarity_score should be larger-is-better (cosine-like).
        """


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n


@dataclass(frozen=True)
class NumpyCosineIndex:
    emb_norm: np.ndarray  # [N, D] normalized

    def query(self, query_emb: np.ndarray, *, topk: int) -> List[Tuple[int, float]]:
        if self.emb_norm.size == 0:
            return []
        q = _normalize_vec(np.asarray(query_emb, dtype=np.float32))
        sims = (self.emb_norm @ q).astype(np.float32)
        k = min(int(topk), int(sims.shape[0]))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[int(i)])) for i in idx.tolist()]


class ChromaCosineIndex:
    """
    Chroma-backed vector index with persistent storage.

    Notes:
    - We store embeddings explicitly (no embedding function inside chroma).
    - We use a dedicated persistent directory per (dataset_id, corpus_fp, chunking, embedder_model) key.
    """

    def __init__(
        self,
        *,
        persist_dir: str,
        collection_name: str,
        embeddings: np.ndarray,  # [N, D] float32
        documents: List[str],  # len N
        metadatas: Optional[List[Dict]] = None,
    ):
        import chromadb
        from chromadb.config import Settings
        import os

        if len(documents) != int(embeddings.shape[0]):
            raise ValueError("documents length must match embeddings rows")

        self._persist_dir = persist_dir
        self._collection_name = collection_name

        # Disable anonymized telemetry to avoid noisy errors in some environments.
        # (Some dependency combinations can cause posthog capture signature issues.)
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        # Rebuild collection if count mismatches (simple correctness guard).
        col = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        if col.count() != len(documents):
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                # if delete fails for any reason, fall back to creating a fresh-named collection
                collection_name = collection_name + "_rebuild"
            col = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        if col.count() == 0:
            ids = [str(i) for i in range(len(documents))]
            embs = np.asarray(embeddings, dtype=np.float32).tolist()
            col.add(
                ids=ids,
                documents=documents,
                embeddings=embs,
                metadatas=metadatas,
            )
        self._col = col

    def query(self, query_emb: np.ndarray, *, topk: int) -> List[Tuple[int, float]]:
        if int(topk) <= 0:
            return []
        q = np.asarray(query_emb, dtype=np.float32)
        # ids are returned by default; include controls additional payload fields.
        res = self._col.query(query_embeddings=[q.tolist()], n_results=int(topk), include=["distances"])
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: List[Tuple[int, float]] = []
        # For cosine space in Chroma/HNSW, distance is typically (1 - cosine_sim).
        for sid, dist in zip(ids, dists):
            try:
                i = int(sid)
            except Exception:
                continue
            sim = 1.0 - float(dist)
            out.append((i, sim))
        return out


def build_vector_index(
    *,
    prefer_chroma: bool,
    persist_dir: str,
    collection_name: str,
    embeddings: np.ndarray,
    documents: List[str],
) -> VectorIndex:
    emb = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    if prefer_chroma:
        try:
            return ChromaCosineIndex(
                persist_dir=persist_dir,
                collection_name=collection_name,
                embeddings=emb,
                documents=documents,
                metadatas=[{"i": int(i)} for i in range(len(documents))],
            )
        except Exception:
            # If chroma isn't available or fails to init, fall back to in-memory numpy.
            return NumpyCosineIndex(emb_norm=emb)
    return NumpyCosineIndex(emb_norm=emb)

