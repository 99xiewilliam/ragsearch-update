from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..modules.chunking import ChunkingConfig, chunk_docs
from ..modules.embedding import embed_query, embed_texts
from ..modules.generator_fixed import GeneratorConfig, generate_answer
from ..modules.logging_utils import log, log_kv
from ..modules.pruner import PrunerConfig, prune_chunks
from ..modules.retrieval import BM25Index, RetrievalConfig, retrieve_indices
from ..modules.reranking import rerank
from ..modules.rewriter import RewriterConfig, rewrite_query
from ..modules.vector_index import VectorIndex, build_vector_index
from ..types import Doc
from .config import NormalizedConfig, normalize_config, resolve_model


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _corpus_fingerprint(docs: Sequence[Doc]) -> str:
    h = hashlib.sha1()
    h.update(f"n={len(docs)}".encode("utf-8"))
    h.update(b"\n")
    for d in docs:
        h.update((d.doc_id or "").encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(((d.contents or "")[:512]).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass
class _CacheKeys:
    chunks_key: str
    embed_key: str


def _cache_keys(dataset_id: str, corpus_fp: str, cfg: NormalizedConfig) -> _CacheKeys:
    chunk_part = {
        "chunking_method": cfg.chunking_method,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "min_chunk_words": cfg.min_chunk_words,
        # token chunking depends on tokenizer model (we use embedder model for tokenization)
        "token_model": cfg.embedder_model,
    }
    emb_part = {
        "embedder_model": cfg.embedder_model,
    }
    return _CacheKeys(
        chunks_key=_sha1(dataset_id + "::" + corpus_fp + "::chunks::" + json.dumps(chunk_part, sort_keys=True)),
        embed_key=_sha1(dataset_id + "::" + corpus_fp + "::embed::" + json.dumps({**chunk_part, **emb_part}, sort_keys=True)),
    )


class CommonRagPipeline:
    """
    New "common" RAG pipeline:
    rewriter -> chunking -> embedding -> retriever -> reranker -> pruner -> generator(fixed prompt)
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        self.raw_config = dict(config or {})
        self.cfg = normalize_config(self.raw_config)
        self.docs = list(docs)
        self.dataset_id = str(self.raw_config.get("dataset_id") or "dataset")
        self.cache_dir = str(self.raw_config.get("cache_dir") or ".autorag_cache")
        self.verbose = bool(self.raw_config.get("verbose", False))
        _ensure_dir(self.cache_dir)

        self._corpus_fp = _corpus_fingerprint(self.docs)
        self._chunk_texts: List[str] = []
        self._chunk_doc_ids: List[str] = []
        self._chunk_pos: List[int] = []
        self._vector_index: Optional[VectorIndex] = None
        self._chunk_emb_raw: Optional[np.ndarray] = None
        self._bm25: Optional[BM25Index] = None

        self._init_chunks_embeddings_and_bm25()

    def _init_chunks_embeddings_and_bm25(self) -> None:
        log(self.cfg, f"[COMMON:init] dataset_id={self.dataset_id} cache_dir={self.cache_dir}")
        keys = _cache_keys(self.dataset_id, self._corpus_fp, self.cfg)
        chunks_path = os.path.join(self.cache_dir, f"chunks_{keys.chunks_key}.jsonl")
        emb_path = os.path.join(self.cache_dir, f"emb_{keys.embed_key}.npy")

        # chunks
        if os.path.exists(chunks_path):
            log(self.cfg, f"[COMMON:chunking] load cached chunks: {chunks_path}")
            chunks: List[str] = []
            doc_ids: List[str] = []
            poss: List[int] = []
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    t = str(o.get("text", "") or "").strip()
                    if not t:
                        continue
                    chunks.append(t)
                    doc_ids.append(str(o.get("doc_id", "") or ""))
                    try:
                        poss.append(int(o.get("pos", 0)))
                    except Exception:
                        poss.append(0)
            self._chunk_texts = chunks
            self._chunk_doc_ids = doc_ids
            self._chunk_pos = poss
        else:
            log(
                self.cfg,
                f"[COMMON:chunking] build chunks enabled={self.cfg.chunking_enabled} method={self.cfg.chunking_method} "
                f"size={self.cfg.chunk_size} overlap={self.cfg.chunk_overlap} min_words={self.cfg.min_chunk_words}",
            )
            if self.cfg.chunking_enabled:
                ch_cfg = ChunkingConfig(
                    method=self.cfg.chunking_method,
                    chunk_size=self.cfg.chunk_size,
                    chunk_overlap=self.cfg.chunk_overlap,
                    min_chunk_words=self.cfg.min_chunk_words,
                    token_model=self.cfg.embedder_model,
                )
                from ..modules.chunking import chunk_docs_with_meta

                chunks_meta = chunk_docs_with_meta(self.docs, cfg=ch_cfg)
                self._chunk_texts = [str(c.get("text", "") or "").strip() for c in chunks_meta if str(c.get("text", "") or "").strip()]
                self._chunk_doc_ids = [str(c.get("doc_id", "") or "") for c in chunks_meta if str(c.get("text", "") or "").strip()]
                self._chunk_pos = [int(c.get("pos", 0) or 0) for c in chunks_meta if str(c.get("text", "") or "").strip()]
            else:
                # Skip chunking: treat each doc as one "chunk"
                self._chunk_texts = []
                self._chunk_doc_ids = []
                self._chunk_pos = []
                for d in self.docs:
                    t = str(d.contents or "").strip()
                    if not t:
                        continue
                    self._chunk_texts.append(t)
                    self._chunk_doc_ids.append(str(d.doc_id or ""))
                    self._chunk_pos.append(0)
            with open(chunks_path, "w", encoding="utf-8") as f:
                for i, t in enumerate(self._chunk_texts):
                    f.write(
                        json.dumps(
                            {"i": i, "doc_id": self._chunk_doc_ids[i] if i < len(self._chunk_doc_ids) else "", "pos": self._chunk_pos[i] if i < len(self._chunk_pos) else 0, "text": t},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            log(self.cfg, f"[COMMON:chunking] wrote chunks: {chunks_path}")
        log(self.cfg, f"[COMMON:chunking] chunks={len(self._chunk_texts)}")
        if len(self._chunk_texts) == 0:
            # Avoid downstream crashes (e.g. embedding/index building on empty arrays).
            # This can happen for languages without whitespace when min_chunk_words > 0,
            # or when docs are empty.
            log(self.cfg, "[COMMON:warn] no chunks produced; skip embedding/bm25 and return empty answers")
            self._chunk_emb_raw = None
            self._vector_index = None
            self._bm25 = None
            return

        # embeddings (only needed for cosine/hybrid)
        if self.cfg.embedding_enabled and self.cfg.retriever in {"cosine", "hybrid"}:
            log(self.cfg, f"[COMMON:embedding] enabled model={self.cfg.embedder_model}")
            if os.path.exists(emb_path):
                log(self.cfg, f"[COMMON:embedding] load cached embeddings: {emb_path}")
                emb = np.load(emb_path).astype(np.float32)
            else:
                log(self.cfg, f"[COMMON:embedding] compute embeddings for {len(self._chunk_texts)} chunks (may take time)...")
                emb = embed_texts(self.cfg.embedder_model, self._chunk_texts).astype(np.float32)
                np.save(emb_path, emb)
                log(self.cfg, f"[COMMON:embedding] saved embeddings: {emb_path}")
            self._chunk_emb_raw = emb
            chroma_dir = os.path.join(self.cache_dir, f"chroma_{keys.embed_key}")
            self._vector_index = build_vector_index(
                prefer_chroma=True,
                persist_dir=chroma_dir,
                collection_name="chunks",
                embeddings=emb,
                documents=self._chunk_texts,
            )
            log(self.cfg, f"[COMMON:vector_index] {type(self._vector_index).__name__} dir={chroma_dir}")
        else:
            self._chunk_emb_raw = None
            self._vector_index = None
            log(self.cfg, f"[COMMON:embedding] disabled or not needed (retriever={self.cfg.retriever})")

        # bm25 (only needed for bm25/hybrid)
        if self.cfg.retriever in {"bm25", "hybrid"}:
            self._bm25 = BM25Index(self._chunk_texts)
            log(self.cfg, f"[COMMON:bm25] enabled (chunks={len(self._chunk_texts)})")
        else:
            self._bm25 = None
            log(self.cfg, f"[COMMON:bm25] disabled (retriever={self.cfg.retriever})")

    def answer(self, query: str) -> str:
        log(self.cfg, "[COMMON] ===== answer() =====")
        log_kv(self.cfg, prefix="[COMMON] ", key="query", value=query, limit=200)
        q0 = str(query or "")
        log(self.cfg, f"[COMMON:rewriter] enabled={self.cfg.rewriter_enabled} model={self.cfg.rewriter_model} max_tokens={self.cfg.rewriter_max_tokens}")
        if self.cfg.rewriter_enabled:
            log_kv(self.cfg, prefix="[COMMON:rewriter] ", key="prompt", value=self.cfg.rewriter_prompt, limit=240)
        q = rewrite_query(
            query=q0,
            cfg=RewriterConfig(
                enabled=self.cfg.rewriter_enabled,
                model=self.cfg.rewriter_model,
                prompt=self.cfg.rewriter_prompt,
                max_tokens=self.cfg.rewriter_max_tokens,
            ),
            llm_base_url=self.cfg.llm_base_url,
            model_resolver=resolve_model,
        )
        log_kv(self.cfg, prefix="[COMMON:rewriter] ", key="rewritten_query", value=q, limit=200)

        q_emb = None
        if self._vector_index is not None:
            log(self.cfg, f"[COMMON:embed_query] model={self.cfg.embedder_model}")
            q_emb = embed_query(self.cfg.embedder_model, q).astype(np.float32)

        log(self.cfg, f"[COMMON:retriever] method={self.cfg.retriever} topk={self.cfg.retriever_topk} hybrid_alpha={self.cfg.hybrid_alpha}")
        idx = retrieve_indices(
            cfg=RetrievalConfig(method=self.cfg.retriever, topk=self.cfg.retriever_topk, hybrid_alpha=self.cfg.hybrid_alpha),
            query=q,
            vector_index=self._vector_index,
            query_emb=q_emb,
            bm25=self._bm25,
        )
        log(self.cfg, f"[COMMON:retriever] retrieved_indices={idx[:10]} (n={len(idx)})")
        if not idx:
            return ""

        docs = [self._chunk_texts[i] for i in idx]
        if self.cfg.reranker_enabled:
            log(self.cfg, f"[COMMON:reranker] enabled model={self.cfg.reranker_model} topk={self.cfg.rerank_topk}")
            ridx = rerank(model_name=self.cfg.reranker_model, query=q, docs=docs, topk=self.cfg.rerank_topk)
            final = [docs[i] for i in ridx]
            log(self.cfg, f"[COMMON:reranker] reranked_indices={ridx[:10]} (n={len(ridx)})")
        else:
            log(self.cfg, "[COMMON:reranker] disabled")
            final = docs

        log(self.cfg, f"[COMMON:pruner] enabled={self.cfg.pruner_enabled} model={self.cfg.pruner_model} max_tokens={self.cfg.pruner_max_tokens}")
        if self.cfg.pruner_enabled:
            log_kv(self.cfg, prefix="[COMMON:pruner] ", key="prompt", value=self.cfg.pruner_prompt, limit=240)
        final = prune_chunks(
            query=q,
            chunks=final,
            cfg=PrunerConfig(
                enabled=self.cfg.pruner_enabled,
                model=self.cfg.pruner_model,
                prompt=self.cfg.pruner_prompt,
                max_tokens=self.cfg.pruner_max_tokens,
            ),
            llm_base_url=self.cfg.llm_base_url,
            model_resolver=resolve_model,
        )
        log(self.cfg, f"[COMMON:pruner] final_chunks={len(final)}")

        ctx = "\n\n---\n\n".join(final)
        log(self.cfg, f"[COMMON:generator] model={self.cfg.generator_model} max_tokens={self.cfg.generator_max_tokens}")
        log_kv(self.cfg, prefix="[COMMON:generator] ", key="context_preview", value=ctx, limit=240)
        gen = generate_answer(
            query=q0,  # answer original question
            context=ctx,
            cfg=GeneratorConfig(model=resolve_model(self.cfg.generator_model), max_tokens=self.cfg.generator_max_tokens),
            llm_base_url=self.cfg.llm_base_url,
        )
        log_kv(self.cfg, prefix="[COMMON:generator] ", key="answer", value=gen, limit=240)
        return gen

    def answer_with_trace(self, query: str) -> Dict:
        """
        Best-effort debug output for eval dumping.
        """
        out: Dict = {"query": query, "pipeline": "common"}
        try:
            q0 = str(query or "")
            q = rewrite_query(
                query=q0,
                cfg=RewriterConfig(
                    enabled=self.cfg.rewriter_enabled,
                    model=self.cfg.rewriter_model,
                    prompt=self.cfg.rewriter_prompt,
                    max_tokens=self.cfg.rewriter_max_tokens,
                ),
                llm_base_url=self.cfg.llm_base_url,
                model_resolver=resolve_model,
            )
            out["rewritten_query"] = q

            q_emb = None
            if self._vector_index is not None:
                q_emb = embed_query(self.cfg.embedder_model, q).astype(np.float32)

            idx = retrieve_indices(
                cfg=RetrievalConfig(method=self.cfg.retriever, topk=self.cfg.retriever_topk, hybrid_alpha=self.cfg.hybrid_alpha),
                query=q,
                vector_index=self._vector_index,
                query_emb=q_emb,
                bm25=self._bm25,
            )
            out["retrieved_indices"] = idx
            docs = [self._chunk_texts[i] for i in idx]
            out["retrieved"] = [{"i": int(i), "text": self._chunk_texts[i]} for i in idx[: min(10, len(idx))]]

            if self.cfg.reranker_enabled:
                ridx = rerank(model_name=self.cfg.reranker_model, query=q, docs=docs, topk=self.cfg.rerank_topk)
                final = [docs[i] for i in ridx]
                out["reranked_indices"] = ridx
            else:
                final = docs
                out["reranked_indices"] = []

            pruned = prune_chunks(
                query=q,
                chunks=final,
                cfg=PrunerConfig(
                    enabled=self.cfg.pruner_enabled,
                    model=self.cfg.pruner_model,
                    prompt=self.cfg.pruner_prompt,
                    max_tokens=self.cfg.pruner_max_tokens,
                ),
                llm_base_url=self.cfg.llm_base_url,
                model_resolver=resolve_model,
            )
            out["final_chunks"] = pruned[: min(10, len(pruned))]
            ctx = "\n\n---\n\n".join(pruned)
            ans = generate_answer(
                query=q0,
                context=ctx,
                cfg=GeneratorConfig(model=resolve_model(self.cfg.generator_model), max_tokens=self.cfg.generator_max_tokens),
                llm_base_url=self.cfg.llm_base_url,
            )
            out["answer"] = ans
        except Exception as e:
            out["error"] = repr(e)
            out.setdefault("answer", "")
        return out

