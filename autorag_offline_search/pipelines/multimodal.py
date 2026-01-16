from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Sequence

import numpy as np

from ..modules.chunking import ChunkingConfig, chunk_docs
from ..modules.embedding import embed_query, embed_texts
from ..modules.logging_utils import log, log_kv
from ..modules.multimodal_embedding import embed_mm_chunks, embed_mm_query
from ..modules.multimodal_utils import extract_image_paths, is_probably_clip_model
from ..modules.retrieval import BM25Index, RetrievalConfig, retrieve_indices
from ..modules.reranking import rerank
from ..modules.multimodal_reranking import rerank_multimodal
from ..modules.generator_fixed import GeneratorConfig, generate_answer, generate_answer_with_images
from ..modules.vector_index import build_vector_index
from ..types import Doc
from .common import CommonRagPipeline
from .config import normalize_config, resolve_model


class MultiModalRagPipeline(CommonRagPipeline):
    """
    Multimodal RAG (minimal but functional):
    - Keeps the same retrieval/rerank/prune/generate stack as common
    - Adds two pragmatic multimodal upgrades:
      1) Augment `Doc.contents` with textual signals from `Doc.metadata` (caption/ocr/etc.)
      2) Preserve chunk<->image association and enable CLIP-style embedding when `embedder_model` looks like a CLIP model

    Notes:
    - For Qwen/Qwen3-VL-Embedding-* we keep text-only embedding for now (many of these models support text inputs;
      image support varies by backend).
    - For Qwen/Qwen3-VL-Reranker-* we try to run it via sentence-transformers CrossEncoder (same as common reranker);
      if the model isn't compatible, we fall back to "no rerank" to keep the pipeline running.
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        # Respect explicit switch: allow "multimodal pipeline but ignore metadata text".
        cfg = normalize_config(config or {})
        if not cfg.multimodal_metadata_enabled:
            super().__init__(docs, config)
            return

        aug = []
        for d in docs:
            meta = d.metadata or {}
            extra_parts = []
            for k in ("caption", "image_caption", "ocr", "alt_text", "transcript", "asr", "summary"):
                v = meta.get(k)
                if isinstance(v, str) and v.strip():
                    extra_parts.append(f"[{k}] {v.strip()}")
            contents = (d.contents or "").strip()
            if extra_parts:
                contents = (contents + "\n\n" + "\n".join(extra_parts)).strip()
            aug.append(Doc(doc_id=d.doc_id, contents=contents, metadata=d.metadata))
        super().__init__(aug, config)

    def _corpus_fingerprint_mm(self) -> str:
        """
        Multimodal fingerprint extends common fingerprint with image paths, so caches won't collide.
        """
        h = hashlib.sha1()
        h.update(f"n={len(self.docs)}".encode("utf-8"))
        h.update(b"\n")
        for d in self.docs:
            h.update((d.doc_id or "").encode("utf-8", errors="ignore"))
            h.update(b"\0")
            h.update(((d.contents or "")[:512]).encode("utf-8", errors="ignore"))
            h.update(b"\0")
            imgs = extract_image_paths(d.metadata)
            if imgs:
                h.update(("|".join(imgs[:4])[:512]).encode("utf-8", errors="ignore"))
            h.update(b"\0")
        return h.hexdigest()

    def _init_chunks_embeddings_and_bm25(self) -> None:
        """
        Override to keep chunk<->image association for multimodal embedding.
        """
        self._corpus_fp = self._corpus_fingerprint_mm()
        log(self.cfg, f"[MM:init] dataset_id={self.dataset_id} cache_dir={self.cache_dir}")

        chunk_part = {
            "chunking_method": self.cfg.chunking_method,
            "chunk_size": self.cfg.chunk_size,
            "chunk_overlap": self.cfg.chunk_overlap,
            "min_chunk_words": self.cfg.min_chunk_words,
            "token_model": self.cfg.embedder_model,
            "mm_meta": bool(self.cfg.multimodal_metadata_enabled),
        }
        emb_part = {
            "embedder_model": self.cfg.embedder_model,
            "mm_clip": bool(is_probably_clip_model(self.cfg.embedder_model)),
        }

        chunks_key = hashlib.sha1(
            (self.dataset_id + "::" + self._corpus_fp + "::mm_chunks::" + json.dumps(chunk_part, sort_keys=True)).encode("utf-8")
        ).hexdigest()
        embed_key = hashlib.sha1(
            (
                self.dataset_id
                + "::"
                + self._corpus_fp
                + "::mm_embed::"
                + json.dumps({**chunk_part, **emb_part}, sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()

        chunks_path = os.path.join(self.cache_dir, f"mm_chunks_{chunks_key}.jsonl")
        emb_path = os.path.join(self.cache_dir, f"mm_emb_{embed_key}.npy")

        self._chunk_texts = []
        self._chunk_images: List[List[str]] = []
        self._vector_index = None
        self._chunk_emb_raw = None
        self._bm25 = None

        # chunks (+ images)
        if os.path.exists(chunks_path):
            log(self.cfg, f"[MM:chunking] load cached chunks: {chunks_path}")
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    t = str(o.get("text", "") or "").strip()
                    if not t:
                        continue
                    self._chunk_texts.append(t)
                    imgs = o.get("images", []) or []
                    if isinstance(imgs, list):
                        self._chunk_images.append([str(x) for x in imgs if str(x).strip()])
                    else:
                        self._chunk_images.append([])
        else:
            log(
                self.cfg,
                f"[MM:chunking] build chunks enabled={self.cfg.chunking_enabled} method={self.cfg.chunking_method} "
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
                for d in self.docs:
                    imgs = extract_image_paths(d.metadata)
                    parts = [t for t in chunk_docs([d], cfg=ch_cfg) if t.strip()]
                    for t in parts:
                        self._chunk_texts.append(t)
                        self._chunk_images.append(imgs)
            else:
                for d in self.docs:
                    t = str(d.contents or "").strip()
                    if not t:
                        continue
                    self._chunk_texts.append(t)
                    self._chunk_images.append(extract_image_paths(d.metadata))

            with open(chunks_path, "w", encoding="utf-8") as f:
                for i, t in enumerate(self._chunk_texts):
                    f.write(json.dumps({"i": i, "text": t, "images": self._chunk_images[i]}, ensure_ascii=False) + "\n")
            log(self.cfg, f"[MM:chunking] wrote chunks: {chunks_path}")

        log(self.cfg, f"[MM:chunking] chunks={len(self._chunk_texts)} images_chunks={sum(1 for x in self._chunk_images if x)}")
        images_chunks = sum(1 for x in self._chunk_images if x)
        if images_chunks > 0:
            em = str(self.cfg.embedder_model or "")
            rr = str(self.cfg.reranker_model or "")
            rw = str(self.cfg.rewriter_model or "")
            pr = str(self.cfg.pruner_model or "")
            # Heuristic warning: image-heavy corpus but text-only models selected.
            looks_text_embedder = (not is_probably_clip_model(em)) and ("qwen3-vl-embedding" not in em.lower())
            looks_text_reranker = (("qwen3-vl-reranker" not in rr.lower()) and ("/qwen3-vl-reranker" not in rr.lower()))
            looks_text_llm = ("vl" not in rw.lower()) and ("vl" not in pr.lower()) and ("/qwen3-vl" not in (rw.lower() + pr.lower()))
            if looks_text_embedder:
                log(self.cfg, "[MM:warn] Detected image chunks, but embedder_model looks text-only. Retrieval may degrade for image-heavy docs.")
            if self.cfg.reranker_enabled and looks_text_reranker:
                log(self.cfg, "[MM:warn] Detected image chunks, but reranker_model looks text-only. Consider Qwen3-VL-Reranker-* for multimodal rerank.")
            if (self.cfg.rewriter_enabled or self.cfg.pruner_enabled) and looks_text_llm:
                log(self.cfg, "[MM:warn] Detected image chunks, but rewriter/pruner models look text-only. Consider using a VL-capable model (e.g. Qwen3-VL-4B-Instruct) for multimodal rewriting/pruning.")

        # embeddings (only needed for cosine/hybrid)
        if self.cfg.embedding_enabled and self.cfg.retriever in {"cosine", "hybrid"}:
            log(self.cfg, f"[MM:embedding] enabled model={self.cfg.embedder_model}")
            if os.path.exists(emb_path):
                log(self.cfg, f"[MM:embedding] load cached embeddings: {emb_path}")
                emb = np.load(emb_path).astype(np.float32)
            else:
                log(self.cfg, f"[MM:embedding] compute embeddings for {len(self._chunk_texts)} chunks (may take time)...")
                # Multimodal embedders:
                # - CLIP: image/text encoder
                # - Qwen3-VL-Embedding: unified multimodal embedding model
                if is_probably_clip_model(self.cfg.embedder_model) or ("qwen3-vl-embedding" in str(self.cfg.embedder_model).lower()):
                    emb = embed_mm_chunks(model_name=self.cfg.embedder_model, chunk_texts=self._chunk_texts, chunk_images=self._chunk_images)
                else:
                    emb = embed_texts(self.cfg.embedder_model, self._chunk_texts).astype(np.float32)
                np.save(emb_path, emb)
                log(self.cfg, f"[MM:embedding] saved embeddings: {emb_path}")
            self._chunk_emb_raw = emb
            chroma_dir = os.path.join(self.cache_dir, f"mm_chroma_{embed_key}")
            self._vector_index = build_vector_index(
                prefer_chroma=True,
                persist_dir=chroma_dir,
                collection_name="chunks",
                embeddings=emb,
                documents=self._chunk_texts,
            )
            log(self.cfg, f"[MM:vector_index] {type(self._vector_index).__name__} dir={chroma_dir}")
        else:
            self._chunk_emb_raw = None
            self._vector_index = None
            log(self.cfg, f"[MM:embedding] disabled or not needed (retriever={self.cfg.retriever})")

        # bm25 (only needed for bm25/hybrid)
        if self.cfg.retriever in {"bm25", "hybrid"}:
            self._bm25 = BM25Index(self._chunk_texts)
            log(self.cfg, f"[MM:bm25] enabled (chunks={len(self._chunk_texts)})")
        else:
            self._bm25 = None
            log(self.cfg, f"[MM:bm25] disabled (retriever={self.cfg.retriever})")

    def answer(self, query: str) -> str:
        """
        Override query embedding for CLIP-like multimodal embeddings.
        Everything else reuses the common stack.
        """
        from ..modules.pruner import PrunerConfig, prune_chunks
        from ..modules.rewriter import RewriterConfig, rewrite_query

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
            if is_probably_clip_model(self.cfg.embedder_model) or ("qwen3-vl-embedding" in str(self.cfg.embedder_model).lower()):
                q_emb = embed_mm_query(self.cfg.embedder_model, q).astype(np.float32)
            else:
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
        # Gather candidate images from retrieved (and reranked) chunks.
        chosen_chunk_idx = list(idx)
        if self.cfg.reranker_enabled:
            log(self.cfg, f"[COMMON:reranker] enabled model={self.cfg.reranker_model} topk={self.cfg.rerank_topk}")
            try:
                if "Qwen3-VL-Reranker" in str(self.cfg.reranker_model) or "/Qwen3-VL-Reranker" in str(self.cfg.reranker_model):
                    ridx = rerank_multimodal(
                        model_name=self.cfg.reranker_model,
                        query=q,
                        doc_texts=docs,
                        doc_images=[self._chunk_images[i] for i in idx] if hasattr(self, "_chunk_images") else [[] for _ in idx],
                        topk=self.cfg.rerank_topk,
                    )
                else:
                    ridx = rerank(model_name=self.cfg.reranker_model, query=q, docs=docs, topk=self.cfg.rerank_topk)
                chosen_chunk_idx = [idx[i] for i in ridx]
                final = [docs[i] for i in ridx]
                log(self.cfg, f"[COMMON:reranker] reranked_indices={ridx[:10]} (n={len(ridx)})")
            except Exception as e:
                log(self.cfg, f"[COMMON:reranker] failed ({type(e).__name__}: {e}); fallback=no_rerank")
                final = docs
        else:
            log(self.cfg, "[COMMON:reranker] disabled")
            final = docs

        # Collect up to 2 images to pass into the vision generator (if available).
        images: List[str] = []
        if hasattr(self, "_chunk_images"):
            for ci in chosen_chunk_idx[: min(5, len(chosen_chunk_idx))]:
                try:
                    images.extend([p for p in (self._chunk_images[int(ci)] or []) if str(p).strip()])
                except Exception:
                    pass
        # de-dup preserve order
        seen = set()
        images_uniq: List[str] = []
        for p in images:
            if p in seen:
                continue
            seen.add(p)
            images_uniq.append(p)
        images = images_uniq[:2]

        log(self.cfg, f"[COMMON:pruner] enabled={self.cfg.pruner_enabled} model={self.cfg.pruner_model} mode={self.cfg.pruner_mode} max_tokens={self.cfg.pruner_max_tokens}")
        if self.cfg.pruner_enabled:
            log_kv(self.cfg, prefix="[COMMON:pruner] ", key="prompt", value=self.cfg.pruner_prompt, limit=240)
            log(self.cfg, f"[COMMON:pruner] input_chunks={len(final)}")
        final = prune_chunks(
            query=q,
            chunks=final,
            cfg=PrunerConfig(
                enabled=self.cfg.pruner_enabled,
                model=self.cfg.pruner_model,
                prompt=self.cfg.pruner_prompt,
                max_tokens=self.cfg.pruner_max_tokens,
                mode=self.cfg.pruner_mode,
            ),
            llm_base_url=self.cfg.llm_base_url,
            model_resolver=resolve_model,
        )
        log(self.cfg, f"[COMMON:pruner] final_chunks={len(final)}")
        if final:
            preview = "\n\n---\n\n".join(final[:3])  # Show first 3 chunks
            if len(final) > 3:
                preview += f"\n\n... (and {len(final) - 3} more chunks)"
            log_kv(self.cfg, prefix="[COMMON:pruner] ", key="pruned_content", value=preview, limit=400)

        ctx = "\n\n---\n\n".join(final)
        log(self.cfg, f"[COMMON:generator] model={self.cfg.generator_model} max_tokens={self.cfg.generator_max_tokens}")
        log_kv(self.cfg, prefix="[COMMON:generator] ", key="context_preview", value=ctx, limit=240)
        if images:
            gen = generate_answer_with_images(
                query=q0,
                context=ctx,
                image_paths=images,
                cfg=GeneratorConfig(model=resolve_model(self.cfg.generator_model), max_tokens=self.cfg.generator_max_tokens),
                llm_base_url=self.cfg.llm_base_url,
            )
        else:
            gen = generate_answer(
                query=q0,
                context=ctx,
                cfg=GeneratorConfig(model=resolve_model(self.cfg.generator_model), max_tokens=self.cfg.generator_max_tokens),
                llm_base_url=self.cfg.llm_base_url,
            )
        log_kv(self.cfg, prefix="[COMMON:generator] ", key="answer", value=gen, limit=240)
        return gen

    def answer_with_trace(self, query: str) -> Dict:
        """
        Best-effort debug output for eval dumping (multimodal version).
        """
        from ..modules.pruner import PrunerConfig, prune_chunks
        from ..modules.rewriter import RewriterConfig, rewrite_query

        out: Dict = {"query": query, "pipeline": "multimodal"}
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
                if is_probably_clip_model(self.cfg.embedder_model) or ("qwen3-vl-embedding" in str(self.cfg.embedder_model).lower()):
                    q_emb = embed_mm_query(self.cfg.embedder_model, q).astype(np.float32)
                else:
                    q_emb = embed_query(self.cfg.embedder_model, q).astype(np.float32)

            idx = retrieve_indices(
                cfg=RetrievalConfig(method=self.cfg.retriever, topk=self.cfg.retriever_topk, hybrid_alpha=self.cfg.hybrid_alpha),
                query=q,
                vector_index=self._vector_index,
                query_emb=q_emb,
                bm25=self._bm25,
            )
            out["retrieved_indices"] = idx
            out["retrieved"] = [
                {"i": int(i), "text": self._chunk_texts[i], "images": self._chunk_images[i] if hasattr(self, "_chunk_images") else []}
                for i in idx[: min(10, len(idx))]
            ]

            docs = [self._chunk_texts[i] for i in idx]
            chosen_chunk_idx = list(idx)
            if self.cfg.reranker_enabled:
                try:
                    if "Qwen3-VL-Reranker" in str(self.cfg.reranker_model) or "/Qwen3-VL-Reranker" in str(self.cfg.reranker_model):
                        ridx = rerank_multimodal(
                            model_name=self.cfg.reranker_model,
                            query=q,
                            doc_texts=docs,
                            doc_images=[self._chunk_images[i] for i in idx] if hasattr(self, "_chunk_images") else [[] for _ in idx],
                            topk=self.cfg.rerank_topk,
                        )
                    else:
                        ridx = rerank(model_name=self.cfg.reranker_model, query=q, docs=docs, topk=self.cfg.rerank_topk)
                    chosen_chunk_idx = [idx[i] for i in ridx]
                    final = [docs[i] for i in ridx]
                    out["reranked_indices"] = ridx
                except Exception as e:
                    out["reranked_indices"] = []
                    out["rerank_error"] = f"{type(e).__name__}: {e}"
                    final = docs
            else:
                out["reranked_indices"] = []
                final = docs

            images: List[str] = []
            if hasattr(self, "_chunk_images"):
                for ci in chosen_chunk_idx[: min(5, len(chosen_chunk_idx))]:
                    try:
                        images.extend([p for p in (self._chunk_images[int(ci)] or []) if str(p).strip()])
                    except Exception:
                        pass
            seen = set()
            images_uniq: List[str] = []
            for p in images:
                if p in seen:
                    continue
                seen.add(p)
                images_uniq.append(p)
            images = images_uniq[:2]
            out["used_images"] = images

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
            if images:
                ans = generate_answer_with_images(
                    query=q0,
                    context=ctx,
                    image_paths=images,
                    cfg=GeneratorConfig(model=resolve_model(self.cfg.generator_model), max_tokens=self.cfg.generator_max_tokens),
                    llm_base_url=self.cfg.llm_base_url,
                )
            else:
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

