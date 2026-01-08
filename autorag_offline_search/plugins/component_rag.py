from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..chunking import chunk_docs_word
from ..types import Doc


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _cosine_topk(mat: np.ndarray, q: np.ndarray, k: int) -> List[int]:
    """
    mat: [N, D] normalized
    q: [D] normalized
    """
    if mat.size == 0:
        return []
    sims = mat @ q
    k = min(int(k), sims.shape[0])
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n


class Embedder:
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


class LocalSentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # OpenAI embeddings API supports batching
        resp = self.client.embeddings.create(model=self.model, input=texts)
        # preserve order by index
        data = sorted(resp.data, key=lambda x: x.index)
        arr = np.asarray([d.embedding for d in data], dtype=np.float32)
        return arr


class Reranker:
    def rerank(self, query: str, docs: List[str], topk: int) -> List[int]:
        raise NotImplementedError


class NoReranker(Reranker):
    def rerank(self, query: str, docs: List[str], topk: int) -> List[int]:
        return list(range(min(topk, len(docs))))


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: List[str], topk: int) -> List[int]:
        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs)
        scores = np.asarray(scores, dtype=np.float32)
        k = min(int(topk), len(docs))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx.tolist()


@dataclass
class CacheKeys:
    chunks_key: str
    embed_key: str


def _corpus_fingerprint(docs: Sequence[Doc]) -> str:
    """
    Build a stable fingerprint for the current corpus.
    Important: train/validation corpora can differ even under the same dataset_id.
    If we don't include a corpus fingerprint, cached chunks/embeddings can be reused
    across different corpora, causing totally wrong retrieval (and lots of 'unknown').
    """
    h = hashlib.sha1()
    h.update(f"n={len(docs)}".encode("utf-8"))
    h.update(b"\n")
    # Hash doc_id + a prefix of contents to differentiate corpora cheaply.
    for d in docs:
        did = (d.doc_id or "").encode("utf-8", errors="ignore")
        h.update(did)
        h.update(b"\0")
        c = (d.contents or "")[:512].encode("utf-8", errors="ignore")
        h.update(c)
        h.update(b"\0")
    return h.hexdigest()


def _cache_keys(dataset_id: str, corpus_fp: str, config: Dict) -> CacheKeys:
    # Only include components that affect chunking + embeddings.
    chunk_part = {
        "chunker": config.get("chunker", "fixed_window"),
        "chunk_size": config.get("chunk_size"),
        "chunk_overlap": config.get("chunk_overlap"),
        "min_chunk_words": config.get("min_chunk_words", 8),
    }
    emb_part = {
        "embedder_backend": config.get("embedder_backend", "local"),
        "embedder_model": config.get("embedder_model", "BAAI/bge-m3"),
        "embedder_device": config.get("embedder_device", "cpu"),
        "openai_base_url": config.get("openai_base_url", None),
    }
    return CacheKeys(
        chunks_key=_sha1(dataset_id + "::" + corpus_fp + "::chunks::" + json.dumps(chunk_part, sort_keys=True)),
        embed_key=_sha1(
            dataset_id + "::" + corpus_fp + "::embed::" + json.dumps({**chunk_part, **emb_part}, sort_keys=True)
        ),
    )


class ComponentRag:
    """
    A componentized RAG plugin that supports:
    - Chunker: fixed_window (word-based) and a simple semantic-like sentence splitter
    - Embedder: local SentenceTransformer OR OpenAI embeddings
    - Retriever: vector_topk OR hybrid (bm25 + vector)
    - Reranker: none OR cross-encoder (local)  (LLM rerank can be added later via API)

    All heavy parts (chunks + embeddings) are cached to disk per dataset+config.
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        self.config = dict(config)
        self.docs = list(docs)
        self.verbose = bool(self.config.get("verbose", False))

        self.dataset_id = str(self.config.get("dataset_id") or "dataset")
        self._corpus_fp = _corpus_fingerprint(self.docs)
        self.cache_dir = str(self.config.get("cache_dir") or ".autorag_cache")
        _ensure_dir(self.cache_dir)

        # init once; reused for both chunk embeddings and query embeddings
        self._embedder: Optional[Embedder] = None

        self._init_chunks_and_embeddings()
        self._init_retrieval_helpers()
        self._init_reranker()
        self._init_llm()

    def _init_chunks_and_embeddings(self) -> None:
        keys = _cache_keys(self.dataset_id, self._corpus_fp, self.config)

        chunks_path = os.path.join(self.cache_dir, f"chunks_{keys.chunks_key}.jsonl")
        emb_path = os.path.join(self.cache_dir, f"emb_{keys.embed_key}.npy")

        # 1) chunks
        if os.path.exists(chunks_path):
            if self.verbose:
                print(f"[ComponentRag] loading cached chunks: {chunks_path}")
            chunks = []
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    chunks.append(o)
            self.chunk_texts = [c["text"] for c in chunks]
        else:
            if self.verbose:
                print("[ComponentRag] building chunks...")
            chunker = str(self.config.get("chunker", "fixed_window"))
            if chunker == "fixed_window":
                ch = chunk_docs_word(
                    self.docs,
                    chunk_size=int(self.config.get("chunk_size", 256)),
                    chunk_overlap=int(self.config.get("chunk_overlap", 32)),
                    min_chunk_words=int(self.config.get("min_chunk_words", 8)),
                )
                self.chunk_texts = [c.text for c in ch]
            elif chunker == "semantic_split":
                # lightweight "semantic": sentence based split, then pack into ~chunk_size words
                import re

                sent_re = re.compile(r"(?<=[.!?])\s+")
                target = int(self.config.get("chunk_size", 256))
                self.chunk_texts = []
                for d in self.docs:
                    sents = [s.strip() for s in sent_re.split(d.contents or "") if s.strip()]
                    buf: List[str] = []
                    w = 0
                    for s in sents:
                        ws = s.split()
                        if w + len(ws) > target and buf:
                            self.chunk_texts.append(" ".join(buf).strip())
                            buf, w = [], 0
                        buf.append(s)
                        w += len(ws)
                    if buf:
                        self.chunk_texts.append(" ".join(buf).strip())
            else:
                raise ValueError(f"Unknown chunker: {chunker}")

            with open(chunks_path, "w", encoding="utf-8") as f:
                for i, t in enumerate(self.chunk_texts):
                    f.write(json.dumps({"i": i, "text": t}, ensure_ascii=False) + "\n")

        if self.verbose:
            print(f"[ComponentRag] chunks={len(self.chunk_texts)}")

        # 2) init embedder once
        embedder_backend = str(self.config.get("embedder_backend", "local"))
        embedder_model = str(self.config.get("embedder_model", "BAAI/bge-m3"))
        if embedder_backend == "local":
            device = str(self.config.get("embedder_device", "cuda"))
            if self.verbose:
                print(f"[ComponentRag] init local embedder model={embedder_model} device={device}")
            self._embedder = LocalSentenceTransformerEmbedder(embedder_model, device=device)
        elif embedder_backend == "openai":
            base_url = self.config.get("openai_base_url", None)
            api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if self.verbose:
                print(f"[ComponentRag] init OpenAI embedder model={embedder_model} base_url={base_url or 'default'}")
            self._embedder = OpenAIEmbedder(embedder_model, api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unknown embedder_backend: {embedder_backend}")

        # 2) embeddings
        if os.path.exists(emb_path):
            if self.verbose:
                print(f"[ComponentRag] loading cached embeddings: {emb_path}")
            self.chunk_emb = np.load(emb_path).astype(np.float32)
        else:
            if self.verbose:
                print(f"[ComponentRag] computing embeddings for {len(self.chunk_texts)} chunks (may take time)...")
            emb = self._embedder.embed_texts(self.chunk_texts)
            self.chunk_emb = emb.astype(np.float32)
            np.save(emb_path, self.chunk_emb)

        self.chunk_emb = _normalize_rows(self.chunk_emb)

    def _init_retrieval_helpers(self) -> None:
        # BM25 is optional (only used in hybrid mode).
        self._bm25 = None
        if str(self.config.get("retriever", "vector_topk")) == "hybrid":
            try:
                from rank_bm25 import BM25Okapi
            except Exception as e:
                raise RuntimeError("hybrid retriever requires rank_bm25. Please install it.") from e
            toks = [t.lower().split() for t in self.chunk_texts]
            self._bm25 = BM25Okapi(toks)

    def _init_reranker(self) -> None:
        r = str(self.config.get("reranker", "none"))
        if r == "none":
            self.reranker = NoReranker()
            return
        if r == "cross_encoder":
            model = str(self.config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
            device = str(self.config.get("reranker_device", "cuda"))
            self.reranker = CrossEncoderReranker(model, device=device)
            return
        raise ValueError(f"Unknown reranker: {r}")

    def _init_llm(self) -> None:
        """
        LLM generator backend (optional).

        Supported:
        - llm_backend = "none" (default): use deterministic generators (best_sentence/concat_chunks)
        - llm_backend = "openai": OpenAI Chat Completions
        - llm_backend = "openai_compat": OpenAI-compatible endpoint (e.g., vLLM)
        - llm_backend = "gemini": Google Gemini (requires google-generativeai)
        """
        self.llm_backend = str(self.config.get("llm_backend", "none"))
        self.llm_model = self.config.get("llm_model", None)
        self.llm_base_url = self.config.get("llm_base_url", None)
        self.llm_temperature = float(self.config.get("llm_temperature", 0.0))
        self.llm_max_tokens = int(self.config.get("llm_max_tokens", 256))
        # Prompt template selection is a hyper-parameter (searchable).
        # Users can extend this dict or provide a custom prompt in config.
        self.prompt_id = str(self.config.get("prompt_id", "factoid_short"))
        self.custom_prompt = self.config.get("prompt_template", None)  # optional override

        self._openai_client = None
        self._gemini_model = None

        if self.llm_backend in {"none", "", None}:
            self.llm_backend = "none"
            return

        if self.llm_backend in {"openai", "openai_compat"}:
            from openai import OpenAI

            api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if self.llm_backend == "openai_compat" and not api_key:
                api_key = "EMPTY"  # vLLM/local endpoints often require a non-empty string

            base_url = self.llm_base_url
            if self.llm_backend == "openai":
                base_url = base_url or None  # use default OpenAI
            else:
                # openai-compatible endpoint (vLLM). Require explicit base_url unless user set a default.
                if not base_url:
                    raise ValueError("llm_backend=openai_compat requires llm_base_url (e.g. http://localhost:9000/v1)")
            self._openai_client = OpenAI(api_key=api_key, base_url=base_url)
            if not self.llm_model:
                raise ValueError("llm_backend=openai/openai_compat requires llm_model")
            return

        if self.llm_backend == "gemini":
            api_key = self.config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("llm_backend=gemini requires GEMINI_API_KEY (or GOOGLE_API_KEY)")
            if not self.llm_model:
                raise ValueError("llm_backend=gemini requires llm_model (e.g. gemini-1.5-flash)")
            try:
                import google.generativeai as genai
            except Exception as e:
                raise RuntimeError("Gemini backend requires 'google-generativeai' package") from e
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(self.llm_model)
            return

        raise ValueError(f"Unknown llm_backend: {self.llm_backend}")

    def _embed_query(self, query: str) -> np.ndarray:
        if self._embedder is None:
            raise RuntimeError("Embedder was not initialized")
        q = self._embedder.embed_query(query)
        return _normalize_vec(np.asarray(q, dtype=np.float32))

    def _retrieve_indices(self, query: str, topk: int) -> List[int]:
        mode = str(self.config.get("retriever", "vector_topk"))
        qv = self._embed_query(query)
        if mode == "vector_topk":
            return _cosine_topk(self.chunk_emb, qv, topk)
        if mode == "hybrid":
            # score = alpha * bm25 + (1-alpha) * cosine
            alpha = float(self.config.get("hybrid_alpha", 0.5))
            bm25_scores = np.asarray(self._bm25.get_scores(query.lower().split()), dtype=np.float32)
            bm25_scores = (bm25_scores - bm25_scores.min()) / (np.ptp(bm25_scores) + 1e-12)
            cos = (self.chunk_emb @ qv).astype(np.float32)
            score = alpha * bm25_scores + (1.0 - alpha) * cos
            k = min(int(topk), score.shape[0])
            idx = np.argpartition(-score, kth=k - 1)[:k]
            idx = idx[np.argsort(-score[idx])]
            return idx.tolist()
        raise ValueError(f"Unknown retriever: {mode}")

    def answer(self, query: str) -> str:
        topk = int(self.config.get("retriever_topk", 10))
        idx = self._retrieve_indices(query, topk=topk)
        if not idx:
            return ""

        # optional rerank
        rerank_topk = int(self.config.get("rerank_topk", min(10, topk)))
        docs = [self.chunk_texts[i] for i in idx]
        ridx = self.reranker.rerank(query, docs, topk=rerank_topk)
        final = [docs[i] for i in ridx]

        # Generation: either LLM (if enabled) or deterministic style.
        answer_style = str(self.config.get("answer_style", self.config.get("generator", "concat_chunks")))

        if self.llm_backend != "none":
            ctx_limit = int(self.config.get("llm_context_chunks", min(8, len(final))))
            ctx = "\n\n---\n\n".join(final[:ctx_limit])
            prompt_templates = {
                # Best for BioASQ-like factoid answers (very short GT).
                "factoid_short": (
                    "You are a QA system. Use ONLY the provided CONTEXT.\n"
                    "Return ONLY the final answer, as short as possible (ideally 1-5 tokens). "
                    "No explanations, no extra words.\n\n"
                    "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nFINAL ANSWER:"
                ),
                # More robust when answers require a little composition; may be longer.
                "evidence_then_answer": (
                    "You are a QA system. Use ONLY the provided CONTEXT.\n"
                    "First extract minimal supporting evidence (1-2 bullet points), then give the final answer.\n"
                    "If the context is insufficient, say 'unknown'.\n\n"
                    "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nEVIDENCE:\n- \n\nFINAL ANSWER:"
                ),
            }

            if self.custom_prompt:
                tmpl = str(self.custom_prompt)
            else:
                if self.prompt_id not in prompt_templates:
                    raise ValueError(f"Unknown prompt_id: {self.prompt_id}. Available: {sorted(prompt_templates.keys())}")
                tmpl = prompt_templates[self.prompt_id]

            prompt = tmpl.format(context=ctx, question=query)

            if self.llm_backend in {"openai", "openai_compat"}:
                resp = self._openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()

            if self.llm_backend == "gemini":
                resp = self._gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.llm_temperature,
                        "max_output_tokens": self.llm_max_tokens,
                    },
                )
                return (getattr(resp, "text", "") or "").strip()

            raise ValueError(f"Unknown llm_backend: {self.llm_backend}")

        # deterministic styles (offline, reproducible)
        if answer_style == "concat_chunks":
            limit = int(self.config.get("concat_word_limit", 120))
            words: List[str] = []
            for t in final:
                for w in t.split():
                    words.append(w)
                    if len(words) >= limit:
                        break
                if len(words) >= limit:
                    break
            return " ".join(words).strip()

        if answer_style == "best_sentence":
            import re

            sent_re = re.compile(r"(?<=[.!?])\s+")
            sents: List[str] = []
            for t in final:
                sents.extend([s.strip() for s in sent_re.split(t) if s.strip()])
            if not sents:
                return final[0].strip()
            qset = set(query.lower().split())
            cand = [s for s in sents if qset.intersection(s.lower().split())]
            cand = cand or sents
            cand.sort(key=lambda s: (len(s), s))
            return cand[0].strip()

        raise ValueError(f"Unknown answer_style/generator: {answer_style}")


    def answer_with_trace(self, query: str) -> Dict:
        """
        Debug helper for analysis: return answer plus retrieval / rerank / generation details.
        This is intentionally best-effort and should not break evaluation if something fails.
        """
        topk = int(self.config.get("retriever_topk", 10))
        idx = self._retrieve_indices(query, topk=topk)
        if not idx:
            return {
                "query": query,
                "answer": "",
                "retriever_topk": topk,
                "retrieved": [],
                "final": [],
            }

        # optional rerank
        rerank_topk = int(self.config.get("rerank_topk", min(10, topk)))
        docs = [self.chunk_texts[i] for i in idx]
        ridx = self.reranker.rerank(query, docs, topk=rerank_topk)
        final = [docs[i] for i in ridx]

        # Generation: either LLM (if enabled) or deterministic style.
        answer_style = str(self.config.get("answer_style", self.config.get("generator", "concat_chunks")))
        out: Dict = {
            "query": query,
            "retriever": str(self.config.get("retriever", "vector_topk")),
            "retriever_topk": topk,
            "retrieved": [{"chunk_i": int(i), "text": self.chunk_texts[i]} for i in idx],
            "reranker": str(self.config.get("reranker", "none")),
            "rerank_topk": rerank_topk,
            "final": [{"rank": int(j), "text": t} for j, t in enumerate(final)],
            "answer_style": answer_style,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "prompt_id": self.prompt_id,
        }

        if self.llm_backend != "none":
            ctx_limit = int(self.config.get("llm_context_chunks", min(8, len(final))))
            ctx = "\n\n---\n\n".join(final[:ctx_limit])
            prompt_templates = {
                "factoid_short": (
                    "You are a QA system. Use ONLY the provided CONTEXT.\n"
                    "Return ONLY the final answer, as short as possible (ideally 1-5 tokens). "
                    "No explanations, no extra words.\n\n"
                    "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nFINAL ANSWER:"
                ),
                "evidence_then_answer": (
                    "You are a QA system. Use ONLY the provided CONTEXT.\n"
                    "First extract minimal supporting evidence (1-2 bullet points), then give the final answer.\n"
                    "If the context is insufficient, say 'unknown'.\n\n"
                    "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nEVIDENCE:\n- \n\nFINAL ANSWER:"
                ),
            }
            if self.custom_prompt:
                tmpl = str(self.custom_prompt)
            else:
                tmpl = prompt_templates.get(self.prompt_id, prompt_templates["factoid_short"])
            prompt = tmpl.format(context=ctx, question=query)

            out["llm_context_chunks"] = ctx_limit
            out["prompt_template"] = tmpl
            out["prompt"] = prompt

            if self.llm_backend in {"openai", "openai_compat"}:
                resp = self._openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                )
                ans = (resp.choices[0].message.content or "").strip()
                out["answer"] = ans
                return out

            if self.llm_backend == "gemini":
                resp = self._gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.llm_temperature,
                        "max_output_tokens": self.llm_max_tokens,
                    },
                )
                ans = (getattr(resp, "text", "") or "").strip()
                out["answer"] = ans
                return out

            out["answer"] = ""
            out["error"] = f"Unknown llm_backend: {self.llm_backend}"
            return out

        # deterministic styles
        if answer_style == "concat_chunks":
            limit = int(self.config.get("concat_word_limit", 120))
            words: List[str] = []
            for t in final:
                for w in t.split():
                    words.append(w)
                    if len(words) >= limit:
                        break
                if len(words) >= limit:
                    break
            out["answer"] = " ".join(words).strip()
            out["concat_word_limit"] = limit
            return out

        if answer_style == "best_sentence":
            import re

            sent_re = re.compile(r"(?<=[.!?])\s+")
            sents: List[str] = []
            for t in final:
                sents.extend([s.strip() for s in sent_re.split(t) if s.strip()])
            if not sents:
                out["answer"] = final[0].strip()
                return out
            qset = set(query.lower().split())
            cand = [s for s in sents if qset.intersection(s.lower().split())]
            cand = cand or sents
            cand.sort(key=lambda s: (len(s), s))
            out["answer"] = cand[0].strip()
            return out

        out["answer"] = ""
        out["error"] = f"Unknown answer_style/generator: {answer_style}"
        return out


