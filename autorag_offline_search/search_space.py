from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class SearchSpace:
    """
    New RAG search space (simplified).

    Keys match `pipelines/config.py::normalize_config`.
    """

    # pipeline category (usually injected by CLI; keep here for completeness)
    pipeline: Sequence[str] = ("common", "multimodal")

    # rewriter
    rewriter_enabled: Sequence[bool] = (False, True)
    # Default to the vLLM-served model only; users can add more models explicitly later.
    rewriter_model: Sequence[str] = ("qwen3",)
    # Multimodal rewriter (opt-in via pipeline == "multimodal")
    multimodal_rewriter_model: Sequence[str] = ("qwen3_vl_4b",)
    rewriter_prompt_id: Sequence[str] = ("rewrite_v1", "rewrite_v2_keywords")

    # chunking
    chunking_enabled: Sequence[bool] = (True, False)
    # Only search chunk_size; other chunking knobs are treated as fixed defaults
    # (users can still override them in YAML/CLI if needed).
    chunk_size: Sequence[int] = (256, 512)

    # embedding
    # NOTE: embedding_enabled is not directly searched; we will auto-disable it for bm25.
    embedding_enabled: Sequence[bool] = (True,)
    # Base (text) embedders
    embedder_model: Sequence[str] = ("BAAI/bge-m3", "intfloat/e5-large-v2")
    # Multimodal embedders (opt-in via pipeline == "multimodal")
    multimodal_embedder_model: Sequence[str] = (
        "Qwen/Qwen3-VL-Embedding-2B",
        "Qwen/Qwen3-VL-Embedding-8B",
        "openai/clip-vit-base-patch32",
    )

    # retrieval (now searchable)
    retriever_topk: Sequence[int] = (3, 5, 10)
    # Single retrieval knob: bm25_weight in [0,1]
    # - 0: cosine
    # - 1: bm25
    # - (0,1): hybrid, with hybrid_alpha=bm25_weight
    bm25_weight: Sequence[float] = (0.0, 0.2, 0.5, 0.8, 1.0)

    # rerank
    reranker_enabled: Sequence[bool] = (True, False)
    # Base (text) rerankers
    reranker_model: Sequence[str] = ("none", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Multimodal rerankers (opt-in via pipeline == "multimodal")
    multimodal_reranker_model: Sequence[str] = (
        "Qwen/Qwen3-VL-Reranker-2B",
        "Qwen/Qwen3-VL-Reranker-8B",
    )
    # rerank_topk (now searchable); we will enforce rerank_topk <= retriever_topk in enumeration.
    rerank_topk: Sequence[int] = (3, 5, 10)

    # multimodal
    multimodal_metadata_enabled: Sequence[bool] = (True,)

    # pruner
    pruner_enabled: Sequence[bool] = (False, True)
    # pruner prompt is fixed to prune_v1 (see normalize_config), but we still allow searching pruner *model*.
    pruner_model: Sequence[str] = ("qwen3",)
    # Multimodal pruner model (opt-in via pipeline == "multimodal")
    multimodal_pruner_model: Sequence[str] = ("qwen3_vl_4b",)

    def space_dict(self) -> Dict[str, Sequence]:
        # TPE/GRPO-style samplers use this dict directly, so it must be
        # pipeline-aware when pipeline is pinned to a single value.
        pipe = tuple(self.pipeline)
        pinned_mm = (len(pipe) == 1 and pipe[0] == "multimodal")

        return {
            "pipeline": self.pipeline,
            "rewriter_enabled": self.rewriter_enabled,
            "rewriter_model": (self.multimodal_rewriter_model if pinned_mm else self.rewriter_model),
            "rewriter_prompt_id": self.rewriter_prompt_id,
            "chunking_enabled": self.chunking_enabled,
            "chunk_size": self.chunk_size,
            "embedding_enabled": self.embedding_enabled,
            "embedder_model": (self.multimodal_embedder_model if pinned_mm else self.embedder_model),
            "retriever_topk": self.retriever_topk,
            "bm25_weight": self.bm25_weight,
            "reranker_enabled": self.reranker_enabled,
            "reranker_model": (self.multimodal_reranker_model if pinned_mm else self.reranker_model),
            "rerank_topk": self.rerank_topk,
            "multimodal_metadata_enabled": self.multimodal_metadata_enabled,
            "pruner_enabled": self.pruner_enabled,
            "pruner_model": (self.multimodal_pruner_model if pinned_mm else self.pruner_model),
        }

    def all_configs(self) -> List[Dict]:
        out: List[Dict] = []
        # Simplified enumeration:
        # - rewriter: (enabled, model, prompt_id)
        # - chunking: (enabled, chunk_size)
        # - embedding: (model) [auto-disabled for bm25]
        # - retrieval: (retriever, retriever_topk, hybrid_alpha)
        # - reranker: (enabled, model, rerank_topk)
        # - pruner: (enabled only)
        for pipeline in self.pipeline:
            mm_meta_list = self.multimodal_metadata_enabled if pipeline == "multimodal" else (True,)

            for re_on in self.rewriter_enabled:
                if pipeline == "multimodal":
                    cand_re = tuple(self.multimodal_rewriter_model)
                else:
                    cand_re = tuple(self.rewriter_model)
                re_models = cand_re if re_on else ("qwen3",)
                re_prompts = self.rewriter_prompt_id if re_on else ("rewrite_v1",)

                for ch_on in self.chunking_enabled:
                    ch_sizes = self.chunk_size if ch_on else (256,)
                    # embedding model candidates (only used when retriever uses vectors)
                    if pipeline == "multimodal":
                        base_emb_models = tuple(self.multimodal_embedder_model)
                    else:
                        base_emb_models = tuple(self.embedder_model)

                    r_topks = tuple(int(x) for x in self.retriever_topk)
                    for w in self.bm25_weight:
                        w = float(w)
                        # derive retriever/hybrid_alpha from bm25_weight
                        if w <= 1e-6:
                            r_method = "cosine"
                            alpha = 0.5
                            embedding_enabled = True
                            emb_models = base_emb_models
                            w = 0.0
                        elif w >= 1.0 - 1e-6:
                            r_method = "bm25"
                            alpha = 0.5
                            embedding_enabled = False
                            # bm25 doesn't need embeddings; also avoid enumerating embedders for it.
                            emb_models = ("BAAI/bge-m3",)
                            w = 1.0
                        else:
                            r_method = "hybrid"
                            alpha = w
                            embedding_enabled = True
                            emb_models = base_emb_models

                        for rr_on in self.reranker_enabled:
                            if pipeline == "multimodal":
                                cand_rr = tuple(self.multimodal_reranker_model)
                            else:
                                cand_rr = tuple(self.reranker_model)
                            rr_models = cand_rr if rr_on else ("none",)

                            for ch_size in ch_sizes:
                                for rt in r_topks:
                                    for emb in emb_models:
                                        for rr in rr_models:
                                            # rerank_topk only meaningful when reranker is enabled.
                                            rr_topks = tuple(int(x) for x in self.rerank_topk) if rr_on else (min(10, rt),)
                                            for rrk in rr_topks:
                                                if rr_on and (rrk <= 0 or rrk > rt):
                                                    continue
                                                for mm in mm_meta_list:
                                                    for pr_on in self.pruner_enabled:
                                                        if pipeline == "multimodal":
                                                            cand_pr = tuple(self.multimodal_pruner_model)
                                                        else:
                                                            cand_pr = tuple(self.pruner_model)
                                                        pr_models = cand_pr if pr_on else ("qwen3",)

                                                        for rm, rp, pm in itertools.product(re_models, re_prompts, pr_models):
                                                            out.append(
                                                                {
                                                                    "pipeline": pipeline,
                                                                    "rewriter_enabled": bool(re_on),
                                                                    "rewriter_model": rm,
                                                                    "rewriter_prompt_id": rp,
                                                                    "chunking_enabled": bool(ch_on),
                                                                    "chunk_size": int(ch_size),
                                                                    "embedding_enabled": bool(embedding_enabled),
                                                                    "embedder_model": emb,
                                                                    "bm25_weight": float(w),
                                                                    "retriever": str(r_method),
                                                                    "retriever_topk": int(rt),
                                                                    "hybrid_alpha": float(alpha),
                                                                    "reranker_enabled": bool(rr_on),
                                                                    "reranker_model": rr,
                                                                    "rerank_topk": int(rrk),
                                                                    "multimodal_metadata_enabled": bool(mm),
                                                                    "pruner_enabled": bool(pr_on),
                                                                    "pruner_model": pm,
                                                                }
                                                            )
        return out


def config_to_str(cfg: Dict) -> str:
    return ",".join(f"{k}={cfg[k]}" for k in sorted(cfg.keys()))
