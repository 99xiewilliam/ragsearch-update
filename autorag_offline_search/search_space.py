from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class SearchSpace:
    """
    New RAG search space (only keep the knobs requested by the user).

    Keys match `pipelines/config.py::normalize_config`.
    """

    # pipeline category (usually injected by CLI; keep here for completeness)
    pipeline: Sequence[str] = ("common", "graph", "multimodal")

    # rewriter
    rewriter_enabled: Sequence[bool] = (False, True)
    # Default to the vLLM-served model only; users can add more models explicitly later.
    rewriter_model: Sequence[str] = ("qwen3",)
    rewriter_prompt_id: Sequence[str] = ("rewrite_v1", "rewrite_v2_keywords")
    rewriter_max_tokens: Sequence[int] = (64, 128)

    # chunking
    chunking_enabled: Sequence[bool] = (True, False)
    chunking_method: Sequence[str] = ("semantic", "token")
    # Keep defaults modest; conditional enumeration further reduces total configs.
    chunk_size: Sequence[int] = (256, 512)
    chunk_overlap: Sequence[int] = (0, 64)
    min_chunk_words: Sequence[int] = (8, 16)

    # embedding
    embedding_enabled: Sequence[bool] = (True, False)
    embedder_model: Sequence[str] = ("BAAI/bge-m3", "intfloat/e5-large-v2")

    # retrieval
    retriever: Sequence[str] = ("cosine", "bm25", "hybrid")
    retriever_topk: Sequence[int] = (10, 40)
    hybrid_alpha: Sequence[float] = (0.3, 0.7)

    # rerank
    reranker_enabled: Sequence[bool] = (True, False)
    reranker_model: Sequence[str] = ("none", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_topk: Sequence[int] = (10, 15)

    # graph
    graph_expand_enabled: Sequence[bool] = (True, False)

    # multimodal
    multimodal_metadata_enabled: Sequence[bool] = (True, False)

    # pruner
    pruner_enabled: Sequence[bool] = (False, True)
    pruner_model: Sequence[str] = ("qwen3",)
    pruner_prompt_id: Sequence[str] = ("prune_v1",)
    pruner_max_tokens: Sequence[int] = (64, 128)

    def space_dict(self) -> Dict[str, Sequence]:
        return {
            "pipeline": self.pipeline,
            "rewriter_enabled": self.rewriter_enabled,
            "rewriter_model": self.rewriter_model,
            "rewriter_prompt_id": self.rewriter_prompt_id,
            "rewriter_max_tokens": self.rewriter_max_tokens,
            "chunking_enabled": self.chunking_enabled,
            "chunking_method": self.chunking_method,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_words": self.min_chunk_words,
            "embedding_enabled": self.embedding_enabled,
            "embedder_model": self.embedder_model,
            "retriever": self.retriever,
            "retriever_topk": self.retriever_topk,
            "hybrid_alpha": self.hybrid_alpha,
            "reranker_enabled": self.reranker_enabled,
            "reranker_model": self.reranker_model,
            "rerank_topk": self.rerank_topk,
            "graph_expand_enabled": self.graph_expand_enabled,
            "multimodal_metadata_enabled": self.multimodal_metadata_enabled,
            "pruner_enabled": self.pruner_enabled,
            "pruner_model": self.pruner_model,
            "pruner_prompt_id": self.pruner_prompt_id,
            "pruner_max_tokens": self.pruner_max_tokens,
        }

    def all_configs(self) -> List[Dict]:
        out: List[Dict] = []
        # Conditional enumeration to avoid exploding irrelevant combinations:
        # - If rewriter/pruner disabled, their sub-params are fixed to a single default.
        # - If retriever != hybrid, hybrid_alpha is fixed to 0.5.
        for pipeline in self.pipeline:
            # pipeline-specific toggles
            graph_expand_list = self.graph_expand_enabled if pipeline == "graph" else (True,)
            mm_meta_list = self.multimodal_metadata_enabled if pipeline == "multimodal" else (True,)

            for re_on in self.rewriter_enabled:
                re_models = self.rewriter_model if re_on else ("qwen3",)
                re_prompts = self.rewriter_prompt_id if re_on else ("rewrite_v1",)
                re_maxtoks = self.rewriter_max_tokens if re_on else (128,)

                for ch_on in self.chunking_enabled:
                    ch_methods = self.chunking_method if ch_on else ("semantic",)
                    ch_sizes = self.chunk_size if ch_on else (256,)
                    ch_overlaps = self.chunk_overlap if ch_on else (0,)
                    minws = self.min_chunk_words if ch_on else (8,)

                    for emb_on in self.embedding_enabled:
                        emb_models = self.embedder_model if emb_on else ("BAAI/bge-m3",)
                        retrievers = self.retriever if emb_on else ("bm25",)

                        for rr_on in self.reranker_enabled:
                            rr_models = self.reranker_model if rr_on else ("none",)
                            rrks = self.rerank_topk if rr_on else (10,)

                            for ch_method in ch_methods:
                                for ch_size in ch_sizes:
                                    for ch_overlap in ch_overlaps:
                                        if int(ch_overlap) >= int(ch_size):
                                            continue
                                        for minw in minws:
                                            for emb in emb_models:
                                                for r in retrievers:
                                                    for topk in self.retriever_topk:
                                                        alphas = self.hybrid_alpha if r == "hybrid" else (0.5,)
                                                        for alpha in alphas:
                                                            for rr in rr_models:
                                                                for rrk in rrks:
                                                                    for ge in graph_expand_list:
                                                                        for mm in mm_meta_list:
                                                                            for pr_on in self.pruner_enabled:
                                                                                pr_models = self.pruner_model if pr_on else ("qwen3",)
                                                                                pr_prompts = self.pruner_prompt_id if pr_on else ("prune_v1",)
                                                                                pr_maxtoks = self.pruner_max_tokens if pr_on else (128,)

                                                                                for rm, rp, rmt in itertools.product(re_models, re_prompts, re_maxtoks):
                                                                                    for pm, pp, pmt in itertools.product(pr_models, pr_prompts, pr_maxtoks):
                                                                                        out.append(
                                                                                            {
                                                                                                "pipeline": pipeline,
                                                                                                "rewriter_enabled": bool(re_on),
                                                                                                "rewriter_model": rm,
                                                                                                "rewriter_prompt_id": rp,
                                                                                                "rewriter_max_tokens": int(rmt),
                                                                                                "chunking_enabled": bool(ch_on),
                                                                                                "chunking_method": ch_method,
                                                                                                "chunk_size": int(ch_size),
                                                                                                "chunk_overlap": int(ch_overlap),
                                                                                                "min_chunk_words": int(minw),
                                                                                                "embedding_enabled": bool(emb_on),
                                                                                                "embedder_model": emb,
                                                                                                "retriever": r,
                                                                                                "retriever_topk": int(topk),
                                                                                                "hybrid_alpha": float(alpha),
                                                                                                "reranker_enabled": bool(rr_on),
                                                                                                "reranker_model": rr,
                                                                                                "rerank_topk": int(rrk),
                                                                                                "graph_expand_enabled": bool(ge),
                                                                                                "multimodal_metadata_enabled": bool(mm),
                                                                                                "pruner_enabled": bool(pr_on),
                                                                                                "pruner_model": pm,
                                                                                                "pruner_prompt_id": pp,
                                                                                                "pruner_max_tokens": int(pmt),
                                                                                            }
                                                                                        )
        return out


def config_to_str(cfg: Dict) -> str:
    return ",".join(f"{k}={cfg[k]}" for k in sorted(cfg.keys()))
