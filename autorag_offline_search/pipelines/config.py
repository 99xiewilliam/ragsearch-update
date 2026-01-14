from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..modules.prompts import PRUNER_PROMPTS, REWRITER_PROMPTS


MODEL_PRESETS: Dict[str, str] = {
    # Keep qwen3 as the only "known good" local path from the original repo.
    "qwen3": "/home/xwh/models/Qwen3-4B-Instruct-2507",
    # Multimodal generator (example local path). Users can override freely via config/CLI.
    "qwen3_vl_4b": "/home/xwh/models/Qwen3-VL-4B-Instruct",
}


def resolve_model(model_id_or_path: str) -> str:
    s = str(model_id_or_path or "").strip()
    if not s:
        return MODEL_PRESETS["qwen3"]
    return MODEL_PRESETS.get(s, s)


@dataclass(frozen=True)
class NormalizedConfig:
    pipeline: str  # "common" | "graph" | "multimodal"
    llm_base_url: str
    # rewriter
    rewriter_enabled: bool
    rewriter_model: str
    rewriter_prompt: str
    rewriter_max_tokens: int
    # chunking
    chunking_enabled: bool
    chunking_method: str  # "semantic" | "token"
    chunk_size: int
    chunk_overlap: int
    min_chunk_words: int
    # embedding
    embedding_enabled: bool
    embedder_model: str
    # retrieval
    retriever: str  # "cosine" | "bm25" | "hybrid"
    retriever_topk: int
    hybrid_alpha: float
    # rerank
    reranker_enabled: bool
    reranker_model: str  # "none" or cross-encoder model name
    rerank_topk: int
    # pruner
    pruner_enabled: bool
    pruner_model: str
    pruner_prompt: str
    pruner_max_tokens: int
    # generator (fixed prompt, but model can be configured as a "non-hyperparam")
    generator_model: str
    generator_max_tokens: int
    # graph
    graph_expand_enabled: bool
    graph_mode: str  # "global" | "local" | "hybrid"
    graph_edge_source: str  # "provided" | "structure" | "knn" | "keyword"
    graph_edges_path: str
    graph_hops: int
    graph_seed_topk: int
    graph_neighbor_topk: int
    graph_max_expanded: int
    # multimodal
    multimodal_metadata_enabled: bool
    # logging
    module_logs: bool


def normalize_config(cfg: Dict) -> NormalizedConfig:
    c = dict(cfg or {})
    pipeline = str(c.get("pipeline", "common"))
    if pipeline not in {"common", "graph", "multimodal"}:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    llm_base_url = str(c.get("llm_base_url", "") or "")
    if not llm_base_url:
        raise ValueError(
            "llm_base_url is required (this project assumes a local OpenAI-compatible vLLM endpoint, e.g. http://localhost:9000/v1)"
        )

    rewriter_enabled = bool(c.get("rewriter_enabled", False))
    rewriter_model = str(c.get("rewriter_model", "qwen3"))
    if "rewriter_prompt_id" in c and str(c.get("rewriter_prompt_id") or "").strip():
        pid = str(c.get("rewriter_prompt_id"))
        rewriter_prompt = str(REWRITER_PROMPTS.get(pid, REWRITER_PROMPTS["rewrite_v1"]))
    else:
        rewriter_prompt = str(c.get("rewriter_prompt", REWRITER_PROMPTS["rewrite_v1"]))
    rewriter_max_tokens = int(c.get("rewriter_max_tokens", 128))

    chunking_enabled = bool(c.get("chunking_enabled", True))
    chunking_method = str(c.get("chunking_method", "semantic"))
    if chunking_method not in {"semantic", "token"}:
        raise ValueError(f"Unknown chunking_method: {chunking_method}")
    chunk_size = int(c.get("chunk_size", 256))
    chunk_overlap = int(c.get("chunk_overlap", 0))
    min_chunk_words = int(c.get("min_chunk_words", 8))

    embedding_enabled = bool(c.get("embedding_enabled", True))
    embedder_model = str(c.get("embedder_model", "BAAI/bge-m3"))

    retriever = str(c.get("retriever", "cosine"))
    if retriever not in {"cosine", "bm25", "hybrid"}:
        raise ValueError(f"Unknown retriever: {retriever}")
    retriever_topk = int(c.get("retriever_topk", 10))
    hybrid_alpha = float(c.get("hybrid_alpha", 0.5))

    reranker_enabled = bool(c.get("reranker_enabled", True))
    reranker_model = str(c.get("reranker_model", "none"))
    rerank_topk = int(c.get("rerank_topk", min(10, retriever_topk)))

    pruner_enabled = bool(c.get("pruner_enabled", False))
    pruner_model = str(c.get("pruner_model", "qwen3"))
    if "pruner_prompt_id" in c and str(c.get("pruner_prompt_id") or "").strip():
        pid = str(c.get("pruner_prompt_id"))
        pruner_prompt = str(PRUNER_PROMPTS.get(pid, PRUNER_PROMPTS["prune_v1"]))
    else:
        pruner_prompt = str(c.get("pruner_prompt", PRUNER_PROMPTS["prune_v1"]))
    pruner_max_tokens = int(c.get("pruner_max_tokens", 128))

    # Generator prompt is fixed, but generator *model* can be overridden by YAML/CLI.
    # We make the default model pipeline-aware:
    # - common/graph: text LLM (qwen3)
    # - multimodal  : VL LLM (qwen3_vl_4b)
    if "generator_model" in c and str(c.get("generator_model") or "").strip():
        generator_model = str(c.get("generator_model"))
    else:
        generator_model = "qwen3_vl_4b" if pipeline == "multimodal" else "qwen3"
    generator_max_tokens = int(c.get("generator_max_tokens", 128))

    graph_expand_enabled = bool(c.get("graph_expand_enabled", True))
    graph_mode = str(c.get("graph_mode", "hybrid"))
    if graph_mode not in {"global", "local", "hybrid"}:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")
    graph_edge_source = str(c.get("graph_edge_source", c.get("graph_builder", "keyword")))
    if graph_edge_source not in {"provided", "structure", "knn", "keyword"}:
        raise ValueError(f"Unknown graph_edge_source: {graph_edge_source}")
    graph_edges_path = str(c.get("graph_edges_path", "") or "")
    graph_hops = int(c.get("graph_hops", 1))
    if graph_hops <= 0:
        raise ValueError("graph_hops must be >= 1")
    graph_seed_topk = int(c.get("graph_seed_topk", min(3, retriever_topk)))
    graph_neighbor_topk = int(c.get("graph_neighbor_topk", 50))
    graph_max_expanded = int(c.get("graph_max_expanded", max(200, int(retriever_topk) * 10)))
    multimodal_metadata_enabled = bool(c.get("multimodal_metadata_enabled", True))
    module_logs = bool(c.get("module_logs", False))

    # basic guards
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    if not embedding_enabled and retriever in {"cosine", "hybrid"}:
        raise ValueError("embedding_enabled=false requires retriever=bm25")

    return NormalizedConfig(
        pipeline=pipeline,
        llm_base_url=llm_base_url,
        rewriter_enabled=rewriter_enabled,
        rewriter_model=rewriter_model,
        rewriter_prompt=rewriter_prompt,
        rewriter_max_tokens=rewriter_max_tokens,
        chunking_enabled=chunking_enabled,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_words=min_chunk_words,
        embedding_enabled=embedding_enabled,
        embedder_model=embedder_model,
        retriever=retriever,
        retriever_topk=retriever_topk,
        hybrid_alpha=hybrid_alpha,
        reranker_enabled=reranker_enabled,
        reranker_model=reranker_model,
        rerank_topk=rerank_topk,
        pruner_enabled=pruner_enabled,
        pruner_model=pruner_model,
        pruner_prompt=pruner_prompt,
        pruner_max_tokens=pruner_max_tokens,
        generator_model=generator_model,
        generator_max_tokens=generator_max_tokens,
        graph_expand_enabled=graph_expand_enabled,
        graph_mode=graph_mode,
        graph_edge_source=graph_edge_source,
        graph_edges_path=graph_edges_path,
        graph_hops=graph_hops,
        graph_seed_topk=graph_seed_topk,
        graph_neighbor_topk=graph_neighbor_topk,
        graph_max_expanded=graph_max_expanded,
        multimodal_metadata_enabled=multimodal_metadata_enabled,
        module_logs=module_logs,
    )

