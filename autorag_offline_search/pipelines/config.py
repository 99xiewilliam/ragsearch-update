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
    pipeline: str  # "common" | "multimodal"
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
    # unified retrieval knob (bm25 weight in [0,1])
    bm25_weight: float
    # rerank
    reranker_enabled: bool
    reranker_model: str  # "none" or cross-encoder model name
    rerank_topk: int
    # pruner
    pruner_enabled: bool
    pruner_model: str
    pruner_prompt: str
    pruner_max_tokens: int
    pruner_mode: str  # "select" (choose chunks) or "compress" (compress each chunk)
    # generator (fixed prompt, but model can be configured as a "non-hyperparam")
    generator_model: str
    generator_max_tokens: int
    # multimodal
    multimodal_metadata_enabled: bool
    # logging
    module_logs: bool


def normalize_config(cfg: Dict) -> NormalizedConfig:
    c = dict(cfg or {})
    pipeline = str(c.get("pipeline", "common"))
    if pipeline not in {"common", "multimodal"}:
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
    # NOTE: rewriter_max_tokens is an *output* token budget for vLLM/OpenAI.
    # Many users confuse it with context length (e.g. 32768). We default to 32678 as a
    # sentinel meaning "no explicit max_tokens" (see OpenAICompatLLM.generate()).
    rewriter_max_tokens = int(c.get("rewriter_max_tokens", 32678))

    chunking_enabled = bool(c.get("chunking_enabled", True))
    chunking_method = str(c.get("chunking_method", "semantic"))
    if chunking_method not in {"semantic", "token"}:
        raise ValueError(f"Unknown chunking_method: {chunking_method}")
    chunk_size = int(c.get("chunk_size", 256))
    chunk_overlap = int(c.get("chunk_overlap", 0))
    min_chunk_words = int(c.get("min_chunk_words", 8))

    embedding_enabled = bool(c.get("embedding_enabled", True))
    embedder_model = str(c.get("embedder_model", "BAAI/bge-m3"))

    # Unified retrieval knob:
    # - bm25_weight=0   => cosine
    # - bm25_weight=1   => bm25 (auto-disable embedding)
    # - (0,1)           => hybrid, with hybrid_alpha=bm25_weight
    eps = 1e-6
    bm25_weight_raw = c.get("bm25_weight", None)
    if bm25_weight_raw is not None and str(bm25_weight_raw).strip() != "":
        bm25_weight = float(bm25_weight_raw)
        if not (0.0 <= bm25_weight <= 1.0):
            raise ValueError("bm25_weight must be in [0,1]")
        if bm25_weight <= eps:
            retriever = "cosine"
            hybrid_alpha = 0.5
            embedding_enabled = True if "embedding_enabled" not in c else embedding_enabled
            bm25_weight = 0.0
        elif bm25_weight >= 1.0 - eps:
            retriever = "bm25"
            hybrid_alpha = 0.5
            embedding_enabled = False
            bm25_weight = 1.0
        else:
            retriever = "hybrid"
            hybrid_alpha = float(bm25_weight)
            embedding_enabled = True if "embedding_enabled" not in c else embedding_enabled
    else:
        # Backward compatible: accept explicit retriever/hybrid_alpha.
        retriever = str(c.get("retriever", "cosine"))
        if retriever not in {"cosine", "bm25", "hybrid"}:
            raise ValueError(f"Unknown retriever: {retriever}")
        hybrid_alpha = float(c.get("hybrid_alpha", 0.5))
        if retriever == "cosine":
            bm25_weight = 0.0
        elif retriever == "bm25":
            bm25_weight = 1.0
            # If user explicitly chooses bm25, embedding isn't needed.
            embedding_enabled = False if "embedding_enabled" not in c else embedding_enabled
        else:
            # hybrid: interpret hybrid_alpha as bm25 weight
            if not (0.0 <= hybrid_alpha <= 1.0):
                raise ValueError("hybrid_alpha must be in [0,1] for hybrid retriever")
            bm25_weight = float(hybrid_alpha)

    retriever_topk = int(c.get("retriever_topk", 10))

    reranker_enabled = bool(c.get("reranker_enabled", True))
    reranker_model = str(c.get("reranker_model", "none"))
    rerank_topk = int(c.get("rerank_topk", min(10, retriever_topk)))

    pruner_enabled = bool(c.get("pruner_enabled", False))
    pruner_model = str(c.get("pruner_model", "qwen3"))
    pruner_mode = str(c.get("pruner_mode", "select")).strip().lower()
    if pruner_mode not in {"select", "compress"}:
        pruner_mode = "select"
    # pruner prompt: use compress_v1 if mode is compress, otherwise prune_v1
    if pruner_mode == "compress":
        pruner_prompt = str(PRUNER_PROMPTS.get("compress_v1", PRUNER_PROMPTS["prune_v1"]))
    else:
        # pruner prompt is fixed (not a hyperparam); keep it stable for fair comparisons.
        pruner_prompt = str(PRUNER_PROMPTS["prune_v1"])
    # Same sentinel convention as rewriter_max_tokens.
    pruner_max_tokens = int(c.get("pruner_max_tokens", 32678))

    # Generator prompt is fixed, but generator *model* can be overridden by YAML/CLI.
    # We make the default model pipeline-aware:
    # - common      : text LLM (qwen3)
    # - multimodal  : VL LLM (qwen3_vl_4b)
    if "generator_model" in c and str(c.get("generator_model") or "").strip():
        generator_model = str(c.get("generator_model"))
    else:
        generator_model = "qwen3_vl_4b" if pipeline == "multimodal" else "qwen3"
    # NOTE: generator_max_tokens is an *output* token budget for vLLM/OpenAI.
    # Use 32768 as a sentinel meaning "no explicit max_tokens" (see OpenAICompatLLM.generate()).
    generator_max_tokens = int(c.get("generator_max_tokens", 32768))
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
        bm25_weight=float(bm25_weight),
        reranker_enabled=reranker_enabled,
        reranker_model=reranker_model,
        rerank_topk=rerank_topk,
        pruner_enabled=pruner_enabled,
        pruner_model=pruner_model,
        pruner_prompt=pruner_prompt,
        pruner_max_tokens=pruner_max_tokens,
        pruner_mode=pruner_mode,
        generator_model=generator_model,
        generator_max_tokens=generator_max_tokens,
        multimodal_metadata_enabled=multimodal_metadata_enabled,
        module_logs=module_logs,
    )

