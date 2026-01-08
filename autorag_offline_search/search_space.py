from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class SearchSpace:
    chunk_size: Sequence[int] = (64, 256, 512)
    chunk_overlap: Sequence[int] = (0, 32)
    retriever_topk: Sequence[int] = (1, 3, 5, 10)
    generator: Sequence[str] = ("best_sentence", "concat_chunks")

    def all_configs(self) -> List[Dict]:
        keys = ["chunk_size", "chunk_overlap", "retriever_topk", "generator"]
        values = [self.chunk_size, self.chunk_overlap, self.retriever_topk, self.generator]
        out: List[Dict] = []
        for combo in itertools.product(*values):
            cfg = dict(zip(keys, combo))
            # simple validity: overlap < chunk_size
            if int(cfg["chunk_overlap"]) >= int(cfg["chunk_size"]):
                continue
            out.append(cfg)
        return out


def config_to_str(cfg: Dict) -> str:
    return ",".join(f"{k}={cfg[k]}" for k in sorted(cfg.keys()))


@dataclass(frozen=True)
class BinarySpace:
    """
    Build an *exact* 2^bits search space by using only binary hyper-parameters.

    This is for your scaling experiments: bits in {4,6,8,10,12,14}.
    """

    bits: int

    def fixed_params(self) -> Dict:
        """Parameters that are fixed for all experiments to ensure LLM usage and GPU."""
        return {
            # --- RAG pipeline required params (must exist even for low bits) ---
            # For small bits (e.g. 4/6/8) the binary knobs may not include these keys yet,
            # but the baseline RagPipeline expects them unconditionally.
            "chunk_size": 256,
            "chunk_overlap": 0,
            "retriever_topk": 10,
            "generator": "best_sentence",
            "min_chunk_words": 4,

            "llm_backend": "openai_compat",
            "llm_model": "/home/xwh/models/Qwen3-4B-Instruct-2507",
            "embedder_device": "cuda",
            "reranker_device": "cuda",
            "verbose": True,
        }

    def space_dict(self) -> Dict[str, Sequence]:
        # Exactly 14 binary knobs to create a 2^14 search space.
        knobs: List[Tuple[str, Tuple]] = [
            # --- LLM Generation Knobs (4 bits) ---
            ("prompt_id", ("factoid_short", "evidence_then_answer")),
            ("llm_temperature", (0.0, 0.7)),
            ("llm_max_tokens", (128, 512)),
            ("llm_context_chunks", (4, 10)),  # How many retrieved chunks to feed the LLM

            # --- Chunking Knobs (4 bits) ---
            ("chunker", ("fixed_window", "semantic_split")),
            ("chunk_size", (256, 512)),
            ("chunk_overlap", (0, 64)),
            ("min_chunk_words", (4, 16)),

            # --- Retrieval Knobs (3 bits) ---
            ("retriever", ("vector_topk", "hybrid")),
            ("retriever_topk", (10, 40)),
            ("hybrid_alpha", (0.3, 0.7)),

            # --- Reranking & Embedding Knobs (3 bits) ---
            ("reranker", ("none", "cross_encoder")),
            ("rerank_topk", (5, 15)),
            ("embedder_model", ("BAAI/bge-m3", "intfloat/e5-large-v2")),
        ]
        if self.bits > len(knobs):
            raise ValueError(f"bits={self.bits} exceeds available binary knobs ({len(knobs)})")
        return {k: v for k, v in knobs[: self.bits]}

    def all_configs(self) -> List[Dict]:
        sd = self.space_dict()
        fixed = self.fixed_params()
        keys = list(sd.keys())
        values = [sd[k] for k in keys]
        out: List[Dict] = []
        for combo in itertools.product(*values):
            cfg = dict(zip(keys, combo))
            # Merge fixed params
            full_cfg = {**fixed, **cfg}

            # validity guard: overlap < chunk_size if both exist
            if "chunk_overlap" in full_cfg and "chunk_size" in full_cfg:
                if int(full_cfg["chunk_overlap"]) >= int(full_cfg["chunk_size"]):
                    continue
            out.append(full_cfg)
        return out


