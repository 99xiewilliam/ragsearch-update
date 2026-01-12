from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from ..types import Doc
from .common import CommonRagPipeline
from .config import normalize_config


def _keywords(text: str) -> List[str]:
    # simple, cheap keyword extraction for graph edges
    ws = (text or "").lower().split()
    return [w for w in ws if 3 <= len(w) <= 20]


class GraphRagPipeline(CommonRagPipeline):
    """
    Minimal GraphRAG skeleton:
    build a chunk-level graph (edges by shared keywords) and expand retrieval by 1 hop.
    No additional hyper-parameters are introduced (by user request).
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        super().__init__(docs, config)
        # ensure config says graph (useful for traces)
        _ = normalize_config({**(config or {}), "pipeline": "graph"})
        self._graph = self._build_graph()

    def _build_graph(self) -> Dict[int, List[int]]:
        inv = defaultdict(list)
        for i, t in enumerate(self._chunk_texts):
            for k in set(_keywords(t)):
                inv[k].append(i)

        g = defaultdict(set)
        for ids in inv.values():
            if len(ids) <= 1:
                continue
            for a in ids:
                for b in ids:
                    if a != b:
                        g[a].add(b)
        return {k: sorted(list(v)) for k, v in g.items()}

    def answer(self, query: str) -> str:
        # GraphRAG: base retrieve -> 1-hop expand -> rerank -> prune -> generate
        trace = self.answer_with_trace(query)
        if trace.get("error"):
            return str(trace.get("answer", "") or "")
        return str(trace.get("answer", "") or "")

    def answer_with_trace(self, query: str) -> Dict:
        """
        Graph-aware trace:
        - retrieved_indices: base retrieval
        - expanded_indices: retrieval expanded by 1-hop neighbors
        - reranked_indices: rerank indices within expanded list
        """
        from ..modules.embedding import embed_query
        from ..modules.generator_fixed import GeneratorConfig, generate_answer
        from ..modules.pruner import PrunerConfig, prune_chunks
        from ..modules.retrieval import RetrievalConfig, retrieve_indices
        from ..modules.reranking import rerank
        from ..modules.rewriter import RewriterConfig, rewrite_query
        from .config import resolve_model

        out: Dict = {"query": query, "pipeline": "graph"}
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
                import numpy as np

                q_emb = np.asarray(embed_query(self.cfg.embedder_model, q), dtype=np.float32)

            base_idx = retrieve_indices(
                cfg=RetrievalConfig(method=self.cfg.retriever, topk=self.cfg.retriever_topk, hybrid_alpha=self.cfg.hybrid_alpha),
                query=q,
                vector_index=self._vector_index,
                query_emb=q_emb,
                bm25=self._bm25,
            )
            out["retrieved_indices"] = base_idx
            if not base_idx:
                out["answer"] = ""
                return out

            # 1-hop expansion (bounded)
            if self.cfg.graph_expand_enabled:
                expanded: List[int] = []
                seen = set()
                for i in base_idx:
                    ii = int(i)
                    if ii not in seen:
                        seen.add(ii)
                        expanded.append(ii)
                    for nb in self._graph.get(ii, [])[:50]:
                        if nb not in seen:
                            seen.add(nb)
                            expanded.append(nb)
                    if len(expanded) >= max(200, int(self.cfg.retriever_topk) * 10):
                        break
            else:
                expanded = [int(i) for i in base_idx]
            out["expanded_indices"] = expanded

            docs = [self._chunk_texts[i] for i in expanded]
            # rerank within expanded pool (returns indices into `docs`)
            ridx = rerank(model_name=self.cfg.reranker_model, query=q, docs=docs, topk=self.cfg.rerank_topk)
            out["reranked_indices"] = ridx
            final = [docs[i] for i in ridx]

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
            out["final_chunks"] = final[: min(10, len(final))]

            ctx = "\n\n---\n\n".join(final)
            ans = generate_answer(
                query=q0,
                context=ctx,
                cfg=GeneratorConfig(model=resolve_model(self.cfg.generator_model), max_tokens=self.cfg.generator_max_tokens),
                llm_base_url=self.cfg.llm_base_url,
            )
            out["answer"] = ans
            return out
        except Exception as e:
            out["error"] = repr(e)
            out.setdefault("answer", "")
            return out

