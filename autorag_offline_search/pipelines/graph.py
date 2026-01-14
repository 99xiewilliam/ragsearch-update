from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Sequence, Tuple, Optional

from ..types import Doc
from .common import CommonRagPipeline
from .config import normalize_config
from ..modules.logging_utils import log, log_kv


def _keywords(text: str) -> List[str]:
    # simple, cheap keyword extraction for graph edges
    ws = (text or "").lower().split()
    return [w for w in ws if 3 <= len(w) <= 20]


class GraphRagPipeline(CommonRagPipeline):
    """
    GraphRAG:
    build an explicit edge index (graph) on top of chunk/content indexing, then do
    graph-aware retrieval via bounded multi-hop expansion (global/local/hybrid).
    """

    def __init__(self, docs: Sequence[Doc], config: Dict):
        super().__init__(docs, config)
        # ensure config says graph (useful for traces)
        _ = normalize_config({**(config or {}), "pipeline": "graph"})
        self._graph = self._build_graph()

    def _build_graph(self) -> Dict[int, List[int]]:
        """
        Build a lightweight chunk graph (edge index).

        graph_edge_source:
        - provided : load edges from graph_edges_path (by doc_id or by chunk index)
        - structure: connect adjacent chunks within the same doc_id (pos +/- 1)
        - knn      : edges by embedding nearest neighbors (uses the existing vector index)
        - keyword  : edges by shared keywords (cheap, but noisy)
        """
        src = str(getattr(self.cfg, "graph_edge_source", "keyword"))
        if src == "provided":
            return self._build_graph_provided()
        if src == "structure":
            return self._build_graph_structure()
        if src == "knn":
            return self._build_graph_knn()
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

    def _build_graph_structure(self) -> Dict[int, List[int]]:
        """
        Structure graph: within each doc_id, link adjacent chunk positions.
        Works for HotpotQA/MultiHopRAG style corpora where same document produces multiple chunks.
        """
        by_doc: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # doc_id -> [(pos, idx)]
        for i, (did, pos) in enumerate(zip(self._chunk_doc_ids, self._chunk_pos)):
            by_doc[str(did or "")].append((int(pos), int(i)))
        g: Dict[int, List[int]] = {i: [] for i in range(len(self._chunk_texts))}
        for _did, items in by_doc.items():
            items.sort()
            for j in range(len(items)):
                _pos, idx = items[j]
                if j - 1 >= 0:
                    g[idx].append(items[j - 1][1])
                if j + 1 < len(items):
                    g[idx].append(items[j + 1][1])
        log(self.cfg, f"[GRAPH:build_graph] source=structure nodes={len(g)}")
        return g

    def _build_graph_provided(self) -> Dict[int, List[int]]:
        """
        Load edges from a JSONL file at cfg.graph_edges_path.
        Each line supports either:
          {"src": "<doc_id>", "dst": "<doc_id>"}  (recommended)
        or:
          {"src_i": 12, "dst_i": 34}
        """
        path = str(getattr(self.cfg, "graph_edges_path", "") or "").strip()
        if not path:
            log(self.cfg, "[GRAPH:build_graph] source=provided but graph_edges_path empty; fallback=keyword")
            return self._build_graph()  # will fall through to keyword

        id2i = {str(did): i for i, did in enumerate(self._chunk_doc_ids) if str(did)}
        g: Dict[int, List[int]] = defaultdict(list)
        try:
            import json

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    si: Optional[int] = None
                    di: Optional[int] = None
                    if "src_i" in o and "dst_i" in o:
                        try:
                            si = int(o["src_i"])
                            di = int(o["dst_i"])
                        except Exception:
                            si = None
                            di = None
                    else:
                        s = str(o.get("src", "") or "").strip()
                        d = str(o.get("dst", "") or "").strip()
                        si = id2i.get(s)
                        di = id2i.get(d)
                    if si is None or di is None:
                        continue
                    g[int(si)].append(int(di))
        except Exception as e:
            log(self.cfg, f"[GRAPH:build_graph] source=provided failed ({type(e).__name__}: {e}); fallback=keyword")
            return self._build_graph()

        out = {i: sorted(list(set(g.get(i, [])))) for i in range(len(self._chunk_texts))}
        log(self.cfg, f"[GRAPH:build_graph] source=provided edges_file={path} nodes={len(out)}")
        return out

    def _build_graph_knn(self) -> Dict[int, List[int]]:
        if self._vector_index is None or self._chunk_emb_raw is None:
            # Can't build kNN without embeddings; fall back to keyword graph.
            log(self.cfg, "[GRAPH:build_graph] knn requested but embeddings unavailable; fallback=keyword")
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

        k = max(1, int(getattr(self.cfg, "graph_neighbor_topk", 50)))
        g = {}
        for i in range(int(self._chunk_emb_raw.shape[0])):
            # Query the existing index for nearest neighbors of chunk i.
            neigh = self._vector_index.query(self._chunk_emb_raw[i], topk=k + 1)
            out = []
            for j, _score in neigh:
                if int(j) == int(i):
                    continue
                out.append(int(j))
                if len(out) >= k:
                    break
            g[int(i)] = out
        log(self.cfg, f"[GRAPH:build_graph] source=knn k={k} nodes={len(g)}")
        return g

    def answer(self, query: str) -> str:
        # GraphRAG: base retrieve (global) -> graph expand (local/hybrid) -> rerank -> prune -> generate
        trace = self.answer_with_trace(query)
        if trace.get("error"):
            return str(trace.get("answer", "") or "")
        return str(trace.get("answer", "") or "")

    def answer_with_trace(self, query: str) -> Dict:
        """
        Graph-aware trace:
        - retrieved_indices: base retrieval
        - expanded_indices: retrieval expanded by multi-hop neighbors (bounded)
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
            log(self.cfg, "[GRAPH] ===== answer_with_trace() =====")
            log_kv(self.cfg, prefix="[GRAPH] ", key="query", value=query, limit=200)
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
            log_kv(self.cfg, prefix="[GRAPH:rewriter] ", key="rewritten_query", value=q, limit=200)

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
            log(self.cfg, f"[GRAPH:retriever] retrieved_indices={base_idx[:10]} (n={len(base_idx)})")
            if not base_idx:
                out["answer"] = ""
                return out

            # LightRAG-style modes (global/local/hybrid) over an explicit edge index.
            mode = str(getattr(self.cfg, "graph_mode", "hybrid"))
            seed_topk = int(getattr(self.cfg, "graph_seed_topk", min(3, int(self.cfg.retriever_topk))))
            nb_topk = int(getattr(self.cfg, "graph_neighbor_topk", 50))
            max_expanded = int(getattr(self.cfg, "graph_max_expanded", max(200, int(self.cfg.retriever_topk) * 10)))
            hops = int(getattr(self.cfg, "graph_hops", 1))

            base_idx_int = [int(i) for i in base_idx]
            local_seeds = base_idx_int[: max(0, min(seed_topk, len(base_idx_int)))]

            def _expand_multihop(seeds: List[int]) -> Tuple[List[int], Dict[int, int]]:
                """
                BFS-like expansion with bounded neighbor fanout. Returns (expanded_nodes, parent_map)
                where parent_map[child] = parent for path reconstruction (first time seen).
                """
                expanded: List[int] = []
                seen = set()
                parent: Dict[int, int] = {}
                dq = deque([(int(s), 0) for s in seeds])
                for s in seeds:
                    if s not in seen:
                        seen.add(int(s))
                        expanded.append(int(s))
                while dq and len(expanded) < max_expanded:
                    node, depth = dq.popleft()
                    if int(depth) >= int(hops):
                        continue
                    neigh = self._graph.get(int(node), [])[:nb_topk]
                    for nb in neigh:
                        nb = int(nb)
                        if nb in seen:
                            continue
                        seen.add(nb)
                        parent[nb] = int(node)
                        expanded.append(nb)
                        if len(expanded) >= max_expanded:
                            break
                        dq.append((nb, int(depth) + 1))
                    if len(expanded) >= max_expanded:
                        break
                return expanded, parent

            if (not self.cfg.graph_expand_enabled) or mode == "global":
                expanded = base_idx_int
                out["expanded_indices"] = expanded
            else:
                local_pool, parent = _expand_multihop(local_seeds)
                if mode == "local":
                    expanded = local_pool
                else:
                    # hybrid: union(local_pool, base_idx)
                    expanded = list(local_pool)
                    seen = set(expanded)
                    for i in base_idx_int:
                        if i not in seen:
                            seen.add(i)
                            expanded.append(i)
                out["expanded_indices"] = expanded
                # keep a few reasoning paths for inspection
                paths_n = int(getattr(self.cfg, "graph_trace_paths_n", 20))
                paths = []
                for leaf in expanded[: max(0, min(paths_n, len(expanded)))]:
                    cur = int(leaf)
                    path = [cur]
                    while cur in parent:
                        cur = int(parent[cur])
                        path.append(cur)
                        if len(path) > int(hops) + 1:
                            break
                    paths.append(list(reversed(path)))
                out["paths"] = paths

            log(
                self.cfg,
                f"[GRAPH:expand] enabled={self.cfg.graph_expand_enabled} mode={mode} source={getattr(self.cfg,'graph_edge_source','keyword')} "
                f"hops={hops} seed_topk={seed_topk} nb_topk={nb_topk} max_expanded={max_expanded} expanded_n={len(expanded)}",
            )

            # small preview (avoid dumping everything)
            preview_n = int(getattr(self.cfg, "graph_trace_preview_n", 20))
            out["expanded"] = [
                {
                    "i": int(i),
                    "doc_id": self._chunk_doc_ids[i] if i < len(self._chunk_doc_ids) else "",
                    "pos": int(self._chunk_pos[i]) if i < len(self._chunk_pos) else 0,
                    "text": self._chunk_texts[i],
                }
                for i in expanded[: max(0, min(preview_n, len(expanded)))]
            ]

            docs = [self._chunk_texts[i] for i in expanded]
            if self.cfg.reranker_enabled:
                # rerank within expanded pool (returns indices into `docs`)
                ridx = rerank(model_name=self.cfg.reranker_model, query=q, docs=docs, topk=self.cfg.rerank_topk)
                out["reranked_indices"] = ridx
                log(self.cfg, f"[GRAPH:reranker] enabled reranked_indices={ridx[:10]} (n={len(ridx)})")
                final = [docs[i] for i in ridx]
            else:
                out["reranked_indices"] = []
                log(self.cfg, "[GRAPH:reranker] disabled")
                final = docs

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
            log(self.cfg, f"[GRAPH:pruner] final_chunks={len(final)}")

            ctx = "\n\n---\n\n".join(final)
            ans = generate_answer(
                query=q0,
                context=ctx,
                cfg=GeneratorConfig(model=resolve_model(self.cfg.generator_model), max_tokens=self.cfg.generator_max_tokens),
                llm_base_url=self.cfg.llm_base_url,
            )
            out["answer"] = ans
            log_kv(self.cfg, prefix="[GRAPH:generator] ", key="answer", value=ans, limit=240)
            return out
        except Exception as e:
            out["error"] = repr(e)
            out.setdefault("answer", "")
            return out

