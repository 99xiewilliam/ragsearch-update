from dataclasses import dataclass
from autorag_offline_search.algorithms.base import SearchInput, SearchOutput, best_trial

@dataclass
class MyAlgo:
    name: str = "my_algo"

    def run(self, inp: SearchInput) -> SearchOutput:
        """
        Demo algorithm:
        - Run up to `budget` trials
        - Prefer *diverse* configs (so demos show meaningful differences)

        NOTE: This is intentionally simple and deterministic.
        """
        trials = []
        used = set()
        cfgs = list(inp.configs or [])

        def sig(c: dict) -> tuple:
            # Keep signature focused on knobs that usually change runtime/behavior.
            # (Avoid keys that are pinned by base config and would lead to duplicates.)
            keys = (
                "pipeline",
                "chunking_enabled",
                "chunking_method",
                "chunk_size",
                "chunk_overlap",
                "min_chunk_words",
                "embedding_enabled",
                "retriever",
                "hybrid_alpha",
                "retriever_topk",
                "reranker_enabled",
                "rerank_topk",
            )
            return tuple(c.get(k) for k in keys)

        for cfg in cfgs:
            if len(trials) >= int(inp.budget):
                break
            s = sig(cfg)
            if s in used:
                continue
            used.add(s)
            tr = inp.objective(cfg)
            trials.append(tr)

        # If the space is tiny / too constrained, fill remaining budget with the next configs.
        if len(trials) < int(inp.budget):
            for cfg in cfgs:
                if len(trials) >= int(inp.budget):
                    break
                tr = inp.objective(cfg)
                trials.append(tr)

        return SearchOutput(algo=self.name, trials=trials, best=best_trial(trials))