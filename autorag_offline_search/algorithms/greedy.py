from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .base import ObjectiveFn, Trial


@dataclass
class GreedyCoordinate:
    """
    Coordinate descent over discrete choices.
    Each trial evaluates one candidate config.
    """

    name: str = "greedy"

    def __init__(self, search_space: Dict[str, Sequence], seed: int = 42):
        self.space = {k: list(v) for k, v in search_space.items()}
        self.rng = random.Random(seed)

    def _random_init(self) -> Dict:
        return {k: self.rng.choice(v) for k, v in self.space.items()}

    def search(self, *, objective: ObjectiveFn, budget: int) -> List[Trial]:
        if budget <= 0:
            return []
        trials: List[Trial] = []

        cur = self._random_init()
        best = objective(cur)
        trials.append(best)

        keys = list(self.space.keys())
        ki = 0
        while len(trials) < budget:
            k = keys[ki % len(keys)]
            ki += 1

            base = dict(best.config)
            candidates: List[Dict] = []
            for v in self.space[k]:
                cand = dict(base)
                cand[k] = v
                # validity guard: overlap < chunk_size if both exist
                if "chunk_overlap" in cand and "chunk_size" in cand:
                    if int(cand["chunk_overlap"]) >= int(cand["chunk_size"]):
                        continue
                candidates.append(cand)

            # evaluate candidates (but respect remaining budget)
            for cand in candidates:
                if len(trials) >= budget:
                    break
                tr = objective(cand)
                trials.append(tr)
                if tr.reward > best.reward:
                    best = tr

        return trials


