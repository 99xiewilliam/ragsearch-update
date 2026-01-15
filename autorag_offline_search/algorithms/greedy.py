from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .base import SearchInput, SearchOutput, Trial, best_trial, evaluate


@dataclass
class GreedyCoordinate:
    """离散坐标下降（每次沿一个维度枚举，挑提升 reward 的候选）。"""

    name: str = "greedy"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _random_init(self, space: Dict[str, Sequence]) -> Dict:
        return {k: self.rng.choice(list(v)) for k, v in space.items()}

    def run(self, inp: SearchInput) -> SearchOutput:
        if int(inp.budget) <= 0:
            return SearchOutput(algo=self.name, trials=[], best=None)
        if not inp.space:
            raise ValueError("GreedyCoordinate requires inp.space")

        space = {k: list(v) for k, v in inp.space.items()}
        trials: List[Trial] = []

        # Random init can be rejected by validate(); retry a few times for robustness.
        tr0 = None
        max_attempts = 50
        for _ in range(max_attempts):
            cur = self._random_init(space)
            tr0 = evaluate(inp, cur)
            if tr0 is not None:
                break
        if tr0 is None:
            return SearchOutput(algo=self.name, trials=[], best=None)

        best = tr0
        trials.append(best)

        keys = list(space.keys())
        ki = 0
        while len(trials) < int(inp.budget):
            k = keys[ki % len(keys)]
            ki += 1

            base = dict(best.config)
            candidates: List[Dict] = []
            for v in space[k]:
                cand = dict(base)
                cand[k] = v
                candidates.append(cand)

            for cand in candidates:
                if len(trials) >= int(inp.budget):
                    break
                tr = evaluate(inp, cand)
                if tr is None:
                    continue
                trials.append(tr)
                if tr.reward > best.reward:
                    best = tr

        return SearchOutput(algo=self.name, trials=trials, best=best)
