from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .base import SearchInput, SearchOutput, Trial, best_trial, evaluate


@dataclass
class RandomSearch:
    name: str = "random"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def run(self, inp: SearchInput) -> SearchOutput:
        if inp.budget <= 0:
            return SearchOutput(algo=self.name, trials=[], best=None)
        if not inp.configs:
            raise ValueError("RandomSearch requires inp.configs (pre-enumerated configs)")

        trials: List[Trial] = []
        # Avoid infinite loops if validate rejects many configs.
        max_attempts = max(10, int(inp.budget) * 20)
        attempts = 0
        while len(trials) < int(inp.budget) and attempts < max_attempts:
            attempts += 1
            cfg = self.rng.choice(inp.configs)
            tr = evaluate(inp, cfg)
            if tr is None:
                continue
            trials.append(tr)

        return SearchOutput(algo=self.name, trials=trials, best=best_trial(trials))
