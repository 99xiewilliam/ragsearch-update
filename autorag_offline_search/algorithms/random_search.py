from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from .base import ObjectiveFn, Trial


@dataclass
class RandomSearch:
    name: str = "random"

    def __init__(self, configs: List[Dict], seed: int = 42):
        self.configs = list(configs)
        self.rng = random.Random(seed)

    def search(self, *, objective: ObjectiveFn, budget: int) -> List[Trial]:
        if not self.configs:
            return []
        trials: List[Trial] = []
        for _ in range(budget):
            cfg = self.rng.choice(self.configs)
            trials.append(objective(cfg))
        return trials


