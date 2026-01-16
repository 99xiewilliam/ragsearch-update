from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from .base import SearchInput, SearchOutput, Trial, best_trial, evaluate


@dataclass
class ThompsonSamplingGaussian:
    """连续 reward 的近似 TS（posterior ~ Normal(mean, 1/sqrt(n+1))）。"""

    name: str = "ts"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def run(self, inp: SearchInput) -> SearchOutput:
        if int(inp.budget) <= 0:
            return SearchOutput(algo=self.name, trials=[], best=None)
        if not inp.configs:
            raise ValueError("ThompsonSamplingGaussian requires inp.configs")

        configs = list(inp.configs)
        n_arms = len(configs)
        if n_arms == 0:
            return SearchOutput(algo=self.name, trials=[], best=None)

        counts = [0] * n_arms
        means = [0.0] * n_arms
        trials: List[Trial] = []

        max_attempts = max(10, int(inp.budget) * 50)
        attempts = 0
        while len(trials) < int(inp.budget) and attempts < max_attempts:
            attempts += 1
            samples = []
            for i in range(n_arms):
                mu = means[i]
                sigma = 1.0 / math.sqrt(counts[i] + 1.0)
                samples.append(self.rng.gauss(mu, sigma))
            arm = max(range(n_arms), key=lambda i: samples[i])

            tr = evaluate(inp, configs[arm])
            if tr is None:
                continue
            trials.append(tr)

            counts[arm] += 1
            n = counts[arm]
            means[arm] += (tr.reward - means[arm]) / n

        return SearchOutput(algo=self.name, trials=trials, best=best_trial(trials))
