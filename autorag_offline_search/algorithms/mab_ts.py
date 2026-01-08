from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

from .base import ObjectiveFn, Trial


@dataclass
class ThompsonSamplingGaussian:
    """
    Approximate Thompson Sampling for continuous rewards in [0,1]:
      posterior ~ Normal(mean, 1/(n+1))

    This is a pragmatic choice for offline comparisons (simple + no heavy priors).
    """

    name: str = "ts"

    def __init__(self, configs: List[Dict], seed: int = 42):
        self.configs = list(configs)
        self.rng = random.Random(seed)

    def search(self, *, objective: ObjectiveFn, budget: int) -> List[Trial]:
        if budget <= 0 or not self.configs:
            return []

        n_arms = len(self.configs)
        counts = [0] * n_arms
        means = [0.0] * n_arms
        trials: List[Trial] = []

        for t in range(budget):
            # sample a score for each arm
            samples = []
            for i in range(n_arms):
                mu = means[i]
                sigma = 1.0 / math.sqrt(counts[i] + 1.0)
                samples.append(self.rng.gauss(mu, sigma))
            arm = max(range(n_arms), key=lambda i: samples[i])

            tr = objective(self.configs[arm])
            trials.append(tr)

            # update mean incrementally
            counts[arm] += 1
            n = counts[arm]
            means[arm] += (tr.reward - means[arm]) / n

        return trials


