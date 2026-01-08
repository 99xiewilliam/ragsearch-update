from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from .base import ObjectiveFn, Trial


@dataclass
class UCB1Bandit:
    """
    Treat each full config as an arm. Select arms using UCB1.
    """

    name: str = "ucb1"

    def __init__(self, configs: List[Dict]):
        self.configs = list(configs)

    def search(self, *, objective: ObjectiveFn, budget: int) -> List[Trial]:
        if budget <= 0 or not self.configs:
            return []

        n_arms = len(self.configs)
        counts = [0] * n_arms
        sums = [0.0] * n_arms
        trials: List[Trial] = []

        # initial pull each arm once if budget allows
        for i in range(min(n_arms, budget)):
            tr = objective(self.configs[i])
            trials.append(tr)
            counts[i] += 1
            sums[i] += tr.reward

        t = len(trials)
        while t < budget:
            # select arm with max UCB
            ucb_vals = []
            for i in range(n_arms):
                if counts[i] == 0:
                    ucb_vals.append(float("inf"))
                else:
                    mean = sums[i] / counts[i]
                    bonus = math.sqrt(2.0 * math.log(max(2, t)) / counts[i])
                    ucb_vals.append(mean + bonus)
            arm = max(range(n_arms), key=lambda i: ucb_vals[i])
            tr = objective(self.configs[arm])
            trials.append(tr)
            counts[arm] += 1
            sums[arm] += tr.reward
            t += 1

        return trials


