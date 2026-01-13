from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from .base import SearchInput, SearchOutput, Trial, best_trial, evaluate


@dataclass
class UCB1Bandit:
    """把每个完整 config 视为一个 arm，用 UCB1 选择评测。"""

    name: str = "ucb1"

    def run(self, inp: SearchInput) -> SearchOutput:
        if int(inp.budget) <= 0:
            return SearchOutput(algo=self.name, trials=[], best=None)
        if not inp.configs:
            raise ValueError("UCB1Bandit requires inp.configs")

        configs = list(inp.configs)
        n_arms = len(configs)
        if n_arms == 0:
            return SearchOutput(algo=self.name, trials=[], best=None)

        counts = [0] * n_arms
        sums = [0.0] * n_arms
        trials: List[Trial] = []

        # initial pull each arm once if budget allows
        for i in range(min(n_arms, int(inp.budget))):
            tr = evaluate(inp, configs[i])
            if tr is None:
                continue
            trials.append(tr)
            counts[i] += 1
            sums[i] += tr.reward

        t = len(trials)
        while t < int(inp.budget):
            ucb_vals = []
            for i in range(n_arms):
                if counts[i] == 0:
                    ucb_vals.append(float("inf"))
                else:
                    mean = sums[i] / counts[i]
                    bonus = math.sqrt(2.0 * math.log(max(2, t)) / counts[i])
                    ucb_vals.append(mean + bonus)

            arm = max(range(n_arms), key=lambda i: ucb_vals[i])
            tr = evaluate(inp, configs[arm])
            if tr is None:
                # invalid under validate(); try next iteration
                t += 1
                continue
            trials.append(tr)
            counts[arm] += 1
            sums[arm] += tr.reward
            t += 1

        return SearchOutput(algo=self.name, trials=trials, best=best_trial(trials))
