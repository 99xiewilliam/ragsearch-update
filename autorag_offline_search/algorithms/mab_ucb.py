from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

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
        # Track arms that are permanently invalid under validate() to avoid repeated selection.
        invalid = [False] * n_arms
        trials: List[Trial] = []

        max_attempts = max(10, int(inp.budget) * 50)
        attempts = 0

        # initial pull each arm once (best-effort) if budget allows
        i = 0
        while len(trials) < int(inp.budget) and i < n_arms and attempts < max_attempts:
            attempts += 1
            tr = evaluate(inp, configs[i])
            if tr is None:
                # Mark arm as invalid and skip
                invalid[i] = True
                i += 1
                continue
            trials.append(tr)
            counts[i] += 1
            sums[i] += tr.reward
            i += 1

        while len(trials) < int(inp.budget) and attempts < max_attempts:
            attempts += 1
            ucb_vals = []
            for i in range(n_arms):
                if invalid[i]:
                    # Permanently invalid arm: assign -inf so it's never selected
                    ucb_vals.append(float("-inf"))
                elif counts[i] == 0:
                    ucb_vals.append(float("inf"))
                else:
                    mean = sums[i] / counts[i]
                    t = max(2, len(trials))
                    bonus = math.sqrt(2.0 * math.log(t) / counts[i])
                    ucb_vals.append(mean + bonus)

            # If all arms are invalid, stop early
            if all(v == float("-inf") for v in ucb_vals):
                break

            arm = max(range(n_arms), key=lambda i: ucb_vals[i])
            tr = evaluate(inp, configs[arm])
            if tr is None:
                # Mark arm as invalid so it won't be selected again
                invalid[arm] = True
                continue
            trials.append(tr)
            counts[arm] += 1
            sums[arm] += tr.reward

        return SearchOutput(algo=self.name, trials=trials, best=best_trial(trials))
