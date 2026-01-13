from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import optuna

from .base import SearchInput, SearchOutput, Trial, best_trial, evaluate


@dataclass
class TPESearch:
    name: str = "tpe"

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def run(self, inp: SearchInput) -> SearchOutput:
        if int(inp.budget) <= 0:
            return SearchOutput(algo=self.name, trials=[], best=None)
        if not inp.space:
            raise ValueError("TPESearch requires inp.space")

        space = {k: list(v) for k, v in inp.space.items()}

        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        trials: List[Trial] = []

        def _objective(trial: optuna.Trial) -> float:
            cfg: Dict = {}
            for k, choices in space.items():
                cfg[k] = trial.suggest_categorical(k, choices)
            tr = evaluate(inp, cfg)
            if tr is None:
                return -1.0
            trials.append(tr)
            return float(tr.reward)

        study.optimize(_objective, n_trials=int(inp.budget), show_progress_bar=False)

        return SearchOutput(algo=self.name, trials=trials, best=best_trial(trials))
