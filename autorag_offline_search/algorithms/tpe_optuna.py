from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import optuna

from .base import ObjectiveFn, Trial


@dataclass
class TPESearch:
    name: str = "tpe"

    def __init__(self, space: Dict[str, Sequence], seed: int = 42):
        self.space = {k: list(v) for k, v in space.items()}
        self.seed = seed

    def search(self, *, objective: ObjectiveFn, budget: int) -> List[Trial]:
        if budget <= 0:
            return []

        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        trials: List[Trial] = []

        def _objective(trial: optuna.Trial) -> float:
            cfg: Dict = {}
            for k, choices in self.space.items():
                cfg[k] = trial.suggest_categorical(k, choices)
            # validity guard
            if "chunk_overlap" in cfg and "chunk_size" in cfg:
                if int(cfg["chunk_overlap"]) >= int(cfg["chunk_size"]):
                    return -1.0
            tr = objective(cfg)
            trials.append(tr)
            return tr.reward

        study.optimize(_objective, n_trials=budget, show_progress_bar=False)
        return trials


