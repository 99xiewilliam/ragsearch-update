from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import optuna

from .base import SearchInput, SearchOutput, Trial, best_trial, evaluate


@dataclass
class TPESearch:
    name: str = "tpe"

    def __init__(
        self,
        seed: int = 42,
        *,
        patience: int = 0,
        min_delta: float = 0.0,
        warmup: int = 0,
    ):
        self.seed = int(seed)
        # Early stopping (disabled by default):
        # - patience: stop if best reward doesn't improve by >= min_delta for `patience` valid trials
        # - warmup  : don't early-stop until at least `warmup` valid trials are collected
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup = int(warmup)

    def run(self, inp: SearchInput) -> SearchOutput:
        if int(inp.budget) <= 0:
            return SearchOutput(algo=self.name, trials=[], best=None)
        if not inp.space:
            raise ValueError("TPESearch requires inp.space")

        space = {k: list(v) for k, v in inp.space.items()}

        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        trials: List[Trial] = []
        best_reward = float("-inf")
        last_improve_at = 0  # 1-based index of valid trial when best improved
        early_stopped = False
        # NOTE: We use ask/tell instead of study.optimize so that "invalid under validate()"
        # does not silently reduce the number of collected Trial objects below budget.
        max_attempts = max(10, int(inp.budget) * 50)
        attempts = 0
        while len(trials) < int(inp.budget) and attempts < max_attempts:
            attempts += 1
            ot = study.ask()
            cfg: Dict = {}
            for k, choices in space.items():
                cfg[k] = ot.suggest_categorical(k, choices)
            tr = evaluate(inp, cfg)
            if tr is None:
                # Still tell Optuna so the sampler can adapt, but don't count it toward budget.
                study.tell(ot, -1.0)
                continue
            trials.append(tr)
            study.tell(ot, float(tr.reward))

            # Early stop check (valid trials only)
            t_idx = len(trials)  # 1-based
            if tr.reward >= best_reward + max(0.0, self.min_delta):
                best_reward = float(tr.reward)
                last_improve_at = t_idx
            if self.patience > 0 and t_idx >= max(0, self.warmup):
                if (t_idx - last_improve_at) >= self.patience:
                    early_stopped = True
                    break

        return SearchOutput(
            algo=self.name,
            trials=trials,
            best=best_trial(trials),
            meta={
                "attempts": int(attempts),
                "max_attempts": int(max_attempts),
                "early_stopped": bool(early_stopped),
                "patience": int(self.patience),
                "min_delta": float(self.min_delta),
                "warmup": int(self.warmup),
            },
        )
