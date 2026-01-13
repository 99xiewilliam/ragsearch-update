from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence


@dataclass
class Trial:
    config: Dict
    reward: float
    metrics: Dict
    seconds: float
    # Optional extension fields (non-breaking for existing algorithms / results dumping)
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


ObjectiveFn = Callable[[Dict], Trial]


@dataclass(frozen=True)
class SearchInput:
    """
    Public input contract for search/optimization algorithms.

    The runtime (experiment) owns evaluation. Algorithms only propose configs under a budget.
    """

    objective: ObjectiveFn
    budget: int

    # determinism hint
    seed: int = 42

    # optional representations of the search domain
    space: Optional[Dict[str, Sequence]] = None
    configs: Optional[List[Dict]] = None

    # optional helpers
    validate: Optional[Callable[[Dict], bool]] = None
    on_trial: Optional[Callable[[Trial], None]] = None


@dataclass
class SearchOutput:
    """Public output contract for algorithms."""

    algo: str
    trials: List[Trial]
    best: Optional[Trial] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class SearchAlgorithm(Protocol):
    """New recommended algorithm interface."""

    name: str

    def run(self, inp: SearchInput) -> SearchOutput:
        ...


class SearchAlgo(Protocol):
    """Legacy interface kept for backward compatibility."""

    name: str

    def search(self, *, objective: ObjectiveFn, budget: int) -> List[Trial]:
        ...


def best_trial(trials: List[Trial]) -> Optional[Trial]:
    return max(trials, key=lambda t: t.reward) if trials else None


def run_algo(algo: object, inp: SearchInput) -> SearchOutput:
    """
    Adapter to support:
    - new style: algo.run(SearchInput) -> SearchOutput
    - legacy   : algo.search(objective=..., budget=...) -> List[Trial]
    """

    algo_name = getattr(algo, "name", algo.__class__.__name__)

    def _objective(cfg: Dict) -> Trial:
        tr = inp.objective(cfg)
        if inp.on_trial is not None:
            inp.on_trial(tr)
        return tr

    run = getattr(algo, "run", None)
    if callable(run):
        out = run(inp)
        if out.best is None:
            out.best = best_trial(out.trials)
        return out

    search = getattr(algo, "search", None)
    if callable(search):
        trials = search(objective=_objective, budget=int(inp.budget))
        return SearchOutput(algo=str(algo_name), trials=trials, best=best_trial(trials))

    raise TypeError(
        f"Algorithm {algo_name!r} does not implement run(SearchInput) or search(objective,budget)."
    )
