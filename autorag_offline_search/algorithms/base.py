from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol


@dataclass
class Trial:
    config: Dict
    reward: float
    metrics: Dict
    seconds: float


ObjectiveFn = Callable[[Dict], Trial]


class SearchAlgo(Protocol):
    name: str

    def search(self, *, objective: ObjectiveFn, budget: int) -> List[Trial]:
        ...


