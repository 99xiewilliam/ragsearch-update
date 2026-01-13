from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence


@dataclass
class Trial:
    config: Dict
    reward: float
    metrics: Dict
    seconds: float
    # Optional extension fields
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


ObjectiveFn = Callable[[Dict], Trial]


@dataclass(frozen=True)
class SearchInput:
    """对外暴露的算法输入标准（新接口，唯一接口）。

    - objective(cfg)->Trial：评测函数（内部跑 RAG + metrics），每调用一次消耗 1 budget
    - budget：最多评测次数
    - seed：随机种子提示
    - space：离散搜索空间（给 greedy/TPE/GRPO 等）
    - configs：预枚举 configs（给 random/UCB/TS 等把 config 当 arm 的算法）
    - validate：可选合法性校验（返回 False 则算法应跳过该 cfg）
    - on_trial：可选回调（每产生一个 Trial 就调用）
    """

    objective: ObjectiveFn
    budget: int
    seed: int = 42
    space: Optional[Dict[str, Sequence]] = None
    configs: Optional[List[Dict]] = None
    validate: Optional[Callable[[Dict], bool]] = None
    on_trial: Optional[Callable[[Trial], None]] = None


@dataclass
class SearchOutput:
    """对外暴露的算法输出标准。"""

    algo: str
    trials: List[Trial]
    best: Optional[Trial] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class SearchAlgorithm(Protocol):
    name: str

    def run(self, inp: SearchInput) -> SearchOutput:
        ...


def best_trial(trials: List[Trial]) -> Optional[Trial]:
    return max(trials, key=lambda t: t.reward) if trials else None


def evaluate(inp: SearchInput, cfg: Dict) -> Optional[Trial]:
    """统一的 objective 调用入口：处理 validate + on_trial。"""

    if inp.validate is not None:
        try:
            if not bool(inp.validate(cfg)):
                return None
        except Exception:
            return None

    tr = inp.objective(cfg)
    if inp.on_trial is not None:
        inp.on_trial(tr)
    return tr
