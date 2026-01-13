from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from autorag_offline_search.algorithms.base import SearchInput, SearchOutput, Trial, best_trial, evaluate


@dataclass
class ExampleAlgo:
    """
    开源友好示例：一个“随机采样 + 保留最优”的最小算法实现。

    特点：
    - 只依赖 SearchInput/SearchOutput（项目唯一算法接口）
    - 支持 validate/on_trial（通过 evaluate() 统一处理）
    - 使用 inp.configs（预枚举 configs）做候选池
    """

    name: str = "example_algo"

    def run(self, inp: SearchInput) -> SearchOutput:
        if int(inp.budget) <= 0:
            return SearchOutput(algo=self.name, trials=[], best=None)
        if not inp.configs:
            raise ValueError("ExampleAlgo requires inp.configs (pre-enumerated configs)")

        rng = random.Random(int(inp.seed))
        trials: List[Trial] = []

        # 尝试次数上限：避免 validate 太严格导致死循环
        max_attempts = max(10, int(inp.budget) * 50)
        attempts = 0
        while len(trials) < int(inp.budget) and attempts < max_attempts:
            attempts += 1
            cfg: Dict = rng.choice(inp.configs)
            tr = evaluate(inp, cfg)
            if tr is None:
                continue
            trials.append(tr)

        return SearchOutput(
            algo=self.name,
            trials=trials,
            best=best_trial(trials),
            meta={"attempts": attempts, "pool_size": len(inp.configs)},
        )

