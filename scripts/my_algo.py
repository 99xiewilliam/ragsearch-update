from dataclasses import dataclass
from autorag_offline_search.algorithms.base import SearchInput, SearchOutput, best_trial

@dataclass
class MyAlgo:
    name: str = "my_algo"

    def run(self, inp: SearchInput) -> SearchOutput:
        # 例：只跑 5 次，从候选 configs 里随便取
        trials = []
        for cfg in (inp.configs or [])[: min(5, inp.budget)]:
            tr = inp.objective(cfg)
            trials.append(tr)
        return SearchOutput(algo=self.name, trials=trials, best=best_trial(trials))