from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple


@dataclass
class _Agg:
    seconds: float = 0.0
    calls: int = 0


class Timing:
    """
    Lightweight timing accumulator.

    - Use `with timing.timer("stage"):` to accumulate wall time.
    - `summary()` returns {stage: {seconds, calls, avg_ms}}.
    """

    def __init__(self) -> None:
        self._agg: Dict[str, _Agg] = {}

    @contextmanager
    def timer(self, key: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            a = self._agg.get(key)
            if a is None:
                a = _Agg()
                self._agg[key] = a
            a.seconds += float(dt)
            a.calls += 1

    def merge(self, other: "Timing") -> None:
        for k, a2 in other._agg.items():
            a1 = self._agg.get(k)
            if a1 is None:
                self._agg[k] = _Agg(seconds=float(a2.seconds), calls=int(a2.calls))
            else:
                a1.seconds += float(a2.seconds)
                a1.calls += int(a2.calls)

    def summary(self, *, reset: bool = False) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k, a in self._agg.items():
            calls = max(1, int(a.calls))
            out[k] = {
                "seconds": float(a.seconds),
                "calls": float(a.calls),
                "avg_ms": float(a.seconds) * 1000.0 / float(calls),
            }
        # stable-ish order: by total time desc, then name
        out = dict(sorted(out.items(), key=lambda kv: (-kv[1]["seconds"], kv[0])))
        if reset:
            self._agg.clear()
        return out

    def topk(self, k: int = 20) -> Tuple[Tuple[str, Dict[str, float]], ...]:
        s = self.summary(reset=False)
        items = list(s.items())[: max(0, int(k))]
        return tuple(items)

