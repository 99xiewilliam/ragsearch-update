#!/usr/bin/env python3
"""
Analyze bits14 runs: summarize per-algo train reward distribution and val reward,
and show where dumped generations are saved.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class AlgoStats:
    n: int
    mean: float
    std: float
    min: float
    max: float


def _stats(xs: List[float]) -> AlgoStats:
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return AlgoStats(n=0, mean=0.0, std=0.0, min=0.0, max=0.0)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / max(1, n)
    std = var**0.5
    return AlgoStats(n=n, mean=mean, std=std, min=min(xs), max=max(xs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_out_dir", type=str, required=True, help="e.g. /home/xwh/autorag_offline_search_runs/<run>/")
    args = ap.parse_args()

    sweep_out = args.sweep_out_dir.rstrip("/")
    sweep_json = os.path.join(sweep_out, "sweep.json")
    if not os.path.exists(sweep_json):
        raise SystemExit(f"missing sweep.json: {sweep_json}")

    with open(sweep_json, "r", encoding="utf-8") as f:
        sj = json.load(f)

    rows = sj.get("rows", [])
    # focus on bits14 only
    rows = [r for r in rows if int(r.get("bits", 0)) == 14]
    # group by (dataset, algo)
    by: Dict[Tuple[str, str], Dict] = {}
    for r in rows:
        key = (r["dataset"], r["algo"])
        by[key] = r

    datasets = sorted({d for (d, _) in by.keys()})
    algos = sorted({a for (_, a) in by.keys()})

    print(f"sweep_out_dir={sweep_out}")
    print("datasets:", ", ".join(datasets))
    print("algos:", ", ".join(algos))
    print("---")

    for ds in datasets:
        print(f"[dataset={ds} bits=14]")
        for algo in algos:
            r = by.get((ds, algo))
            if not r:
                continue
            run_dir = r.get("run_dir")
            # per-algo results live in <run_dir>/results.json (dataset/bits14)
            results_path = os.path.join(run_dir, "results.json")
            if not os.path.exists(results_path):
                print(f"  - {algo}: missing results.json at {results_path}")
                continue
            with open(results_path, "r", encoding="utf-8") as f:
                res = json.load(f)
            trials = (res.get("results", {}).get(algo, {}) or {}).get("train_trials", []) or []
            train_rewards = [float(t.get("reward", 0.0)) for t in trials]
            st = _stats(train_rewards)
            v = (res.get("results", {}).get(algo, {}) or {}).get("validation", {}) or {}
            val_reward = float(v.get("reward", 0.0))
            dump_path = os.path.join(run_dir, f"val_predictions_{algo}.jsonl")
            dump_note = dump_path if os.path.exists(dump_path) else "(no dump file)"
            print(
                f"  - {algo}: "
                f"train(n={st.n} mean={st.mean:.4f} std={st.std:.4f} min={st.min:.4f} max={st.max:.4f}) "
                f"val_reward={val_reward:.6f} "
                f"dump={dump_note}"
            )
        print()


if __name__ == "__main__":
    main()










