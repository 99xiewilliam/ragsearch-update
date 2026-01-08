#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


def _read_tsv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _to_int(x: object) -> Optional[int]:
    try:
        return int(str(x))
    except Exception:
        return None


def _to_float(x: object) -> Optional[float]:
    try:
        v = float(str(x))
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    if p <= 0:
        return xs2[0]
    if p >= 100:
        return xs2[-1]
    k = (len(xs2) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs2[int(k)]
    d0 = xs2[f] * (c - k)
    d1 = xs2[c] * (k - f)
    return d0 + d1


@dataclass
class GroupRow:
    dataset: str
    bits: int
    old_grpo_val: Optional[float]
    new_grpo_val: Optional[float]
    delta_grpo: Optional[float]
    old_best_algo: Optional[str]
    old_best_val: Optional[float]
    new_best_algo: Optional[str]
    new_best_val: Optional[float]
    old_grpo_wins: Optional[bool]
    new_grpo_wins: Optional[bool]


def _index_by_dataset_bits(rows: List[Dict[str, str]]) -> Dict[Tuple[str, int], List[Dict[str, str]]]:
    out: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    for r in rows:
        ds = (r.get("dataset") or "").strip()
        b = _to_int(r.get("bits"))
        if not ds or b is None:
            continue
        out.setdefault((ds, b), []).append(r)
    return out


def _best_algo(group: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[float]]:
    best_a = None
    best_v = None
    for r in group:
        a = (r.get("algo") or "").strip()
        v = _to_float(r.get("val_reward"))
        if not a or v is None:
            continue
        if best_v is None or v > best_v:
            best_v = v
            best_a = a
    return best_a, best_v


def _algo_val(group: List[Dict[str, str]], algo: str) -> Optional[float]:
    algo = algo.strip()
    for r in group:
        if (r.get("algo") or "").strip() == algo:
            return _to_float(r.get("val_reward"))
    return None


def compare(*, old_path: str, new_path: str) -> Tuple[List[GroupRow], Dict]:
    old_rows = _read_tsv(old_path)
    new_rows = _read_tsv(new_path)
    old_idx = _index_by_dataset_bits(old_rows)
    new_idx = _index_by_dataset_bits(new_rows)

    keys = sorted(set(old_idx.keys()) | set(new_idx.keys()))
    out_rows: List[GroupRow] = []
    deltas: List[float] = []

    old_wins = 0
    new_wins = 0
    common = 0

    for k in keys:
        ds, bits = k
        og = old_idx.get(k, [])
        ng = new_idx.get(k, [])
        old_grpo = _algo_val(og, "grpo")
        new_grpo = _algo_val(ng, "grpo")
        if og and ng:
            common += 1
        oba, obv = _best_algo(og)
        nba, nbv = _best_algo(ng)
        ogw = (oba == "grpo") if oba is not None else None
        ngw = (nba == "grpo") if nba is not None else None
        if ogw is True:
            old_wins += 1
        if ngw is True:
            new_wins += 1
        d = None
        if old_grpo is not None and new_grpo is not None:
            d = new_grpo - old_grpo
            deltas.append(d)

        out_rows.append(
            GroupRow(
                dataset=ds,
                bits=bits,
                old_grpo_val=old_grpo,
                new_grpo_val=new_grpo,
                delta_grpo=d,
                old_best_algo=oba,
                old_best_val=obv,
                new_best_algo=nba,
                new_best_val=nbv,
                old_grpo_wins=ogw,
                new_grpo_wins=ngw,
            )
        )

    def _mean(xs: List[float]) -> float:
        return sum(xs) / max(1, len(xs))

    def _std(xs: List[float]) -> float:
        if len(xs) <= 1:
            return 0.0
        m = _mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    summary = {
        "old_path": old_path,
        "new_path": new_path,
        "groups_total": len(keys),
        "groups_common": common,
        "grpo_wins_old": old_wins,
        "grpo_wins_new": new_wins,
        "grpo_win_change": new_wins - old_wins,
        "grpo_delta_count": len(deltas),
        "grpo_delta_mean": _mean(deltas) if deltas else None,
        "grpo_delta_std": _std(deltas) if deltas else None,
        "grpo_delta_min": min(deltas) if deltas else None,
        "grpo_delta_max": max(deltas) if deltas else None,
        "grpo_delta_p10": _percentile(deltas, 10) if deltas else None,
        "grpo_delta_p50": _percentile(deltas, 50) if deltas else None,
        "grpo_delta_p90": _percentile(deltas, 90) if deltas else None,
        "grpo_delta_pos": sum(1 for x in deltas if x > 0),
        "grpo_delta_neg": sum(1 for x in deltas if x < 0),
        "grpo_delta_zero": sum(1 for x in deltas if x == 0),
    }
    return out_rows, summary


def _write_outputs(out_dir: str, rows: List[GroupRow], summary: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # detailed CSV
    csv_path = os.path.join(out_dir, "grpo_compare_by_group.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "bits",
                "old_grpo_val",
                "new_grpo_val",
                "delta_grpo",
                "old_best_algo",
                "old_best_val",
                "new_best_algo",
                "new_best_val",
                "old_grpo_wins",
                "new_grpo_wins",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.dataset,
                    r.bits,
                    r.old_grpo_val,
                    r.new_grpo_val,
                    r.delta_grpo,
                    r.old_best_algo,
                    r.old_best_val,
                    r.new_best_algo,
                    r.new_best_val,
                    r.old_grpo_wins,
                    r.new_grpo_wins,
                ]
            )

    # summary JSON
    with open(os.path.join(out_dir, "grpo_compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # human-readable txt
    txt = os.path.join(out_dir, "grpo_compare_summary.txt")
    lines = []
    lines.append(f"old: {summary['old_path']}")
    lines.append(f"new: {summary['new_path']}")
    lines.append(f"groups_total={summary['groups_total']} common={summary['groups_common']}")
    lines.append(
        f"GRPO wins: old={summary['grpo_wins_old']} new={summary['grpo_wins_new']} change={summary['grpo_win_change']}"
    )
    lines.append(f"GRPO delta count={summary['grpo_delta_count']}")
    if summary["grpo_delta_count"]:
        lines.append(
            "delta stats: "
            f"mean={summary['grpo_delta_mean']:.6f} std={summary['grpo_delta_std']:.6f} "
            f"min={summary['grpo_delta_min']:.6f} p10={summary['grpo_delta_p10']:.6f} "
            f"p50={summary['grpo_delta_p50']:.6f} p90={summary['grpo_delta_p90']:.6f} "
            f"max={summary['grpo_delta_max']:.6f}"
        )
        lines.append(
            f"delta sign: +{summary['grpo_delta_pos']} 0={summary['grpo_delta_zero']} -{summary['grpo_delta_neg']}"
        )
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare GRPO performance between two sweep.tsv files.")
    p.add_argument("--old", required=True, help="Old sweep.tsv path")
    p.add_argument("--new", required=True, help="New sweep.tsv path")
    p.add_argument("--out_dir", required=True, help="Output directory for comparison artifacts")
    p.add_argument("--wait", action="store_true", help="Wait until --new exists before comparing")
    p.add_argument("--wait_timeout", type=int, default=0, help="Max seconds to wait (0=forever)")
    p.add_argument("--wait_interval", type=float, default=30.0, help="Polling interval seconds")
    args = p.parse_args()

    if args.wait:
        t0 = time.time()
        while not os.path.exists(args.new):
            if args.wait_timeout and (time.time() - t0) > args.wait_timeout:
                raise SystemExit(f"Timed out waiting for new sweep.tsv: {args.new}")
            print(f"[WAIT] new sweep not found yet: {args.new} (sleep {args.wait_interval}s)")
            time.sleep(float(args.wait_interval))

    rows, summary = compare(old_path=args.old, new_path=args.new)
    _write_outputs(args.out_dir, rows, summary)
    print("[OK] wrote comparison to:", args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()





