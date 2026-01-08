#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


def _read_tsv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write_tsv(path: str, header: List[str], rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in header})


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill sweep.json/sweep.tsv metric columns from each run_dir/summary.tsv."
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Sweep output dir containing sweep.json and sweep.tsv (e.g. .../scaling_sweep_... ).",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite sweep.json/sweep.tsv (creates .bak backups). Otherwise writes sweep.fixed.*",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    sweep_json = os.path.join(out_dir, "sweep.json")
    sweep_tsv = os.path.join(out_dir, "sweep.tsv")

    if not os.path.exists(sweep_json):
        raise FileNotFoundError(sweep_json)
    if not os.path.exists(sweep_tsv):
        raise FileNotFoundError(sweep_tsv)

    with open(sweep_json, "r", encoding="utf-8") as f:
        sweep = json.load(f)

    metrics_weights: Dict[str, float] = sweep.get("metrics_weights") or {}
    metric_keys = list(metrics_weights.keys())
    metric_cols = [f"val_{k}" for k in metric_keys]

    rows: List[Dict] = sweep.get("rows") or []
    updated = 0
    missing_summary = 0
    missing_algo = 0

    for r in rows:
        run_dir = r.get("run_dir")
        algo = r.get("algo")
        if not run_dir or not algo:
            continue
        summary_path = os.path.join(run_dir, "summary.tsv")
        if not os.path.exists(summary_path):
            missing_summary += 1
            continue
        _, srows = _read_tsv(summary_path)
        hit = None
        for sr in srows:
            if (sr.get("algo") or "").strip() == str(algo):
                hit = sr
                break
        if hit is None:
            missing_algo += 1
            continue
        for k in metric_keys:
            col = f"val_{k}"
            if col in hit and hit[col] not in (None, ""):
                try:
                    r[col] = float(hit[col])
                except Exception:
                    r[col] = hit[col]
            else:
                # keep consistent presence even if missing in summary
                r[col] = r.get(col, 0.0)
        updated += 1

    # Build output TSV header: keep existing base columns, then metric_cols, then tail columns.
    old_header, _ = _read_tsv(sweep_tsv)
    base = [c for c in old_header if c.startswith("val_") is False]
    # Remove any old val_* columns from base, then append the desired metric_cols
    # and finally keep legacy val_* (like val_meteor) if they existed but are not in metric_cols.
    legacy_val = [c for c in old_header if c.startswith("val_") and c not in metric_cols and c != "val_reward"]
    # Ensure val_reward stays right after base numeric columns (we keep original order)
    header = []
    for c in old_header:
        if c == "val_reward" or c.startswith("val_"):
            continue
        header.append(c)
    # Put val_reward back (commonly expected)
    if "val_reward" in old_header:
        header.append("val_reward")
    header.extend(metric_cols)
    header.extend(legacy_val)
    # Ensure best_config/run_dir at end if present
    for tail in ["best_config", "run_dir"]:
        if tail in old_header and tail not in header:
            header.append(tail)

    if args.inplace:
        # backups
        os.replace(sweep_json, sweep_json + ".bak")
        os.replace(sweep_tsv, sweep_tsv + ".bak")
        out_json = sweep_json
        out_tsv = sweep_tsv
    else:
        out_json = os.path.join(out_dir, "sweep.fixed.json")
        out_tsv = os.path.join(out_dir, "sweep.fixed.tsv")

    sweep["rows"] = rows
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sweep, f, ensure_ascii=False, indent=2)
    _write_tsv(out_tsv, header, rows)

    print("[OK] backfilled metrics from summary.tsv")
    print(f"  out_dir={out_dir}")
    print(f"  metrics_keys={metric_keys}")
    print(f"  updated_rows={updated}/{len(rows)}")
    print(f"  missing_summary={missing_summary} missing_algo_row={missing_algo}")
    print(f"  wrote: {out_json}")
    print(f"  wrote: {out_tsv}")


if __name__ == "__main__":
    main()



