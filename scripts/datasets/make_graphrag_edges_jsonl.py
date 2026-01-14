#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Create graph/edges.jsonl from a simple edge list.")
    p.add_argument("--out_dataset_dir", type=str, required=True, help="Dataset dir where graph/edges.jsonl will be written.")
    p.add_argument("--edges", type=str, required=True, help="JSON file path containing a list of edges.")
    p.add_argument("--src_key", type=str, default="src")
    p.add_argument("--dst_key", type=str, default="dst")
    args = p.parse_args()

    out_dir = Path(args.out_dataset_dir) / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "edges.jsonl"

    edges = json.loads(Path(args.edges).read_text(encoding="utf-8"))
    if not isinstance(edges, list):
        raise SystemExit("edges must be a JSON list")

    with open(out_path, "w", encoding="utf-8") as f:
        for e in edges:
            if not isinstance(e, dict):
                continue
            s = str(e.get(args.src_key, "") or "").strip()
            d = str(e.get(args.dst_key, "") or "").strip()
            if not s or not d:
                continue
            f.write(json.dumps({"src": s, "dst": d}, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

