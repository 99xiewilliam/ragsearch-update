#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional


def _norm_list(x: List[str]) -> List[str]:
    out: List[str] = []
    for s in x:
        for part in (s or "").split(","):
            part = part.strip()
            if part:
                out.append(part)
    # de-dup while keeping order
    seen = set()
    uniq: List[str] = []
    for m in out:
        if m in seen:
            continue
        seen.add(m)
        uniq.append(m)
    return uniq


def download_one(
    *,
    repo_id: str,
    cache_dir: str,
    local_dir: str,
    token: Optional[str],
    max_retries: int,
) -> str:
    from huggingface_hub import snapshot_download

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    last_err: Optional[BaseException] = None
    force = False
    for attempt in range(1, int(max_retries) + 1):
        try:
            print(f"[HF] repo={repo_id} attempt={attempt}/{max_retries} force_download={force}", flush=True)
            p = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_dir=local_dir,
                token=token,
                force_download=force,
            )
            return str(p)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # Typical corruption case:
            # "Consistency check failed: file should be of size ... but has size ... (vocab.json)"
            if "consistency check failed" in msg or "file should be of size" in msg:
                force = True
            retryable = any(k in msg for k in ("timed out", "timeout", "connection", "network", "502", "503", "504"))
            if attempt < int(max_retries) and retryable:
                wait = min(30 * attempt, 180)
                print(f"[HF] retryable_error={type(e).__name__}: {e} wait={wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
    assert last_err is not None
    raise last_err


def main() -> None:
    p = argparse.ArgumentParser(description="Batch download HuggingFace models (snapshot_download)")
    p.add_argument(
        "--models",
        action="append",
        default=[],
        help="Model repo id. Can be passed multiple times or as comma-separated list.",
    )
    p.add_argument("--local_root", type=str, default="/home/xwh/models/hf", help="Where to store models (one dir per repo).")
    p.add_argument("--cache_dir", type=str, default="/home/xwh/.cache/huggingface", help="HuggingFace cache dir.")
    p.add_argument("--token", type=str, default="", help="Optional HF token (or set env HF_TOKEN).")
    p.add_argument("--max_retries", type=int, default=8)
    args = p.parse_args()

    models = _norm_list(list(args.models or []))
    if not models:
        raise SystemExit("No models provided. Use --models <repo> (repeatable).")

    token = (args.token or "").strip() or os.getenv("HF_TOKEN") or None
    local_root = Path(args.local_root).expanduser()
    cache_dir = str(Path(args.cache_dir).expanduser())
    local_root.mkdir(parents=True, exist_ok=True)

    for repo in models:
        safe = repo.replace("/", "--")
        local_dir = str(local_root / safe)
        print(f"\n[DOWNLOAD] {repo} -> {local_dir}", flush=True)
        try:
            p = download_one(repo_id=repo, cache_dir=cache_dir, local_dir=local_dir, token=token, max_retries=int(args.max_retries))
            print(f"[DONE] {repo} path={p}", flush=True)
        except Exception as e:
            print(f"[FAIL] {repo} err={type(e).__name__}: {e}", flush=True)
            raise


if __name__ == "__main__":
    main()

