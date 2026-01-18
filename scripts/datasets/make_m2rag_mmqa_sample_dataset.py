#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _resolve_url(path_in_repo: str) -> str:
    # Use "resolve/main" so we don't need a commit hash.
    return f"https://huggingface.co/datasets/whalezzz/M2RAG/resolve/main/{path_in_repo.lstrip('/')}"


def _http_stream_jsonl(url: str, *, timeout_s: int = 120) -> Iterable[Dict[str, Any]]:
    import requests

    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            s = str(line).strip()
            if not s:
                continue
            yield json.loads(s)


def _http_download(url: str, dst: Path, *, timeout_s: int = 120, retries: int = 5) -> None:
    import requests

    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[BaseException] = None
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout_s) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except BaseException as e:
            last_err = e
            time.sleep(min(2**i, 8))
    raise RuntimeError(f"download failed after {retries} retries: {url} ({type(last_err).__name__}: {last_err})")


def _pick_examples(stream: Iterable[Dict[str, Any]], *, n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex in stream:
        out.append(ex)
        if len(out) >= n:
            break
    if len(out) < n:
        raise SystemExit(f"Only got {len(out)} examples from stream, expected {n}.")
    return out


def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return [""]
    if isinstance(x, list):
        out = []
        for v in x:
            s = str(v or "").strip()
            if s:
                out.append(s)
        return out or [""]
    s = str(x).strip()
    return [s] if s else [""]


def _make_rows(exs: List[Dict[str, Any]], *, asset_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    corpus_rows: List[Dict[str, Any]] = []
    qa_rows: List[Dict[str, Any]] = []

    for ex in exs:
        qid = str(ex.get("id", "") or "").strip() or f"ex_{len(qa_rows)+1}"
        query = str(ex.get("query", "") or "").strip()
        answers = _as_list_str(ex.get("answer", ""))

        pos_text = str(ex.get("pos_text", "") or "").strip()
        pos_image_path = str(ex.get("pos_image_path", "") or "").strip()
        pos_image_caption = str(ex.get("pos_image_caption", "") or "").strip()

        local_image_path = ""
        if pos_image_path:
            # Keep extension if any (png/jpg).
            ext = Path(pos_image_path).suffix or ".png"
            dst = asset_dir / "images" / f"{qid}{ext}"
            _http_download(_resolve_url(pos_image_path), dst)
            local_image_path = str(dst.resolve())

        # Build a single "positive" doc per example (smoke dataset).
        # Retrieval uses text (pos_text/caption), generation may use image_path if present.
        doc_id = f"doc_{qid}"
        contents_parts: List[str] = []
        if pos_text:
            contents_parts.append(pos_text)
        if pos_image_caption:
            contents_parts.append(f"[caption] {pos_image_caption}")
        if not contents_parts:
            contents_parts.append("This is a multimodal document. Answer the question using the attached image if available.")
        contents = "\n\n".join(contents_parts).strip()

        md: Dict[str, Any] = {"source": "whalezzz/M2RAG", "task": "mmqa", "m2rag_id": qid}
        if local_image_path:
            md["image_path"] = local_image_path
        if pos_image_caption:
            md["caption"] = pos_image_caption
        if pos_image_path:
            md["m2rag_image_relpath"] = pos_image_path

        corpus_rows.append({"doc_id": doc_id, "contents": contents, "metadata": json.dumps(md, ensure_ascii=False)})
        qa_rows.append({"qid": qid, "query": query, "generation_gt": json.dumps(answers, ensure_ascii=False)})

    return corpus_rows, qa_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Create a tiny M2RAG-MMQA sample dataset for autorag_offline_search (parquet).")
    p.add_argument("--out_dir", type=str, default="/home/xwh/ragsearch-update/datasets/m2rag_mmqa_test10")
    p.add_argument("--n", type=int, default=10, help="How many examples to sample from mmqa/test_data.jsonl")
    p.add_argument(
        "--only_text",
        action="store_true",
        help="Only keep examples with pos_text and without pos_image_path (useful when you don't have a VL model served).",
    )
    args = p.parse_args()

    out = Path(args.out_dir)
    asset_dir = out / "assets"
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "validation").mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    url = _resolve_url("mmqa/test_data.jsonl")
    stream = _http_stream_jsonl(url)
    if args.only_text:
        def _flt(it):
            for ex in it:
                if str(ex.get("pos_text", "") or "").strip() and (not str(ex.get("pos_image_path", "") or "").strip()):
                    yield ex
        stream = _flt(stream)
    exs = _pick_examples(stream, n=int(args.n))
    corpus_rows, qa_rows = _make_rows(exs, asset_dir=asset_dir)

    import pandas as pd

    corpus = pd.DataFrame(corpus_rows)
    qa = pd.DataFrame(qa_rows)

    # For smoke tests, duplicate into both splits to keep the CLI happy.
    for split in ("train", "validation"):
        corpus.to_parquet(out / split / "corpus.parquet", index=False)
        qa.to_parquet(out / split / "qa.parquet", index=False)

    print(f"[OK] wrote dataset: {out}")
    print(f"  docs={len(corpus_rows)} qas={len(qa_rows)}")
    print(f"  assets={asset_dir}")


if __name__ == "__main__":
    main()

