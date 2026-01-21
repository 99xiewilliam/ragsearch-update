from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a tiny HotPotQA sample dataset in this repo's parquet format."
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output dataset directory.",
    )
    p.add_argument(
        "--no_split",
        action="store_true",
        help="If set, write a single-file dataset (corpus.parquet + qa.parquet) under out_dir, without train/validation folders.",
    )
    p.add_argument(
        "--subset",
        type=str,
        default="distractor",
        help="HotPotQA subset/config name: distractor | fullwiki",
    )
    p.add_argument(
        "--split",
        type=str,
        default="validation",
        help="HF split name to sample from (usually train or validation).",
    )
    p.add_argument("--n_train", type=int, default=20, help="Number of train QA examples.")
    p.add_argument(
        "--n_val",
        type=int,
        default=0,
        help="Number of validation QA examples (0 => reuse train QA for validation).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_hotpotqa(subset: str, split: str):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: datasets. Install with `pip install datasets` in your env."
        ) from e
    return load_dataset("hotpotqa/hotpot_qa", subset, split=split)


def _iter_docs_from_examples(examples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    HotPotQA distractor/fullwiki provides:
      context: { title: [...], sentences: [[...], ...] }
    We map each (title, sentences) to one Doc.
    """
    by_title: Dict[str, Dict[str, Any]] = {}
    for ex in examples:
        ctx = ex.get("context") or {}
        titles = ctx.get("title") or []
        sents = ctx.get("sentences") or []
        # titles and sentences are aligned lists
        for i, t in enumerate(list(titles)):
            title = str(t or "").strip()
            if not title:
                continue
            para_sents = []
            try:
                para_sents = list(sents[i] or [])
            except Exception:
                para_sents = []
            text = "\n".join(str(x or "").strip() for x in para_sents if str(x or "").strip())
            text = text.strip()
            if not text:
                continue
            # de-dupe by title (good enough for a smoke sample)
            if title not in by_title:
                by_title[title] = {
                    "doc_id": f"wiki::{title}",
                    "contents": f"{title}\n\n{text}",
                    "metadata": {"title": title, "source": "hotpotqa"},
                }
    return list(by_title.values())


def _iter_qas_from_examples(examples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    qas: List[Dict[str, Any]] = []
    for ex in examples:
        qid = str(ex.get("id") or "").strip()
        q = str(ex.get("question") or "").strip()
        a = str(ex.get("answer") or "").strip()
        # HotPotQA provides supporting_facts {title: [...], sent_id: [...]}
        sf = ex.get("supporting_facts") or {}
        titles = sf.get("title") or []
        supporting_titles = [str(t).strip() for t in list(titles) if str(t).strip()]
        # Map titles -> our doc_id convention used in corpus: wiki::<title>
        supporting_doc_ids = [f"wiki::{t}" for t in supporting_titles]
        if not qid:
            qid = f"hotpotqa_{len(qas)}"
        if not q:
            continue
        qas.append(
            {
                "qid": qid,
                "query": q,
                "generation_gt": [a] if a else [""],
                # evidence labels (used by oracle_upper_bound.py)
                "supporting_titles": supporting_titles,
                "supporting_doc_ids": supporting_doc_ids,
            }
        )
    return qas


def _write_split(out_dir: str, split: str, *, docs: List[Dict[str, Any]], qas: List[Dict[str, Any]]) -> None:
    d = os.path.join(out_dir, split)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(docs).to_parquet(os.path.join(d, "corpus.parquet"), index=False)
    pd.DataFrame(qas).to_parquet(os.path.join(d, "qa.parquet"), index=False)


def _write_no_split(out_dir: str, *, docs: List[Dict[str, Any]], qas: List[Dict[str, Any]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(docs).to_parquet(os.path.join(out_dir, "corpus.parquet"), index=False)
    pd.DataFrame(qas).to_parquet(os.path.join(out_dir, "qa.parquet"), index=False)


def main() -> None:
    args = _parse_args()
    out_dir = str(args.out_dir)
    n_train = int(args.n_train)
    n_val = int(args.n_val)
    if n_train <= 0:
        raise ValueError("n_train must be > 0")
    if n_val < 0:
        raise ValueError("n_val must be >= 0")

    ds = _load_hotpotqa(str(args.subset), str(args.split))
    ds = ds.shuffle(seed=int(args.seed))

    need = n_train + (n_val if n_val > 0 else 0)
    picked = ds.select(range(min(need, len(ds))))
    picked_list = [picked[i] for i in range(len(picked))]
    train_ex = picked_list[:n_train]
    val_ex = picked_list[n_train : n_train + n_val] if n_val > 0 else train_ex

    # Build a shared corpus from union(train,val) to mimic the typical "shared corpus" setting.
    union_ex = list(train_ex) + [x for x in val_ex if x not in train_ex]
    docs = _iter_docs_from_examples(union_ex)
    qas_train = _iter_qas_from_examples(train_ex)
    qas_val = _iter_qas_from_examples(val_ex)

    if bool(args.no_split):
        # One QA file only. If n_val==0, it's the same as train_qas; otherwise use union(train,val).
        if n_val > 0:
            qas_one = _iter_qas_from_examples(list(train_ex) + list(val_ex))
            # de-dup by qid while preserving order
            seen = set()
            qas_one_uniq = []
            for item in qas_one:
                qid = str(item.get("qid", "")).strip()
                if qid and qid in seen:
                    continue
                if qid:
                    seen.add(qid)
                qas_one_uniq.append(item)
            qas_one = qas_one_uniq
        else:
            qas_one = qas_train

        _write_no_split(out_dir, docs=docs, qas=qas_one)
        print(
            f"[OK] wrote dataset (no_split): {out_dir}\n"
            f"  corpus_docs={len(docs)}\n"
            f"  qas={len(qas_one)}"
        )
        return

    _write_split(out_dir, "train", docs=docs, qas=qas_train)
    _write_split(out_dir, "validation", docs=docs, qas=qas_val)

    print(
        f"[OK] wrote dataset: {out_dir}\n"
        f"  corpus_docs={len(docs)}\n"
        f"  train_qas={len(qas_train)}\n"
        f"  val_qas={len(qas_val)}"
    )


if __name__ == "__main__":
    main()

