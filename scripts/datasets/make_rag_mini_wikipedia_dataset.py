from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a rag-mini-wikipedia dataset in this repo's parquet format."
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
        "--n_train",
        type=int,
        default=30,
        help="Number of train QA examples to sample from question-answer/test.",
    )
    p.add_argument(
        "--n_val",
        type=int,
        default=0,
        help="Number of validation QA examples (0 => reuse train QA for validation).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for sampling QA and (optional) corpus subsampling.",
    )
    p.add_argument(
        "--corpus_limit",
        type=int,
        default=0,
        help="If >0, subsample this many passages from the corpus (0 => use full corpus).",
    )
    p.add_argument(
        "--qa_split",
        type=str,
        default="test",
        help="HF split name for question-answer config (dataset only provides test).",
    )
    return p.parse_args()


def _load_hf() -> Tuple[Any, Any]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: datasets. Install with `pip install datasets` in your env."
        ) from e

    corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    return corpus, qa


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
    seed = int(args.seed)
    corpus_limit = int(args.corpus_limit)
    qa_split = str(args.qa_split)

    if n_train <= 0:
        raise ValueError("n_train must be > 0")
    if n_val < 0:
        raise ValueError("n_val must be >= 0")
    if corpus_limit < 0:
        raise ValueError("corpus_limit must be >= 0")

    corpus_ds, qa_ds = _load_hf()
    if "passages" not in corpus_ds:
        raise ValueError(f"Unexpected corpus splits: {list(corpus_ds.keys())}")
    if qa_split not in qa_ds:
        raise ValueError(f"Unexpected QA split {qa_split!r}. Available: {list(qa_ds.keys())}")

    passages = corpus_ds["passages"]
    qasrc = qa_ds[qa_split]

    # Optional corpus subsampling (corpus is small: 3200, so default is full).
    passages = passages.shuffle(seed=seed)
    if corpus_limit > 0:
        passages = passages.select(range(min(corpus_limit, len(passages))))

    docs: List[Dict[str, Any]] = []
    for ex in passages:
        pid = ex.get("id")
        text = str(ex.get("passage") or "").strip()
        if text == "":
            continue
        doc_id = f"miniwiki::{pid}"
        docs.append(
            {
                "doc_id": doc_id,
                "contents": text,
                "metadata": {"source": "rag-mini-wikipedia", "passage_id": int(pid) if pid is not None else None},
            }
        )

    # Sample QA (dataset only provides test split; we sample train/val from it).
    qasrc = qasrc.shuffle(seed=seed)
    need = n_train + (n_val if n_val > 0 else 0)
    picked = qasrc.select(range(min(need, len(qasrc))))
    picked_list = [picked[i] for i in range(len(picked))]
    train_ex = picked_list[:n_train]
    val_ex = picked_list[n_train : n_train + n_val] if n_val > 0 else train_ex

    def _to_qa(ex: Dict[str, Any]) -> Dict[str, Any]:
        qid = ex.get("id")
        q = str(ex.get("question") or "").strip()
        a = str(ex.get("answer") or "").strip()
        if not q:
            return {}
        return {"qid": f"miniwiki_qa_{qid}", "query": q, "generation_gt": [a] if a else [""]}

    qas_train = [x for x in (_to_qa(ex) for ex in train_ex) if x]
    qas_val = [x for x in (_to_qa(ex) for ex in val_ex) if x]

    if bool(args.no_split):
        qas_one = qas_train
        if n_val > 0:
            # de-dup by qid if user wanted both
            seen = set()
            qas_one = []
            for item in qas_train + qas_val:
                qid = item.get("qid")
                if qid in seen:
                    continue
                seen.add(qid)
                qas_one.append(item)

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
        f"  corpus_docs={len(docs)} (corpus_limit={corpus_limit or 'full'})\n"
        f"  train_qas={len(qas_train)}\n"
        f"  val_qas={len(qas_val)}"
    )


if __name__ == "__main__":
    main()

