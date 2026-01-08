from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

import pandas as pd

from .types import Doc, QAExample


def _to_list_str(x) -> List[str]:
    if x is None:
        return [""]
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()] or [""]
    s = str(x)
    return [s] if s.strip() else [""]


def load_split(dataset_dir: str, split: str) -> Tuple[List[Doc], List[QAExample]]:
    """
    Load corpus + qa from:
      {dataset_dir}/{split}/corpus.parquet
      {dataset_dir}/{split}/qa.parquet
    """
    corpus_path = f"{dataset_dir}/{split}/corpus.parquet"
    qa_path = f"{dataset_dir}/{split}/qa.parquet"

    corpus_df = pd.read_parquet(corpus_path)
    qa_df = pd.read_parquet(qa_path)

    docs: List[Doc] = []
    for _, r in corpus_df.iterrows():
        doc_id = str(r.get("doc_id", "")).strip()
        contents = str(r.get("contents", "") or "")
        meta = r.get("metadata", None)
        docs.append(Doc(doc_id=doc_id, contents=contents, metadata=meta if isinstance(meta, dict) else None))

    qas: List[QAExample] = []
    for _, r in qa_df.iterrows():
        qid = str(r.get("qid", "")).strip()
        query = str(r.get("query", "") or "")
        generation_gt = _to_list_str(r.get("generation_gt", ""))
        qas.append(QAExample(qid=qid, query=query, generation_gt=generation_gt))

    return docs, qas


def dataset_brief(docs: List[Doc], qas: List[QAExample]) -> dict:
    return {
        "docs": len(docs),
        "qas": len(qas),
        "sample_qa": asdict(qas[0]) if qas else None,
        "sample_doc": {"doc_id": docs[0].doc_id, "contents_preview": (docs[0].contents[:200] if docs else "")},
    }


