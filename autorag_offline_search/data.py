from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

import pandas as pd
import json
import numpy as np

from .types import Doc, QAExample


def _to_list_str(x) -> List[str]:
    if x is None:
        return [""]
    # Parquet may load list-like columns as numpy arrays
    if isinstance(x, np.ndarray):
        try:
            x = x.tolist()
        except Exception:
            x = [str(v) for v in list(x)]
    if isinstance(x, list):
        out: List[str] = []
        for v in x:
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            # Some datasets store a list-of-strings where each element is itself a stringified list,
            # e.g. ["['Prussian']"]. Try to unwrap once.
            if s.startswith("[") and s.endswith("]"):
                # Try JSON first
                try:
                    obj = json.loads(s)
                    if isinstance(obj, list):
                        out.extend([str(t).strip() for t in obj if str(t).strip()])
                        continue
                except Exception:
                    pass
                # Fallback to Python literal list
                try:
                    import ast

                    obj = ast.literal_eval(s)
                    if isinstance(obj, list):
                        out.extend([str(t).strip() for t in obj if str(t).strip()])
                        continue
                except Exception:
                    pass
            out.append(s)
        return out or [""]
    # Some parquet writers store lists as strings, e.g. "['82']" or '["82"]'
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            # Try JSON first
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(v) for v in obj if str(v).strip()] or [""]
            except Exception:
                pass
            # Fallback to Python literal list
            try:
                import ast

                obj = ast.literal_eval(s)
                if isinstance(obj, list):
                    return [str(v) for v in obj if str(v).strip()] or [""]
            except Exception:
                pass
        return [s] if s else [""]
    s = str(x).strip()
    return [s] if s else [""]


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
        if isinstance(meta, dict):
            md = meta
        elif isinstance(meta, str) and meta.strip():
            try:
                md = json.loads(meta)
                if not isinstance(md, dict):
                    md = None
            except Exception:
                md = None
        else:
            md = None
        docs.append(Doc(doc_id=doc_id, contents=contents, metadata=md))

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


