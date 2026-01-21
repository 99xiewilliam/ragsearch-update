from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

import pandas as pd
import json
import numpy as np
import os

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
    Load corpus + qa.

    Supported dataset layouts:

    1) Split layout (recommended):
       {dataset_dir}/{split}/corpus.parquet
       {dataset_dir}/{split}/qa.parquet

    2) Single-file layout (no train/validation folders):
       {dataset_dir}/corpus.parquet
       {dataset_dir}/qa.parquet

    Notes:
    - The CLI/experiment code will still call load_split(dataset_dir,"train") and
      load_split(dataset_dir,"validation"). Under single-file layout, both calls
      will load the same files.
    """
    # Prefer split layout if present; otherwise fall back to single-file layout.
    cand_split_corpus = os.path.join(dataset_dir, split, "corpus.parquet")
    cand_split_qa = os.path.join(dataset_dir, split, "qa.parquet")

    cand_root_corpus = os.path.join(dataset_dir, "corpus.parquet")
    cand_root_qa = os.path.join(dataset_dir, "qa.parquet")

    corpus_path: str
    qa_path: str

    if os.path.exists(cand_split_corpus) and os.path.exists(cand_split_qa):
        corpus_path, qa_path = cand_split_corpus, cand_split_qa
    elif os.path.exists(cand_root_corpus) and os.path.exists(cand_root_qa):
        corpus_path, qa_path = cand_root_corpus, cand_root_qa
    else:
        # Backward/partial compatibility: if only one split exists, reuse it.
        reused = None
        for alt in ("train", "validation"):
            c = os.path.join(dataset_dir, alt, "corpus.parquet")
            q = os.path.join(dataset_dir, alt, "qa.parquet")
            if os.path.exists(c) and os.path.exists(q):
                reused = (c, q, alt)
                break
        if reused is not None:
            corpus_path, qa_path, alt = reused
        else:
            raise FileNotFoundError(
                "Could not find dataset parquet files. Tried:\n"
                f"- {cand_split_corpus}\n"
                f"- {cand_split_qa}\n"
                f"- {cand_root_corpus}\n"
                f"- {cand_root_qa}\n"
                f"- {os.path.join(dataset_dir, 'train', 'corpus.parquet')} (+ qa.parquet)\n"
                f"- {os.path.join(dataset_dir, 'validation', 'corpus.parquet')} (+ qa.parquet)\n"
            )

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
        # Preserve any extra QA fields (e.g., evidence labels) in metadata.
        md = {}
        for k in list(getattr(qa_df, "columns", [])):
            if k in {"qid", "query", "generation_gt"}:
                continue
            try:
                v = r.get(k, None)
            except Exception:
                v = None
            # Convert numpy scalars/arrays to python types if possible
            if isinstance(v, np.ndarray):
                try:
                    v = v.tolist()
                except Exception:
                    v = [str(x) for x in list(v)]
            # Try to decode JSON strings for structured columns
            if isinstance(v, str) and v.strip():
                s = v.strip()
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    try:
                        vj = json.loads(s)
                        v = vj
                    except Exception:
                        pass
            md[k] = v
        qas.append(QAExample(qid=qid, query=query, generation_gt=generation_gt, metadata=(md or None)))

    return docs, qas


def dataset_brief(docs: List[Doc], qas: List[QAExample]) -> dict:
    return {
        "docs": len(docs),
        "qas": len(qas),
        "sample_qa": asdict(qas[0]) if qas else None,
        "sample_doc": {"doc_id": docs[0].doc_id, "contents_preview": (docs[0].contents[:200] if docs else "")},
    }


