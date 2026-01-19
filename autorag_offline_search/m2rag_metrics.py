from __future__ import annotations

"""
M2RAG-style metric bundle (8 metrics + overall).

Important:
- The official M2RAG repo computes these metrics from a different logging schema
  (webpages/aux_images/output_images + evaluator). Our RAG pipeline does not produce
  that schema.
- This module provides a *compatible, lightweight proxy* using the information we
  do have (query/pred/refs + pipeline trace: retrieved/final_chunks/used_images).

Metric names follow the official naming:
  text: fluency, response_relevancy, context_precision, faithfulness
  multi_modal: image_coherence, image_helpfulness, image_reference, image_recall
  overall: mean of the 8 metrics
"""

import re
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

from .metrics import compute_rouge


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _tokenize(s: str) -> List[str]:
    s = str(s or "").lower()
    # keep words/numbers, drop punctuation
    return re.findall(r"[a-z0-9]+", s)


def _fluency_proxy(pred: str) -> float:
    """
    Cheap fluency proxy:
    - non-empty
    - penalize excessive repetition
    - penalize extremely short outputs
    """
    p = str(pred or "").strip()
    if not p:
        return 0.0
    toks = _tokenize(p)
    if len(toks) <= 1:
        return 0.2
    uniq = len(set(toks))
    rep_ratio = uniq / max(1, len(toks))  # 0..1 (higher => less repetition)
    # length factor: 3..80 tokens is usually OK for short-form QA
    if len(toks) < 3:
        len_factor = 0.4
    elif len(toks) > 120:
        len_factor = 0.6
    else:
        len_factor = 1.0
    return _clamp01(rep_ratio * len_factor)


@lru_cache(maxsize=1024)
def _st_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    # CPU is more predictable for scoring; GPU is optional but may be unavailable.
    return SentenceTransformer(model_name, device="cpu")


@lru_cache(maxsize=20000)
def _embed_cached(model_name: str, text: str) -> Tuple[float, ...]:
    m = _st_model(model_name)
    v = m.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return tuple(float(x) for x in v.tolist())


def _cos_sim(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    s = 0.0
    for i in range(len(a)):
        s += float(a[i]) * float(b[i])
    # dot of normalized embeddings in [-1,1]
    return _clamp01((s + 1.0) / 2.0)


def _response_relevancy_proxy(query: str, pred: str, *, sim_model: str) -> float:
    q = str(query or "").strip()
    p = str(pred or "").strip()
    if not q or not p:
        return 0.0
    try:
        aq = _embed_cached(sim_model, q)
        ap = _embed_cached(sim_model, p)
        return _cos_sim(aq, ap)
    except Exception:
        # fallback: overlap ratio
        qt = set(_tokenize(q))
        pt = set(_tokenize(p))
        if not qt:
            return 0.0
        return _clamp01(len(qt & pt) / max(1, len(qt)))


def _best_ref_by_rougel(pred: str, refs: Sequence[str]) -> str:
    if not refs:
        return ""
    best = ""
    best_rl = -1.0
    for r in refs:
        rl = float(compute_rouge(pred, [str(r)]).get("rougeL", 0.0))
        if rl > best_rl:
            best_rl = rl
            best = str(r)
    return best


def _extract_final_chunks(trace: Any) -> List[str]:
    if not isinstance(trace, dict):
        return []
    fc = trace.get("final_chunks")
    if isinstance(fc, list):
        return [str(x or "") for x in fc if str(x or "").strip()]
    return []


def _extract_candidate_images(trace: Any, *, max_docs: int = 5) -> List[str]:
    """
    Candidate images = images attached to top retrieved docs (multimodal trace).
    """
    if not isinstance(trace, dict):
        return []
    rr = trace.get("retrieved")
    if not isinstance(rr, list) or not rr:
        return []
    out: List[str] = []
    for item in rr[: max_docs]:
        if not isinstance(item, dict):
            continue
        imgs = item.get("images")
        if isinstance(imgs, list):
            for p in imgs:
                s = str(p or "").strip()
                if s:
                    out.append(s)
    # de-dup preserve order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _extract_used_images(trace: Any) -> List[str]:
    if not isinstance(trace, dict):
        return []
    imgs = trace.get("used_images")
    if isinstance(imgs, list):
        return [str(x or "").strip() for x in imgs if str(x or "").strip()]
    return []


def _context_precision_proxy(pred: str, refs: Sequence[str], trace: Any) -> float:
    chunks = _extract_final_chunks(trace)
    if not chunks:
        return 0.0
    top = chunks[: min(5, len(chunks))]
    best_ref = _best_ref_by_rougel(pred, refs) if refs else ""
    hits = 0
    for ch in top:
        ch_l = ch.lower()
        # strong signal: contains any reference answer substring
        ok = False
        for r in refs or []:
            rs = str(r or "").strip()
            if rs and rs.lower() in ch_l:
                ok = True
                break
        if not ok and best_ref:
            # weak signal: chunk overlaps the best ref
            rl = float(compute_rouge(best_ref, [ch]).get("rougeL", 0.0))
            ok = rl >= 0.15
        if ok:
            hits += 1
    return _clamp01(hits / max(1, len(top)))


def _faithfulness_proxy(pred: str, trace: Any) -> float:
    """
    Faithfulness proxy: overlap between answer and final context.
    """
    chunks = _extract_final_chunks(trace)
    if not chunks:
        return 0.0
    ctx = "\n\n".join(chunks[: min(8, len(chunks))])
    if not ctx.strip() or not str(pred or "").strip():
        return 0.0
    rl = float(compute_rouge(str(pred), [ctx]).get("rougeL", 0.0))
    return _clamp01(rl)


def compute_m2rag_proxy(
    *,
    query: str,
    pred: str,
    refs: Sequence[str],
    trace: Any,
    similarity_model: str,
) -> Dict[str, float]:
    """
    Return:
      m2rag_fluency, m2rag_response_relevancy, m2rag_context_precision, m2rag_faithfulness,
      m2rag_image_coherence, m2rag_image_helpfulness, m2rag_image_reference, m2rag_image_recall,
      m2rag_overall
    """
    fluency = _fluency_proxy(pred)
    relevancy = _response_relevancy_proxy(query, pred, sim_model=str(similarity_model))
    ctx_prec = _context_precision_proxy(pred, refs, trace)
    faithful = _faithfulness_proxy(pred, trace)

    cand_imgs = _extract_candidate_images(trace)
    used_imgs = _extract_used_images(trace)

    # image_recall (similar to M2RAG evaluate.py spirit, but without "final_score"):
    # treat top-3 candidate images as "positive".
    positives = cand_imgs[:3]
    if not positives:
        image_recall = 1.0
    else:
        image_recall = len(set(positives) & set(used_imgs)) / max(1, len(set(positives)))

    # The remaining image metrics require an image-aware evaluator in the official repo.
    # We provide lightweight proxies so the overall has signal but remains robust.
    if not cand_imgs:
        # no candidate images => not applicable; don't punish
        image_coh = 1.0
        image_help = 1.0
        image_ref = 1.0
    else:
        if used_imgs:
            image_coh = 1.0 if set(used_imgs).issubset(set(cand_imgs)) else 0.5
            # proxy: if answer is relevant and we used images, slightly reward
            image_help = 0.5 + 0.5 * _clamp01(relevancy)
            # proxy: does answer mention image/figure?
            p_low = str(pred or "").lower()
            image_ref = 1.0 if ("image" in p_low or "figure" in p_low or "pictured" in p_low) else 0.5
        else:
            image_coh = 0.0
            image_help = 0.0
            image_ref = 0.0

    # overall (mean of 8 metrics)
    overall = (
        fluency
        + relevancy
        + ctx_prec
        + faithful
        + _safe_float(image_coh)
        + _safe_float(image_help)
        + _safe_float(image_ref)
        + _safe_float(image_recall)
    ) / 8.0

    return {
        "m2rag_fluency": _clamp01(fluency),
        "m2rag_response_relevancy": _clamp01(relevancy),
        "m2rag_context_precision": _clamp01(ctx_prec),
        "m2rag_faithfulness": _clamp01(faithful),
        "m2rag_image_coherence": _clamp01(image_coh),
        "m2rag_image_helpfulness": _clamp01(image_help),
        "m2rag_image_reference": _clamp01(image_ref),
        "m2rag_image_recall": _clamp01(image_recall),
        "m2rag_overall": _clamp01(overall),
    }

