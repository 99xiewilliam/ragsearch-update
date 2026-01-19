#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from autorag_offline_search.data import load_split
from autorag_offline_search.metrics import MetricsConfig, aggregate_reward, compute_rouge, parse_weights
from autorag_offline_search.m2rag_metrics import compute_m2rag_proxy
from autorag_offline_search.modules.generator_fixed import GeneratorConfig, generate_answer, generate_answer_with_images
from autorag_offline_search.modules.multimodal_utils import extract_image_paths
from autorag_offline_search.pipelines.config import resolve_model
from autorag_offline_search.types import Doc, QAExample


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("YAML must be a dict/mapping")
    return dict(obj)


def _best_ref_for_rouge(pred: str, refs: Sequence[str]) -> str:
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


def _oracle_doc_for_qa(docs: Sequence[Doc], qa: QAExample) -> Optional[Doc]:
    """
    For datasets created by scripts/datasets/make_m2rag_mmqa_sample_dataset.py:
      - doc_id is 'doc_<qid>'
      - metadata contains m2rag_id == qid
    """
    qid = str(qa.qid or "").strip()
    if qid:
        wanted = f"doc_{qid}"
        for d in docs:
            if str(d.doc_id or "").strip() == wanted:
                return d
        for d in docs:
            md = d.metadata or {}
            if isinstance(md, dict) and str(md.get("m2rag_id", "")).strip() == qid:
                return d

    # Fallback: try to find a doc containing any reference answer string.
    # This is only an approximation of "upper bound" when gold evidence is not available.
    refs = list(qa.generation_gt or [])
    refs = [str(r).strip() for r in refs if str(r).strip()]
    if not refs:
        return None
    for d in docs:
        text = str(d.contents or "").lower()
        for r in refs:
            if r.lower() in text:
                return d
    return None


def _oracle_context_and_images(d: Doc, *, allow_images: bool) -> Tuple[str, List[str]]:
    ctx = str(d.contents or "").strip()
    imgs: List[str] = []
    if allow_images:
        imgs = [p for p in extract_image_paths(d.metadata) if str(p).strip()]
    # keep it cheap/safe: at most 2 images (generator supports up to 4, but 2 is enough for oracle probing)
    return ctx, imgs[:2]


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate an oracle(ish) upper bound by feeding oracle context directly to the generator.")
    p.add_argument("--dataset_dir", type=str, required=True, help="Dataset dir (split or no_split layout).")
    p.add_argument("--split", type=str, default="train", help="Which split to load: train|validation (no_split will ignore).")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on number of QA examples (0=no cap).")
    p.add_argument("--print_n", type=int, default=0, help="Print first N example predictions to stdout (0=disable).")

    p.add_argument("--llm_base_url", type=str, required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:9000/v1")
    p.add_argument("--generator_model", type=str, default="qwen3_vl_4b", help="Served model id/path (or preset: qwen3/qwen3_vl_4b).")
    p.add_argument("--generator_max_tokens", type=int, default=256, help="Output token budget (not context length).")
    p.add_argument("--allow_images", action="store_true", help="If set, attach oracle images when available (multimodal upper bound).")

    p.add_argument("--metrics_weights", type=str, default="qa_f1:1,em:1", help="Weight spec, e.g. qa_f1:1,em:1")
    p.add_argument("--bertscore_model", type=str, default="microsoft/deberta-xlarge-mnli")

    p.add_argument("--out_dir", type=str, default="", help="If set, write oracle_results.json and oracle_predictions.jsonl here.")
    args = p.parse_args()

    docs, qas = load_split(args.dataset_dir, args.split)
    if args.limit and int(args.limit) > 0:
        qas = list(qas[: int(args.limit)])

    mcfg = MetricsConfig(weights=parse_weights(args.metrics_weights), bertscore_model=str(args.bertscore_model))

    preds: List[str] = []
    refs_for_bert: List[str] = []
    preds_for_sim: List[str] = []
    refs_for_sim: List[str] = []

    rouge1s: List[float] = []
    rouge2s: List[float] = []
    rougeLs: List[float] = []
    ems: List[float] = []
    f1s: List[float] = []
    accs: List[float] = []
    bleus: List[float] = []
    meteors: List[float] = []
    chrfs: List[float] = []

    need = set(mcfg.weights.keys())
    need_bertscore = "bertscore_f1" in need
    need_similarity = "similarity" in need
    need_bleu = "bleu" in need
    need_meteor = "meteor" in need
    need_chrf = "chrf" in need
    need_em = "em" in need
    need_f1 = "qa_f1" in need
    need_accuracy = "accuracy" in need
    need_m2rag = any(str(k).startswith("m2rag_") for k in need)

    # lazy imports (keep script runnable in minimal envs; project env already has deps)
    from autorag_offline_search.metrics import (
        compute_bertscore_f1_batch,
        compute_bleu,
        compute_chrf,
        compute_exact_match,
        compute_meteor,
        compute_qa_f1,
        compute_similarity_batch,
    )

    # M2RAG-style proxy metrics accumulators
    m2_flu: List[float] = []
    m2_rel: List[float] = []
    m2_ctxp: List[float] = []
    m2_faith: List[float] = []
    m2_icoh: List[float] = []
    m2_ihelp: List[float] = []
    m2_iref: List[float] = []
    m2_irec: List[float] = []

    out_dir = str(args.out_dir or "").strip()
    dump_f = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        dump_f = open(os.path.join(out_dir, "oracle_predictions.jsonl"), "w", encoding="utf-8")
        dump_f.write(json.dumps({"_type": "meta", "dataset_dir": args.dataset_dir, "split": args.split}, ensure_ascii=False) + "\n")

    t0 = time.time()
    printed = 0
    for ex in qas:
        odoc = _oracle_doc_for_qa(docs, ex)
        if odoc is None:
            ctx = ""
            imgs = []
        else:
            ctx, imgs = _oracle_context_and_images(odoc, allow_images=bool(args.allow_images))

        model = resolve_model(str(args.generator_model))
        if imgs:
            pred = generate_answer_with_images(
                query=str(ex.query),
                context=ctx,
                image_paths=imgs,
                cfg=GeneratorConfig(model=model, max_tokens=int(args.generator_max_tokens)),
                llm_base_url=str(args.llm_base_url),
            )
        else:
            pred = generate_answer(
                query=str(ex.query),
                context=ctx,
                cfg=GeneratorConfig(model=model, max_tokens=int(args.generator_max_tokens)),
                llm_base_url=str(args.llm_base_url),
            )

        refs = list(ex.generation_gt or [])
        r = compute_rouge(pred, refs)
        rouge1s.append(float(r["rouge1"]))
        rouge2s.append(float(r["rouge2"]))
        rougeLs.append(float(r["rougeL"]))

        if need_meteor:
            meteors.append(float(compute_meteor(pred, refs)))
        if need_bleu:
            bleus.append(float(compute_bleu(pred, refs)))
        if need_em:
            ems.append(float(compute_exact_match(pred, refs)))
        if need_f1:
            f1s.append(float(compute_qa_f1(pred, refs)))
        if need_chrf:
            chrfs.append(float(compute_chrf(pred, refs)))
        if need_accuracy:
            accs.append(float(compute_exact_match(pred, refs)))

        m2: Dict[str, float] | None = None
        if need_m2rag:
            # Build a minimal "trace" compatible with m2rag proxy:
            # - final_chunks: oracle context we fed
            # - used_images: oracle images attached
            # - retrieved: include images as candidates so image_recall has signal
            trace = {
                "final_chunks": [ctx] if str(ctx or "").strip() else [],
                "used_images": list(imgs or []),
                "retrieved": [{"images": list(imgs or [])}],
            }
            m2 = compute_m2rag_proxy(
                query=str(ex.query),
                pred=str(pred),
                refs=list(refs or []),
                trace=trace,
                similarity_model=str(mcfg.similarity_model),
            )
            m2 = {k: float(v) for k, v in (m2 or {}).items()}
            m2_flu.append(float(m2.get("m2rag_fluency", 0.0)))
            m2_rel.append(float(m2.get("m2rag_response_relevancy", 0.0)))
            m2_ctxp.append(float(m2.get("m2rag_context_precision", 0.0)))
            m2_faith.append(float(m2.get("m2rag_faithfulness", 0.0)))
            m2_icoh.append(float(m2.get("m2rag_image_coherence", 0.0)))
            m2_ihelp.append(float(m2.get("m2rag_image_helpfulness", 0.0)))
            m2_iref.append(float(m2.get("m2rag_image_reference", 0.0)))
            m2_irec.append(float(m2.get("m2rag_image_recall", 0.0)))

        # Align refs for bert/sim using best rougeL ref
        if need_bertscore or need_similarity:
            best_ref = _best_ref_for_rouge(pred, refs)
            if need_bertscore:
                preds.append(pred)
                refs_for_bert.append(best_ref)
            if need_similarity:
                preds_for_sim.append(pred)
                refs_for_sim.append(best_ref)

        item = {
            "qid": ex.qid,
            "query": ex.query,
            "pred": pred,
            "refs": refs,
            "oracle_doc_id": (odoc.doc_id if odoc is not None else None),
            "used_images": imgs,
        }
        if m2 is not None:
            item["m2rag"] = m2

        if dump_f is not None:
            dump_f.write(json.dumps(item, ensure_ascii=False) + "\n")

        if int(args.print_n) > 0 and printed < int(args.print_n):
            printed += 1
            print("\n" + "=" * 80)
            print(f"[{printed}/{int(args.print_n)}] qid={ex.qid}")
            print(f"[Q] {ex.query}")
            print(f"[PRED] {pred}")
            print(f"[REFS] {refs}")
            if imgs:
                print(f"[USED_IMAGES] {imgs}")
            if m2 is not None:
                # Show the key metrics first
                keys = [
                    "m2rag_overall",
                    "m2rag_response_relevancy",
                    "m2rag_context_precision",
                    "m2rag_faithfulness",
                    "m2rag_image_recall",
                    "m2rag_fluency",
                ]
                head = {k: m2.get(k) for k in keys if k in m2}
                print(f"[M2RAG] {json.dumps(head, ensure_ascii=False)}")

    bert_f1s: List[float] = []
    if need_bertscore:
        bert_f1s = compute_bertscore_f1_batch(preds, refs_for_bert, model=mcfg.bertscore_model, batch_size=16)

    sim_scores: List[float] = []
    if need_similarity:
        sim_scores = compute_similarity_batch(preds_for_sim, refs_for_sim, model=mcfg.similarity_model, batch_size=32)

    per_metric: Dict[str, float] = {
        "rouge1": float(sum(rouge1s) / max(1, len(rouge1s))),
        "rouge2": float(sum(rouge2s) / max(1, len(rouge2s))),
        "rougeL": float(sum(rougeLs) / max(1, len(rougeLs))),
        "meteor": float(sum(meteors) / max(1, len(meteors))) if meteors else 0.0,
        "bleu": float(sum(bleus) / max(1, len(bleus))) if bleus else 0.0,
        "bertscore_f1": float(sum(bert_f1s) / max(1, len(bert_f1s))) if bert_f1s else 0.0,
        "em": float(sum(ems) / max(1, len(ems))) if ems else 0.0,
        "qa_f1": float(sum(f1s) / max(1, len(f1s))) if f1s else 0.0,
        "chrf": float(sum(chrfs) / max(1, len(chrfs))) if chrfs else 0.0,
        "similarity": float(sum(sim_scores) / max(1, len(sim_scores))) if sim_scores else 0.0,
        "accuracy": float(sum(accs) / max(1, len(accs))) if accs else 0.0,
        # M2RAG-style proxy metrics (when requested)
        "m2rag_fluency": float(sum(m2_flu) / max(1, len(m2_flu))) if m2_flu else 0.0,
        "m2rag_response_relevancy": float(sum(m2_rel) / max(1, len(m2_rel))) if m2_rel else 0.0,
        "m2rag_context_precision": float(sum(m2_ctxp) / max(1, len(m2_ctxp))) if m2_ctxp else 0.0,
        "m2rag_faithfulness": float(sum(m2_faith) / max(1, len(m2_faith))) if m2_faith else 0.0,
        "m2rag_image_coherence": float(sum(m2_icoh) / max(1, len(m2_icoh))) if m2_icoh else 0.0,
        "m2rag_image_helpfulness": float(sum(m2_ihelp) / max(1, len(m2_ihelp))) if m2_ihelp else 0.0,
        "m2rag_image_reference": float(sum(m2_iref) / max(1, len(m2_iref))) if m2_iref else 0.0,
        "m2rag_image_recall": float(sum(m2_irec) / max(1, len(m2_irec))) if m2_irec else 0.0,
    }
    per_metric["m2rag_overall"] = float(
        (
            per_metric["m2rag_fluency"]
            + per_metric["m2rag_response_relevancy"]
            + per_metric["m2rag_context_precision"]
            + per_metric["m2rag_faithfulness"]
            + per_metric["m2rag_image_coherence"]
            + per_metric["m2rag_image_helpfulness"]
            + per_metric["m2rag_image_reference"]
            + per_metric["m2rag_image_recall"]
        )
        / 8.0
    )
    reward = float(aggregate_reward(per_metric, mcfg.weights))
    t1 = time.time()

    res = {
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "limit": int(args.limit) if args.limit else 0,
        "llm_base_url": args.llm_base_url,
        "generator_model": args.generator_model,
        "generator_max_tokens": int(args.generator_max_tokens),
        "allow_images": bool(args.allow_images),
        "metrics_weights": mcfg.weights,
        "per_metric": per_metric,
        "weighted_reward": reward,
        "seconds": float(t1 - t0),
        "qas": len(qas),
    }

    if dump_f is not None:
        dump_f.close()
    if out_dir:
        with open(os.path.join(out_dir, "oracle_results.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

