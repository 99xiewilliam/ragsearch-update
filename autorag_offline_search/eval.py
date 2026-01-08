from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

from tqdm import tqdm

from .metrics import MetricsConfig, aggregate_reward, compute_bertscore_f1_batch, compute_bleu, compute_chrf, compute_exact_match, compute_meteor, compute_qa_f1, compute_rouge
from .plugins.protocols import RagFactory
from .types import Doc, EvalResult, QAExample


@dataclass
class RunStats:
    seconds: float
    qas: int
    chunks: int


def evaluate_config(
    docs: List[Doc],
    qas: List[QAExample],
    config: Dict,
    *,
    metrics_cfg: MetricsConfig,
    rag_factory: RagFactory,
    show_progress: bool = False,
    dump_predictions_path: str | None = None,
    dump_item_prefix: Dict | None = None,
    dump_limit: int = 0,
) -> tuple[EvalResult, RunStats]:
    """
    Evaluate a pipeline config on a split. Reward is weighted aggregation of metrics.
    """
    t0 = time.time()
    rag = rag_factory(docs, config)

    rouge1s, rouge2s, rougeLs = [], [], []
    meteors: List[float] = []
    bleus: List[float] = []
    ems: List[float] = []
    f1s: List[float] = []
    chrfs: List[float] = []

    need = set(metrics_cfg.weights.keys())
    need_bertscore = "bertscore_f1" in need
    need_meteor = "meteor" in need
    need_bleu = "bleu" in need
    need_em = "em" in need
    need_f1 = "qa_f1" in need
    need_chrf = "chrf" in need

    preds: List[str] = []
    refs_for_bert: List[str] = []

    dump_f = None
    dumped = 0
    if dump_predictions_path:
        import json
        import os

        os.makedirs(os.path.dirname(dump_predictions_path) or ".", exist_ok=True)
        dump_f = open(dump_predictions_path, "w", encoding="utf-8")
        if dump_item_prefix:
            dump_f.write(json.dumps({"_type": "meta", **dump_item_prefix}, ensure_ascii=False) + "\n")

    it = tqdm(qas, desc="eval", disable=not show_progress)
    for ex in it:
        trace = None
        if dump_f is not None and hasattr(rag, "answer_with_trace"):
            try:
                trace = rag.answer_with_trace(ex.query)  # type: ignore[attr-defined]
                pred = str((trace or {}).get("answer", "") or "")
            except Exception:
                # Fall back to normal answer() to keep evaluation robust
                trace = None
                pred = rag.answer(ex.query)
        else:
            pred = rag.answer(ex.query)
        refs = ex.generation_gt

        r = compute_rouge(pred, refs)
        rouge1s.append(r["rouge1"])
        rouge2s.append(r["rouge2"])
        rougeLs.append(r["rougeL"])
        if need_meteor:
            meteors.append(compute_meteor(pred, refs))
        if need_bleu:
            bleus.append(compute_bleu(pred, refs))
        if need_em:
            ems.append(compute_exact_match(pred, refs))
        if need_f1:
            f1s.append(compute_qa_f1(pred, refs))
        if need_chrf:
            chrfs.append(compute_chrf(pred, refs))

        if need_bertscore:
            # For BERTScore, pick a single reference (best rougeL against pred) to keep it feasible.
            if refs:
                best_ref = max(refs, key=lambda rr: compute_rouge(pred, [rr])["rougeL"])
            else:
                best_ref = ""
            preds.append(pred)
            refs_for_bert.append(best_ref)

        if dump_f is not None:
            import json

            # Build per-example metrics (only what we can compute cheaply here).
            ex_metrics: Dict[str, float] = {
                "rouge1": float(r["rouge1"]),
                "rouge2": float(r["rouge2"]),
                "rougeL": float(r["rougeL"]),
            }
            if need_meteor:
                ex_metrics["meteor"] = float(meteors[-1])
            if need_bleu:
                ex_metrics["bleu"] = float(bleus[-1])
            if need_em:
                ex_metrics["em"] = float(ems[-1])
            if need_f1:
                ex_metrics["qa_f1"] = float(f1s[-1])
            if need_chrf:
                ex_metrics["chrf"] = float(chrfs[-1])

            item: Dict = {}
            if dump_item_prefix:
                item.update(dump_item_prefix)
            item.update(
                {
                    "qid": ex.qid,
                    "query": ex.query,
                    "pred": pred,
                    "refs": refs,
                    "metrics": ex_metrics,
                }
            )
            if trace is not None:
                item["trace"] = trace
            dump_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            dumped += 1
            if dump_limit and dumped >= int(dump_limit):
                break

    bert_f1s: List[float] = []
    if need_bertscore:
        bert_f1s = compute_bertscore_f1_batch(preds, refs_for_bert, model=metrics_cfg.bertscore_model, batch_size=16)

    per_metric = {
        "rouge1": float(sum(rouge1s) / max(1, len(rouge1s))),
        "rouge2": float(sum(rouge2s) / max(1, len(rouge2s))),
        "rougeL": float(sum(rougeLs) / max(1, len(rougeLs))),
        "meteor": float(sum(meteors) / max(1, len(meteors))) if meteors else 0.0,
        "bleu": float(sum(bleus) / max(1, len(bleus))) if bleus else 0.0,
        "bertscore_f1": float(sum(bert_f1s) / max(1, len(bert_f1s))) if bert_f1s else 0.0,
        "em": float(sum(ems) / max(1, len(ems))) if ems else 0.0,
        "qa_f1": float(sum(f1s) / max(1, len(f1s))) if f1s else 0.0,
        "chrf": float(sum(chrfs) / max(1, len(chrfs))) if chrfs else 0.0,
    }
    reward = float(aggregate_reward(per_metric, metrics_cfg.weights))
    t1 = time.time()
    if dump_f is not None:
        dump_f.close()
    return EvalResult(per_metric=per_metric, weighted_reward=reward), RunStats(
        seconds=t1 - t0, qas=len(qas), chunks=0
    )


