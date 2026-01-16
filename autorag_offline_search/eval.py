from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

from tqdm import tqdm

from .modules.logging_utils import log as _log
from .metrics import (
    MetricsConfig,
    aggregate_reward,
    compute_bertscore_f1_batch,
    compute_bleu,
    compute_chrf,
    compute_exact_match,
    compute_meteor,
    compute_qa_f1,
    compute_rouge,
    compute_similarity_batch,
)
from .plugins.protocols import RagFactory
from .types import Doc, EvalResult, QAExample


@dataclass
class RunStats:
    seconds: float
    qas: int
    chunks: int


def _cfg_int(cfg: Dict, key: str, default: int = 0) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return int(default)


def _cfg_str(cfg: Dict, key: str, default: str = "") -> str:
    try:
        v = cfg.get(key, default)
        return str(v) if v is not None else str(default)
    except Exception:
        return str(default)


def _short(s: str, n: int) -> str:
    t = str(s or "")
    return (t[:n] + "...") if n > 0 and len(t) > n else t


def evaluate_config(
    docs: List[Doc],
    qas: List[QAExample],
    config: Dict,
    *,
    metrics_cfg: MetricsConfig,
    rag_factory: RagFactory,
    show_progress: bool = False,
    split_name: str = "",
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
    need_similarity = "similarity" in need
    need_accuracy = "accuracy" in need

    preds: List[str] = []
    refs_for_bert: List[str] = []
    preds_for_sim: List[str] = []
    refs_for_sim: List[str] = []
    accs: List[float] = []

    dump_f = None
    dumped = 0
    if dump_predictions_path:
        import json
        import os

        os.makedirs(os.path.dirname(dump_predictions_path) or ".", exist_ok=True)
        dump_f = open(dump_predictions_path, "w", encoding="utf-8")
        if dump_item_prefix:
            dump_f.write(json.dumps({"_type": "meta", **dump_item_prefix}, ensure_ascii=False) + "\n")

    # Debug/trace printing controls (intentionally independent from module_logs to avoid spam in TPE).
    debug_n = max(0, _cfg_int(config, "debug_trace_n", 0))
    debug_every = max(0, _cfg_int(config, "debug_trace_every", 0))
    debug_max_chars = max(0, _cfg_int(config, "debug_trace_max_chars", 400))
    debug_split = _cfg_str(config, "debug_trace_split", "both").strip().lower()
    debug_trial = _cfg_int(config, "debug_trace_trial", 0)
    debug_enabled = debug_n > 0
    if debug_enabled:
        if debug_split not in {"train", "validation", "val", "both"}:
            debug_split = "both"
        if split_name and debug_split != "both":
            if debug_split == "val":
                debug_split = "validation"
            if debug_split != str(split_name).strip().lower():
                debug_enabled = False
        # optional frequency gating by trial index (set by experiment)
        if debug_enabled and debug_every > 0 and debug_trial > 0 and (debug_trial % debug_every != 0):
            debug_enabled = False

    it = tqdm(qas, desc="eval", disable=not show_progress)
    for ex_idx, ex in enumerate(it, start=1):
        trace = None
        use_trace = (dump_f is not None or (debug_enabled and ex_idx <= debug_n)) and hasattr(rag, "answer_with_trace")
        if use_trace:
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

        if need_accuracy:
            # For now, treat accuracy as strict exact match over references.
            accs.append(compute_exact_match(pred, refs))

        # Optional debug trace print (few examples only).
        if debug_enabled and ex_idx <= debug_n:
            ex_metrics_dbg: Dict[str, float] = {
                "rougeL": float(r["rougeL"]),
                "em": float(ems[-1]) if need_em else float(compute_exact_match(pred, refs)),
                "qa_f1": float(f1s[-1]) if need_f1 else float(compute_qa_f1(pred, refs)),
            }
            head = f"[TRACE] split={split_name or '?'} trial={debug_trial or 0} ex={ex_idx} qid={ex.qid}"
            _log(config, head)
            # Print high-signal config knobs so it's obvious which modules are enabled in this trial.
            cfg_dbg = {
                "pipeline": _cfg_str(config, "pipeline", ""),
                "bm25_weight": config.get("bm25_weight", None),
                "retriever": _cfg_str(config, "retriever", ""),
                "retriever_topk": _cfg_int(config, "retriever_topk", 0),
                "chunking_enabled": bool(config.get("chunking_enabled", False)),
                "chunk_size": _cfg_int(config, "chunk_size", 0),
                "embedding_enabled": bool(config.get("embedding_enabled", False)),
                "embedder_model": _cfg_str(config, "embedder_model", ""),
                "rewriter_enabled": bool(config.get("rewriter_enabled", False)),
                "rewriter_model": _cfg_str(config, "rewriter_model", ""),
                "reranker_enabled": bool(config.get("reranker_enabled", False)),
                "reranker_model": _cfg_str(config, "reranker_model", ""),
                "rerank_topk": _cfg_int(config, "rerank_topk", 0),
                "pruner_enabled": bool(config.get("pruner_enabled", False)),
                "pruner_model": _cfg_str(config, "pruner_model", ""),
                "generator_model": _cfg_str(config, "generator_model", ""),
            }
            _log(config, f"[TRACE] cfg={cfg_dbg}")
            _log(config, f"[TRACE] query={_short(ex.query, debug_max_chars)}")
            _log(config, f"[TRACE] gold={_short(str(refs), debug_max_chars)}")
            _log(config, f"[TRACE] pred={_short(pred, debug_max_chars)}")
            _log(config, f"[TRACE] metrics={ex_metrics_dbg}")
            if isinstance(trace, dict):
                if "rewritten_query" in trace:
                    _log(config, f"[TRACE] rewritten_query={_short(str(trace.get('rewritten_query','')), debug_max_chars)}")
                if "retrieved" in trace:
                    rr = trace.get("retrieved") or []
                    if isinstance(rr, list) and rr:
                        _log(config, "[TRACE] retrieved_top:")
                        for it2 in rr[: min(10, len(rr))]:
                            if isinstance(it2, dict):
                                ii = it2.get("i", "")
                                tx = _short(str(it2.get("text", "") or ""), debug_max_chars)
                                _log(config, f"  - i={ii} text={tx}")
                if "reranked_indices" in trace:
                    _log(config, f"[TRACE] reranked_indices={trace.get('reranked_indices')}")
                if "final_chunks" in trace:
                    fc = trace.get("final_chunks") or []
                    if isinstance(fc, list) and fc:
                        _log(config, f"[TRACE] final_chunks(n={len(fc)}) preview0={_short(str(fc[0]), debug_max_chars)}")

        if need_bertscore or need_similarity:
            # Pick a single reference (best rougeL against pred) to keep it feasible.
            if refs:
                best_ref = max(refs, key=lambda rr: compute_rouge(pred, [rr])["rougeL"])
            else:
                best_ref = ""
            if need_bertscore:
                preds.append(pred)
                refs_for_bert.append(best_ref)
            if need_similarity:
                preds_for_sim.append(pred)
                refs_for_sim.append(best_ref)

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

    sim_scores: List[float] = []
    if need_similarity:
        sim_scores = compute_similarity_batch(
            preds_for_sim,
            refs_for_sim,
            model=metrics_cfg.similarity_model,
            batch_size=32,
        )

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
        "similarity": float(sum(sim_scores) / max(1, len(sim_scores))) if sim_scores else 0.0,
        "accuracy": float(sum(accs) / max(1, len(accs))) if accs else 0.0,
    }
    reward = float(aggregate_reward(per_metric, metrics_cfg.weights))
    t1 = time.time()
    if dump_f is not None:
        dump_f.close()
    return EvalResult(per_metric=per_metric, weighted_reward=reward), RunStats(
        seconds=t1 - t0, qas=len(qas), chunks=0
    )


