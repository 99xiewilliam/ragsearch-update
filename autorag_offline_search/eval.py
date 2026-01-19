from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List

from tqdm import tqdm

from .modules.logging_utils import log as _log
from .modules.llm import close_async_clients_for_current_loop
from .m2rag_metrics import compute_m2rag_proxy
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
    need_m2rag = any(str(k).startswith("m2rag_") for k in need)

    preds: List[str] = []
    refs_for_bert: List[str] = []
    preds_for_sim: List[str] = []
    refs_for_sim: List[str] = []
    accs: List[float] = []

    # M2RAG-style metrics (proxy) accumulators
    m2_flu: List[float] = []
    m2_rel: List[float] = []
    m2_ctxp: List[float] = []
    m2_faith: List[float] = []
    m2_icoh: List[float] = []
    m2_ihelp: List[float] = []
    m2_iref: List[float] = []
    m2_irec: List[float] = []

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

    # --- Optional async inference (vLLM/OpenAI compatible) ---
    # Enable when:
    # - request_parallelism=on  -> force async (if rag supports it)
    # - request_parallelism=off -> force sync
    # - request_parallelism=auto (default) -> current behavior: async only when *_max_inflight > 0
    rp = _cfg_str(config, "request_parallelism", "auto").strip().lower()
    has_inflight = (
        _cfg_int(config, "rewriter_max_inflight", 0) > 0
        or _cfg_int(config, "pruner_max_inflight", 0) > 0
        or _cfg_int(config, "generator_max_inflight", 0) > 0
    )
    if rp in {"off", "false", "0", "sync"}:
        want_async = False
    elif rp in {"on", "true", "1", "async"}:
        want_async = hasattr(rag, "answer_async")
    else:
        want_async = bool(has_inflight) and hasattr(rag, "answer_async")

    async_preds: List[str] | None = None
    async_traces: List[Dict | None] | None = None

    if bool(want_async):
        # In async mode:
        # - If user didn't set *_max_inflight but forced request_parallelism=on, use conservative defaults.
        # - If user set *_max_inflight, honor them.
        def _default_inflight(key: str, default: int) -> int:
            v = _cfg_int(config, key, 0)
            if v > 0:
                return int(v)
            if rp in {"on", "true", "1", "async"}:
                return int(default)
            return 1

        rw_n = max(1, _default_inflight("rewriter_max_inflight", 16))
        pr_n = max(1, _default_inflight("pruner_max_inflight", 16))
        ge_n = max(1, _default_inflight("generator_max_inflight", 4))
        llm_sems: Dict[str, object] = {
            "rewriter": asyncio.Semaphore(rw_n),
            "pruner": asyncio.Semaphore(pr_n),
            "generator": asyncio.Semaphore(ge_n),
        }
        # Overall task concurrency: keep modest to avoid CPU thrash building prompts/embeddings.
        task_sem = asyncio.Semaphore(max(1, max(rw_n, pr_n, ge_n)))

        async_preds = ["" for _ in qas]
        async_traces = [None for _ in qas]

        async def _run_one(i: int, ex: QAExample) -> None:
            async with task_sem:
                use_trace = (dump_f is not None or need_m2rag or (debug_enabled and (i + 1) <= debug_n)) and hasattr(
                    rag, "answer_with_trace_async"
                )
                try:
                    if use_trace:
                        tr = await rag.answer_with_trace_async(ex.query, llm_sems=llm_sems)  # type: ignore[attr-defined]
                        async_traces[i] = tr
                        async_preds[i] = str((tr or {}).get("answer", "") or "")
                    else:
                        ans = await rag.answer_async(ex.query, llm_sems=llm_sems)  # type: ignore[attr-defined]
                        async_traces[i] = None
                        async_preds[i] = str(ans or "")
                except Exception as e:
                    # Never crash the whole eval/trial because one request failed.
                    async_preds[i] = ""
                    if use_trace:
                        async_traces[i] = {"query": ex.query, "answer": "", "error": f"{type(e).__name__}: {e}"}
                    else:
                        async_traces[i] = None

        async def _run_all() -> None:
            if show_progress:
                p = tqdm(total=len(qas), desc="eval(async)", disable=not show_progress)
            else:
                p = None
            tasks = [asyncio.create_task(_run_one(i, ex)) for i, ex in enumerate(qas)]
            for fut in asyncio.as_completed(tasks):
                try:
                    await fut
                except Exception:
                    # _run_one already guards, but keep double safety.
                    pass
                if p is not None:
                    p.update(1)
            if p is not None:
                p.close()

        # Run async inference in a fresh loop (CLI context).
        async def _run_all_with_cleanup() -> None:
            try:
                await _run_all()
            finally:
                # Ensure AsyncOpenAI/httpx clients are closed before loop shutdown.
                await close_async_clients_for_current_loop()

        asyncio.run(_run_all_with_cleanup())

    it = tqdm(qas, desc="eval", disable=not show_progress or bool(want_async))
    for ex_idx, ex in enumerate(it, start=1):
        if async_preds is not None:
            pred = str(async_preds[ex_idx - 1] or "")
            trace = async_traces[ex_idx - 1] if async_traces is not None else None
        else:
            trace = None
            use_trace = (dump_f is not None or need_m2rag or (debug_enabled and ex_idx <= debug_n)) and hasattr(
                rag, "answer_with_trace"
            )
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

        if need_m2rag:
            m2 = compute_m2rag_proxy(
                query=str(ex.query),
                pred=str(pred),
                refs=list(refs or []),
                trace=trace or {},
                similarity_model=str(metrics_cfg.similarity_model),
            )
            m2_flu.append(float(m2.get("m2rag_fluency", 0.0)))
            m2_rel.append(float(m2.get("m2rag_response_relevancy", 0.0)))
            m2_ctxp.append(float(m2.get("m2rag_context_precision", 0.0)))
            m2_faith.append(float(m2.get("m2rag_faithfulness", 0.0)))
            m2_icoh.append(float(m2.get("m2rag_image_coherence", 0.0)))
            m2_ihelp.append(float(m2.get("m2rag_image_helpfulness", 0.0)))
            m2_iref.append(float(m2.get("m2rag_image_reference", 0.0)))
            m2_irec.append(float(m2.get("m2rag_image_recall", 0.0)))

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
            if need_m2rag:
                # write per-example M2RAG-style proxy metrics
                ex_metrics.update(
                    {
                        "m2rag_fluency": float(m2_flu[-1]) if m2_flu else 0.0,
                        "m2rag_response_relevancy": float(m2_rel[-1]) if m2_rel else 0.0,
                        "m2rag_context_precision": float(m2_ctxp[-1]) if m2_ctxp else 0.0,
                        "m2rag_faithfulness": float(m2_faith[-1]) if m2_faith else 0.0,
                        "m2rag_image_coherence": float(m2_icoh[-1]) if m2_icoh else 0.0,
                        "m2rag_image_helpfulness": float(m2_ihelp[-1]) if m2_ihelp else 0.0,
                        "m2rag_image_reference": float(m2_iref[-1]) if m2_iref else 0.0,
                        "m2rag_image_recall": float(m2_irec[-1]) if m2_irec else 0.0,
                    }
                )
                # derive per-example overall (mean of 8 components)
                ex_metrics["m2rag_overall"] = float(
                    (
                        ex_metrics["m2rag_fluency"]
                        + ex_metrics["m2rag_response_relevancy"]
                        + ex_metrics["m2rag_context_precision"]
                        + ex_metrics["m2rag_faithfulness"]
                        + ex_metrics["m2rag_image_coherence"]
                        + ex_metrics["m2rag_image_helpfulness"]
                        + ex_metrics["m2rag_image_reference"]
                        + ex_metrics["m2rag_image_recall"]
                    )
                    / 8.0
                )

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
        # M2RAG-style proxy metrics (available when requested via metrics_weights)
        "m2rag_fluency": float(sum(m2_flu) / max(1, len(m2_flu))) if m2_flu else 0.0,
        "m2rag_response_relevancy": float(sum(m2_rel) / max(1, len(m2_rel))) if m2_rel else 0.0,
        "m2rag_context_precision": float(sum(m2_ctxp) / max(1, len(m2_ctxp))) if m2_ctxp else 0.0,
        "m2rag_faithfulness": float(sum(m2_faith) / max(1, len(m2_faith))) if m2_faith else 0.0,
        "m2rag_image_coherence": float(sum(m2_icoh) / max(1, len(m2_icoh))) if m2_icoh else 0.0,
        "m2rag_image_helpfulness": float(sum(m2_ihelp) / max(1, len(m2_ihelp))) if m2_ihelp else 0.0,
        "m2rag_image_reference": float(sum(m2_iref) / max(1, len(m2_iref))) if m2_iref else 0.0,
        "m2rag_image_recall": float(sum(m2_irec) / max(1, len(m2_irec))) if m2_irec else 0.0,
    }
    # derived overall (mean of 8) from aggregated components
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
    reward = float(aggregate_reward(per_metric, metrics_cfg.weights))
    t1 = time.time()
    if dump_f is not None:
        dump_f.close()
    return EvalResult(per_metric=per_metric, weighted_reward=reward), RunStats(
        seconds=t1 - t0, qas=len(qas), chunks=0
    )


