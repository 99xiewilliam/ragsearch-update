from __future__ import annotations

import argparse
import os
from typing import Any, Dict
from typing import List

from .experiment import run_dataset
from .metrics import MetricsConfig, parse_weights


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict, got: {type(data)}")
    return dict(data)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline RAG config search algorithm comparison")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset_dir", type=str, help="Path like /home/xwh/three_dataset_sample_split/bioasq")
    g.add_argument("--all_datasets_root", type=str, help="Path like /home/xwh/three_dataset_sample_split")

    p.add_argument(
        "--algo",
        type=str,
        default="random,greedy,tpe,ucb1,ts,grpo",
        help="Comma-separated algos: random,greedy,tpe,ucb1,ts,grpo,grpo_a2,grpo_a3,portfolio_grpo_ts_tpe,portfolio_grpo_greedy,two_stage_tpe_then_grpo,two_stage_tpe_then_grpo_a2; also supports plugin spec 'module.sub:ClassOrObj' implementing run(SearchInput)->SearchOutput.",
    )
    p.add_argument("--train_trials", type=int, default=10)
    p.add_argument("--gpus", type=str, default="", help="Comma-separated GPU IDs to use for parallel algorithm execution (e.g. 0,1)")
    p.add_argument(
        "--metrics_weights",
        type=str,
        default="bertscore_f1:1,rougeL:1,em:1,meteor:1,similarity:1,accuracy:1",
        help='Weight spec, e.g. "bertscore_f1:1,rougeL:1,em:1,meteor:1,similarity:1,accuracy:1" (aliases supported: bertf1/rougel/exact_match/semilarity/acc).',
    )
    p.add_argument("--bertscore_model", type=str, default="microsoft/deberta-xlarge-mnli")
    p.add_argument(
        "--grpo_group_size",
        type=int,
        default=0,
        help="Override GRPO group size (0 = auto). Group size consumes budget.",
    )
    # TPE (Optuna) early stop controls (optional; budget is still an upper bound)
    p.add_argument(
        "--tpe_patience",
        type=int,
        default=0,
        help="TPE early stop: stop if best reward doesn't improve for this many valid trials (0 = disabled).",
    )
    p.add_argument(
        "--tpe_min_delta",
        type=float,
        default=0.0,
        help="TPE early stop: required improvement over best reward to be considered progress.",
    )
    p.add_argument(
        "--tpe_warmup",
        type=int,
        default=0,
        help="TPE early stop: do not early-stop until at least this many valid trials are collected.",
    )
    # Debug trace (printed via eval.py using answer_with_trace + gold refs)
    p.add_argument(
        "--debug_trace_n",
        type=int,
        default=0,
        help="Print full trace (gold + rewriter/retrieve/rerank/prune/generate) for first N examples in an eval call (0=disable).",
    )
    p.add_argument(
        "--debug_trace_every",
        type=int,
        default=0,
        help="Only print debug trace every N-th trial during training search (0=always when enabled).",
    )
    p.add_argument(
        "--debug_trace_max_chars",
        type=int,
        default=400,
        help="Max chars per printed field in debug trace (truncate long contexts).",
    )
    p.add_argument(
        "--debug_trace_split",
        type=str,
        default="both",
        help="Which split to print debug traces for: train|validation|both.",
    )
    p.add_argument(
        "--rag_plugin",
        type=str,
        default="",
        help="Optional RAG plugin spec 'module:Class' (must implement __init__(docs, config) and answer(query)->str).",
    )
    p.add_argument(
        "--space_plugin",
        type=str,
        default="",
        help="Optional SearchSpace plugin spec 'module:ObjOrClass' (must provide all_configs() and/or space_dict()).",
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional YAML config file path. Provides base config (defaults) for all trials.",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Cache directory for expensive components (chunks/embeddings). Defaults to <out_dir>/.cache",
    )
    p.add_argument(
        "--llm_base_url",
        type=str,
        default="",
        help="Base URL for OpenAI-compatible endpoint (vLLM), e.g. http://localhost:9000/v1",
    )
    # Common config overrides (convenience)
    p.add_argument("--rewriter_model", type=str, default="", help="Override rewriter_model in base config.")
    p.add_argument("--pruner_model", type=str, default="", help="Override pruner_model in base config.")
    p.add_argument("--generator_model", type=str, default="", help="Override generator_model in base config.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--module_logs",
        action="store_true",
        help="If set, print per-module logs inside the pipeline (rewriter/chunking/retrieval/rerank/prune/generate).",
    )
    p.add_argument(
        "--timing_profile",
        action="store_true",
        help="If set, print aggregated timing summary per pipeline stage at the end of each evaluation call.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-trial logs (algo, trial idx, reward, best-so-far).",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="When --verbose, print every N objective evaluations (default 1).",
    )
    p.add_argument(
        "--show_eval_progress",
        action="store_true",
        help="Show tqdm progress bar inside each evaluation (very noisy/slow for many trials).",
    )
    p.add_argument(
        "--show_trial_progress",
        action="store_true",
        help="Show a tqdm progress bar for trial-level objective evaluations (per algorithm).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints under <out_dir>/.checkpoints and skip already evaluated configs.",
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="When --resume, flush checkpoint every N completed trials (default 1).",
    )
    p.add_argument(
        "--request_parallelism",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Request-level parallelism mode for LLM calls during evaluation: auto=enable only when *_max_inflight>0 (current behavior), on=force async request-parallel eval, off=force sync eval.",
    )
    p.add_argument(
        "--dump_val_generations",
        action="store_true",
        help="If set, dump per-example validation predictions (best config only) to <run_dir>/val_predictions_<algo>.jsonl.",
    )
    p.add_argument(
        "--dump_val_limit",
        type=int,
        default=0,
        help="Optional limit for dumped validation examples (0 = no limit).",
    )
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.gpus:
        os.environ["AUTORAG_GPUS"] = args.gpus
    # Convenience: allow providing LLM endpoint via env var.
    # This project assumes a local OpenAI-compatible endpoint (e.g. vLLM).
    if not args.llm_base_url:
        args.llm_base_url = (
            os.environ.get("AUTORAG_LLM_BASE_URL", "")
            or os.environ.get("LLM_BASE_URL", "")
            or os.environ.get("OPENAI_BASE_URL", "")
        )
    algos: List[str] = [a.strip() for a in (args.algo or "").split(",") if a.strip()]
    metrics_cfg = MetricsConfig(weights=parse_weights(args.metrics_weights), bertscore_model=args.bertscore_model)

    base_cfg: Dict[str, Any] = {}
    if args.config:
        base_cfg = _load_yaml(args.config)
    if args.module_logs:
        base_cfg = {**base_cfg, "module_logs": True}
    if args.timing_profile:
        base_cfg = {**base_cfg, "timing_profile": True}
    if args.rewriter_model:
        base_cfg = {**base_cfg, "rewriter_model": str(args.rewriter_model)}
    if args.pruner_model:
        base_cfg = {**base_cfg, "pruner_model": str(args.pruner_model)}
    if args.generator_model:
        base_cfg = {**base_cfg, "generator_model": str(args.generator_model)}

    if args.dataset_dir:
        run_dataset(
            args.dataset_dir,
            out_dir=args.out_dir,
            algos=algos,
            train_trials=args.train_trials,
            metrics_cfg=metrics_cfg,
            seed=args.seed,
            grpo_group_size=args.grpo_group_size,
            tpe_patience=args.tpe_patience,
            tpe_min_delta=args.tpe_min_delta,
            tpe_warmup=args.tpe_warmup,
            debug_trace_n=args.debug_trace_n,
            debug_trace_every=args.debug_trace_every,
            debug_trace_max_chars=args.debug_trace_max_chars,
            debug_trace_split=args.debug_trace_split,
            rag_plugin=args.rag_plugin,
            space_plugin=args.space_plugin,
            cache_dir=args.cache_dir,
            llm_base_url=args.llm_base_url,
            config_base=base_cfg,
            verbose=args.verbose,
            log_every=args.log_every,
            show_eval_progress=args.show_eval_progress,
            show_trial_progress=bool(args.show_trial_progress),
            resume=bool(args.resume),
            checkpoint_every=int(args.checkpoint_every) if args.checkpoint_every else 1,
            request_parallelism=str(args.request_parallelism),
            dump_val_generations=bool(args.dump_val_generations),
            dump_val_limit=int(args.dump_val_limit) if args.dump_val_limit else 0,
        )
        return

    # run all datasets under root
    root = args.all_datasets_root
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        out = os.path.join(args.out_dir, name)
        run_dataset(
            d,
            out_dir=out,
            algos=algos,
            train_trials=args.train_trials,
            metrics_cfg=metrics_cfg,
            seed=args.seed,
            grpo_group_size=args.grpo_group_size,
            tpe_patience=args.tpe_patience,
            tpe_min_delta=args.tpe_min_delta,
            tpe_warmup=args.tpe_warmup,
            debug_trace_n=args.debug_trace_n,
            debug_trace_every=args.debug_trace_every,
            debug_trace_max_chars=args.debug_trace_max_chars,
            debug_trace_split=args.debug_trace_split,
            rag_plugin=args.rag_plugin,
            space_plugin=args.space_plugin,
            cache_dir=args.cache_dir,
            llm_base_url=args.llm_base_url,
            config_base=base_cfg,
            verbose=args.verbose,
            log_every=args.log_every,
            show_eval_progress=args.show_eval_progress,
            show_trial_progress=bool(args.show_trial_progress),
            resume=bool(args.resume),
            checkpoint_every=int(args.checkpoint_every) if args.checkpoint_every else 1,
            request_parallelism=str(args.request_parallelism),
            dump_val_generations=bool(args.dump_val_generations),
            dump_val_limit=int(args.dump_val_limit) if args.dump_val_limit else 0,
        )


if __name__ == "__main__":
    main()


