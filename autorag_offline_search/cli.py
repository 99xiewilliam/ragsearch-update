from __future__ import annotations

import argparse
import os
from typing import List

from .experiment import run_dataset
from .metrics import MetricsConfig, parse_weights


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline RAG config search algorithm comparison")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset_dir", type=str, help="Path like /home/xwh/three_dataset_sample_split/bioasq")
    g.add_argument("--all_datasets_root", type=str, help="Path like /home/xwh/three_dataset_sample_split")

    p.add_argument(
        "--algo",
        type=str,
        default="random,greedy,tpe,ucb1,ts,grpo",
        help="Comma-separated algos: random,greedy,tpe,ucb1,ts,grpo,grpo_a2,grpo_a3,portfolio_grpo_ts_tpe,portfolio_grpo_greedy,two_stage_tpe_then_grpo,two_stage_tpe_then_grpo_a2",
    )
    p.add_argument("--train_trials", type=int, default=10)
    p.add_argument(
        "--space_bits",
        type=int,
        default=0,
        help="If set (e.g. 4/6/8/10/12/14), build an exact 2^bits binary search space.",
    )
    p.add_argument(
        "--sweep_bits",
        type=str,
        default="",
        help="Comma-separated bits list for sweep, e.g. '4,6,8,10,12,14'. If set, run a sweep and ignore --space_bits.",
    )
    p.add_argument(
        "--auto_trials_mode",
        type=str,
        default="linear",
        help="Auto train_trials schedule for sweeps: linear | sqrt | log2",
    )
    p.add_argument("--auto_trials_scale", type=float, default=5.0)
    p.add_argument("--auto_trials_min", type=int, default=10)
    p.add_argument("--auto_trials_max", type=int, default=80)
    p.add_argument("--gpus", type=str, default="", help="Comma-separated GPU IDs to use for parallel algorithm execution (e.g. 0,1)")
    p.add_argument(
        "--metrics_weights",
        type=str,
        default="rougeL:0.34,bertscore_f1:0.33,meteor:0.33",
        help='Weight spec, e.g. "rougeL:0.34,bertscore_f1:0.33,meteor:0.33"',
    )
    p.add_argument("--bertscore_model", type=str, default="microsoft/deberta-xlarge-mnli")
    p.add_argument(
        "--grpo_group_size",
        type=int,
        default=0,
        help="Override GRPO group size (0 = auto). Group size consumes budget.",
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
        "--cache_dir",
        type=str,
        default="",
        help="Cache directory for expensive components (chunks/embeddings). Defaults to <out_dir>/.cache",
    )
    p.add_argument(
        "--llm_base_url",
        type=str,
        default="",
        help="Base URL for llm_backend=openai_compat (vLLM), e.g. http://localhost:9000/v1",
    )
    p.add_argument(
        "--openai_base_url",
        type=str,
        default="",
        help="Optional base URL for embedder_backend=openai (rare). Leave empty for hosted OpenAI.",
    )
    p.add_argument(
        "--openai_api_key",
        type=str,
        default="",
        help="Optional OpenAI API key (prefer env OPENAI_API_KEY). Will be masked in outputs.",
    )
    p.add_argument(
        "--gemini_api_key",
        type=str,
        default="",
        help="Optional Gemini API key (prefer env GEMINI_API_KEY/GOOGLE_API_KEY). Will be masked in outputs.",
    )
    p.add_argument(
        "--embedder_backend",
        type=str,
        default="",
        help="Optional override for embedder_backend (local|openai). If set, overrides space/config.",
    )
    p.add_argument(
        "--embedder_model",
        type=str,
        default="",
        help="Optional override for embedder_model (e.g. BAAI/bge-m3). If set, overrides space/config.",
    )
    p.add_argument(
        "--embedder_device",
        type=str,
        default="",
        help="Optional override for embedder_device (cpu|cuda). If set, overrides space/config.",
    )
    p.add_argument("--seed", type=int, default=42)
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
    algos: List[str] = [a.strip() for a in (args.algo or "").split(",") if a.strip()]
    metrics_cfg = MetricsConfig(weights=parse_weights(args.metrics_weights), bertscore_model=args.bertscore_model)

    if args.sweep_bits:
        from .sweep import AutoTrialsConfig, run_sweep

        bits_list = [int(b.strip()) for b in args.sweep_bits.split(",") if b.strip()]
        auto_cfg = AutoTrialsConfig(
            mode=args.auto_trials_mode,
            scale=args.auto_trials_scale,
            min_trials=args.auto_trials_min,
            max_trials=args.auto_trials_max,
        )

        root = args.all_datasets_root or args.dataset_dir
        if not root:
            raise ValueError("sweep requires --all_datasets_root (recommended) or --dataset_dir")

        run_sweep(
            datasets_root=root if args.all_datasets_root else os.path.dirname(root),
            out_dir=args.out_dir,
            bits_list=bits_list,
            algos=algos,
            metrics_cfg=metrics_cfg,
            seed=args.seed,
            auto_trials_cfg=auto_cfg,
            rag_plugin=args.rag_plugin,
            space_plugin=args.space_plugin,
            grpo_group_size=args.grpo_group_size,
            cache_dir=args.cache_dir,
            llm_base_url=args.llm_base_url,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            gemini_api_key=args.gemini_api_key,
            embedder_backend=args.embedder_backend,
            embedder_model=args.embedder_model,
            embedder_device=args.embedder_device,
            verbose=args.verbose,
            log_every=args.log_every,
            show_eval_progress=args.show_eval_progress,
            dump_val_generations=bool(args.dump_val_generations),
            dump_val_limit=int(args.dump_val_limit) if args.dump_val_limit else 0,
        )
        return

    if args.dataset_dir:
        run_dataset(
            args.dataset_dir,
            out_dir=args.out_dir,
            algos=algos,
            train_trials=args.train_trials,
            metrics_cfg=metrics_cfg,
            seed=args.seed,
            space_bits=args.space_bits,
            grpo_group_size=args.grpo_group_size,
            rag_plugin=args.rag_plugin,
            space_plugin=args.space_plugin,
            cache_dir=args.cache_dir,
            llm_base_url=args.llm_base_url,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            gemini_api_key=args.gemini_api_key,
            embedder_backend=args.embedder_backend,
            embedder_model=args.embedder_model,
            embedder_device=args.embedder_device,
            verbose=args.verbose,
            log_every=args.log_every,
            show_eval_progress=args.show_eval_progress,
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
            space_bits=args.space_bits,
            grpo_group_size=args.grpo_group_size,
            rag_plugin=args.rag_plugin,
            space_plugin=args.space_plugin,
            cache_dir=args.cache_dir,
            llm_base_url=args.llm_base_url,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            gemini_api_key=args.gemini_api_key,
            embedder_backend=args.embedder_backend,
            embedder_model=args.embedder_model,
            embedder_device=args.embedder_device,
            verbose=args.verbose,
            log_every=args.log_every,
            show_eval_progress=args.show_eval_progress,
            dump_val_generations=bool(args.dump_val_generations),
            dump_val_limit=int(args.dump_val_limit) if args.dump_val_limit else 0,
        )


if __name__ == "__main__":
    main()


