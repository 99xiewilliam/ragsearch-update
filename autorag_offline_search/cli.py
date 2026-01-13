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
    p.add_argument(
        "--embedder_model",
        type=str,
        default="",
        help="Optional override for embedder_model (e.g. BAAI/bge-m3). If set, overrides space/config.",
    )
    p.add_argument(
        "--pipeline",
        type=str,
        default="",
        help="Pipeline category: common|graph|multimodal. If set, overrides space/config.",
    )
    p.add_argument(
        "--generator_model",
        type=str,
        default="",
        help="Generator LLM model id/path/name (prompt is fixed). If set, overrides space/config.",
    )
    p.add_argument(
        "--generator_max_tokens",
        type=int,
        default=0,
        help="Generator max tokens (prompt is fixed). If set (>0), overrides space/config.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--module_logs",
        action="store_true",
        help="If set, print per-module logs inside the pipeline (rewriter/chunking/retrieval/rerank/prune/generate).",
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

    base_cfg: Dict[str, Any] = {}
    if args.config:
        base_cfg = _load_yaml(args.config)
    if args.module_logs:
        base_cfg = {**base_cfg, "module_logs": True}

    if args.dataset_dir:
        run_dataset(
            args.dataset_dir,
            out_dir=args.out_dir,
            algos=algos,
            train_trials=args.train_trials,
            metrics_cfg=metrics_cfg,
            seed=args.seed,
            grpo_group_size=args.grpo_group_size,
            rag_plugin=args.rag_plugin,
            space_plugin=args.space_plugin,
            cache_dir=args.cache_dir,
            llm_base_url=args.llm_base_url,
            embedder_model=args.embedder_model,
            pipeline=args.pipeline,
            generator_model=args.generator_model,
            generator_max_tokens=args.generator_max_tokens,
            config_base=base_cfg,
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
            grpo_group_size=args.grpo_group_size,
            rag_plugin=args.rag_plugin,
            space_plugin=args.space_plugin,
            cache_dir=args.cache_dir,
            llm_base_url=args.llm_base_url,
            embedder_model=args.embedder_model,
            pipeline=args.pipeline,
            generator_model=args.generator_model,
            generator_max_tokens=args.generator_max_tokens,
            config_base=base_cfg,
            verbose=args.verbose,
            log_every=args.log_every,
            show_eval_progress=args.show_eval_progress,
            dump_val_generations=bool(args.dump_val_generations),
            dump_val_limit=int(args.dump_val_limit) if args.dump_val_limit else 0,
        )


if __name__ == "__main__":
    main()


