import json
import os
import dataclasses
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Any

from .algorithms.base import SearchInput, Trial, best_trial
from .algorithms.greedy import GreedyCoordinate
from .algorithms.grpo_lite import GRPO
from .algorithms.mab_ts import ThompsonSamplingGaussian
from .algorithms.mab_ucb import UCB1Bandit
from .algorithms.random_search import RandomSearch
from .data import load_split
from .eval import evaluate_config
from .metrics import MetricsConfig
from .plugins.loader import load_object
from .plugins.protocols import RagFactory
from .search_space import SearchSpace, config_to_str

def _worker_run_algorithm(
    gpu_id: Optional[int],
    algo_name: str,
    kwargs: Dict[str, Any]
) -> tuple[str, Dict]:
    """Helper to run a single algorithm in a separate process with a specific GPU."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    result = run_one_algorithm(algo_name, **kwargs)
    return algo_name, result

def run_one_algorithm(
    algo_name: str,
    *,
    docs_train,
    qas_train,
    docs_val,
    qas_val,
    space,
    metrics_cfg: MetricsConfig,
    rag_factory: RagFactory,
    train_trials: int,
    seed: int,
    grpo_group_size: int = 0,
    tpe_patience: int = 0,
    tpe_min_delta: float = 0.0,
    tpe_warmup: int = 0,
    debug_trace_n: int = 0,
    debug_trace_every: int = 0,
    debug_trace_max_chars: int = 400,
    debug_trace_split: str = "both",
    config_base: Optional[Dict] = None,
    config_inject: Optional[Dict] = None,
    verbose: bool = False,
    log_every: int = 1,
    show_eval_progress: bool = False,
    dump_val_predictions_path: Optional[str] = None,
    dump_val_limit: int = 0,
) -> Dict:
    all_cfgs = space.all_configs()
    if hasattr(space, "space_dict"):
        space_dict = space.space_dict()
    else:
        space_dict = {
            "chunk_size": space.chunk_size,
            "chunk_overlap": space.chunk_overlap,
            "retriever_topk": space.retriever_topk,
            "generator": space.generator,
        }

    seen = 0
    best_so_far: Optional[Trial] = None

    def validate_cfg(cfg: Dict) -> bool:
        # Keep this lightweight: just quick guards to avoid obviously invalid combos.
        if "chunk_overlap" in cfg and "chunk_size" in cfg:
            try:
                if int(cfg["chunk_overlap"]) >= int(cfg["chunk_size"]):
                    return False
            except Exception:
                return False
        return True

    def objective(cfg: Dict) -> Trial:
        nonlocal seen, best_so_far
        # Merge order (pinning):
        # - cfg (from search space / algorithm proposal)
        # - config_base (YAML / user-provided overrides that should take precedence)
        # - config_inject (runtime injection: dataset_id/cache_dir/CLI overrides)
        if config_base:
            cfg = {**cfg, **config_base}
        if config_inject:
            cfg = {**cfg, **config_inject}
        # Debug/trace controls (evaluated per objective call / trial)
        # Note: we keep these keys in config (not in eval kwargs) so Rag implementations can also
        # optionally read them in the future.
        cfg = {
            **cfg,
            "debug_trace_n": int(debug_trace_n),
            "debug_trace_every": int(debug_trace_every),
            "debug_trace_max_chars": int(debug_trace_max_chars),
            "debug_trace_split": str(debug_trace_split),
            "debug_trace_trial": int(seen + 1),
        }
        res, stats = evaluate_config(
            docs_train,
            qas_train,
            cfg,
            metrics_cfg=metrics_cfg,
            rag_factory=rag_factory,
            show_progress=show_eval_progress,
            split_name="train",
        )
        tr = Trial(config=cfg, reward=res.weighted_reward, metrics=res.per_metric, seconds=stats.seconds)
        seen += 1
        if best_so_far is None or tr.reward > best_so_far.reward:
            best_so_far = tr
        if verbose and log_every > 0 and (seen % log_every == 0):
            # Show weighted metrics + reward
            m_str = " ".join(f"{k}={tr.metrics.get(k, 0.0):.6f}" for k in sorted(metrics_cfg.weights.keys()))
            print(
                f"[{algo_name}] trial={seen} reward={tr.reward:.6f} "
                f"({m_str}) "
                f"best={best_so_far.reward:.6f}"
            )
        return tr

    def _make_algo(name: str, *, seed_for_algo: int):
        # Plugin algorithm: "module.sub:ClassOrObj"
        # Convention: prefer implementing `run(SearchInput)->SearchOutput` with a no-arg constructor.
        if ":" in name:
            obj = load_object(name)
            if isinstance(obj, type):
                obj = obj()
            if not callable(getattr(obj, "run", None)):
                raise TypeError(
                    f"Plugin algo {name!r} must implement run(SearchInput)->SearchOutput."
                )
            return obj
        if name == "random":
            return RandomSearch(seed=seed_for_algo)
        if name == "greedy":
            return GreedyCoordinate(seed=seed_for_algo)
        if name == "tpe":
            # Lazy import: optuna is an optional dependency.
            # This allows running grpo-only (or other algos) in environments without optuna.
            from .algorithms.tpe_optuna import TPESearch

            return TPESearch(
                seed=seed_for_algo,
                patience=int(tpe_patience),
                min_delta=float(tpe_min_delta),
                warmup=int(tpe_warmup),
            )
        if name == "ucb1":
            # UCB1 不使用 seed
            return UCB1Bandit()
        if name == "ts":
            return ThompsonSamplingGaussian(seed=seed_for_algo)
        if name in ("grpo", "grpo_a2", "grpo_a3"):
            # If not specified, scale group size mildly with space complexity (bits).
            if grpo_group_size and grpo_group_size > 0:
                gs = int(grpo_group_size)
            else:
                # heuristic: 4..8 depending on number of knobs
                gs = min(8, max(4, len(space_dict) // 2 + 2))
            # Scheme A (GRPO-only): merge strong interaction knobs into one categorical.
            # This keeps the global search space unchanged for other algorithms.
            #
            # grpo     : conservative jointing (only chunker x retriever)
            # grpo_a2  : Scheme A++ (add 2 more strong, *disjoint* interactions)
            joint_pairs = []
            if "chunker" in space_dict and "retriever" in space_dict:
                joint_pairs.append(("chunker", "retriever"))

            if name in ("grpo_a2", "grpo_a3"):
                # A++: add more "gating / XOR-like" interactions that factorized policies struggle with.
                # Keep them disjoint to avoid overlapping removals.
                if "reranker" in space_dict and "rerank_topk" in space_dict:
                    joint_pairs.append(("reranker", "rerank_topk"))
                if "prompt_id" in space_dict and "llm_context_chunks" in space_dict:
                    joint_pairs.append(("prompt_id", "llm_context_chunks"))

                if name == "grpo_a3":
                    # A3: more robust update under noisy objective + "remember & reinforce" best configs.
                    lr = 0.07
                    update_epochs = 3
                    kl_beta = 0.02
                    use_rank_adv = True
                    entropy_beta = 0.01
                    elite_buffer_size = 16
                    elite_beta = 0.20
                else:
                    # A2: slightly more stable PPO update (helps under noisy rewards).
                    lr = 0.10
                    update_epochs = 2
                    kl_beta = 0.03
                    use_rank_adv = False
                    entropy_beta = 0.0
                    elite_buffer_size = 0
                    elite_beta = 0.0
            else:
                lr = 0.20
                update_epochs = 1
                kl_beta = 0.02
                use_rank_adv = False
                entropy_beta = 0.0
                elite_buffer_size = 0
                elite_beta = 0.0

            return GRPO(
                space_dict,
                seed=seed_for_algo,
                lr=lr,
                group_size=gs,
                update_epochs=update_epochs,
                kl_beta=kl_beta,
                joint_pairs=joint_pairs,
                use_rank_adv=use_rank_adv,
                entropy_beta=entropy_beta,
                elite_buffer_size=elite_buffer_size,
                elite_beta=elite_beta,
            )
        raise ValueError(f"Unknown algo: {name}")

    def _run_trials(algo_obj: object, *, b: int, seed_for_algo: int) -> List[Trial]:
        inp = SearchInput(
            objective=objective,
            budget=int(b),
            seed=int(seed_for_algo),
            space=space_dict,
            configs=all_cfgs,
            validate=validate_cfg,
        )
        run = getattr(algo_obj, "run", None)
        if not callable(run):
            raise TypeError(f"Algorithm {getattr(algo_obj, 'name', algo_obj)!r} must implement run().")
        out = run(inp)
        return list(getattr(out, "trials", []))

    # --- Composite / portfolio algorithms ---
    # 目标：只用 train 的 reward 选最终 config，然后在 validation 上评估一次，比较稳定性/平均表现。
    # 注意：总预算 train_trials 不变；组合算法内部会分配预算给子算法。
    if algo_name == "portfolio_grpo_ts_tpe":
        sub_algos = ["grpo", "ts", "tpe"]
        # 平均分预算（至少每个 1 次）
        base = max(1, train_trials // len(sub_algos))
        budgets = [base] * len(sub_algos)
        # 把余数补给前几个（优先给 GRPO/TS 这类更“利用”的）
        rem = max(0, train_trials - sum(budgets))
        for i in range(rem):
            budgets[i % len(budgets)] += 1

        train_trial_list: List[Trial] = []
        for i, (sa, b) in enumerate(zip(sub_algos, budgets)):
            if verbose:
                print(f"[{algo_name}] sub_algo={sa} budget={b}")
            # 用不同 seed 偏移，减少完全相关的随机性
            algo = _make_algo(sa, seed_for_algo=int(seed) + 1000 + i)
            train_trial_list.extend(_run_trials(algo, b=int(b), seed_for_algo=int(seed) + 1000 + i))

    elif algo_name == "portfolio_grpo_greedy":
        sub_algos = ["grpo", "greedy"]
        base = max(1, train_trials // len(sub_algos))
        budgets = [base] * len(sub_algos)
        rem = max(0, train_trials - sum(budgets))
        for i in range(rem):
            budgets[i % len(budgets)] += 1

        train_trial_list = []
        for i, (sa, b) in enumerate(zip(sub_algos, budgets)):
            if verbose:
                print(f"[{algo_name}] sub_algo={sa} budget={b}")
            algo = _make_algo(sa, seed_for_algo=int(seed) + 3000 + i)
            train_trial_list.extend(_run_trials(algo, b=int(b), seed_for_algo=int(seed) + 3000 + i))

    elif algo_name == "two_stage_tpe_then_grpo":
        # 两阶段：TPE 探索 -> GRPO 精炼（不做 warm-start，仅顺序分配预算）
        if train_trials <= 1:
            b_tpe, b_grpo = 1, 0
        else:
            b_tpe = max(1, int(round(train_trials * 0.3)))
            b_grpo = max(0, train_trials - b_tpe)

        # 如果你想更“偏向 GRPO”，可以把 0.3 调到 0.2

        train_trial_list = []
        if verbose:
            print(f"[{algo_name}] stage=tpe budget={b_tpe}")
        algo_tpe = _make_algo("tpe", seed_for_algo=int(seed) + 2000)
        train_trial_list.extend(_run_trials(algo_tpe, b=int(b_tpe), seed_for_algo=int(seed) + 2000))
        if b_grpo > 0:
            if verbose:
                print(f"[{algo_name}] stage=grpo budget={b_grpo}")
            algo_grpo = _make_algo("grpo", seed_for_algo=int(seed) + 2001)
            train_trial_list.extend(_run_trials(algo_grpo, b=int(b_grpo), seed_for_algo=int(seed) + 2001))

    elif algo_name == "two_stage_tpe_then_grpo_a2":
        # Two-stage: TPE explore -> GRPO-A++ exploit (higher chance to beat pure TPE under interactions)
        if train_trials <= 1:
            b_tpe, b_grpo = 1, 0
        else:
            b_tpe = max(1, int(round(train_trials * 0.3)))
            b_grpo = max(0, train_trials - b_tpe)

        train_trial_list = []
        if verbose:
            print(f"[{algo_name}] stage=tpe budget={b_tpe}")
        algo_tpe = _make_algo("tpe", seed_for_algo=int(seed) + 2100)
        train_trial_list.extend(_run_trials(algo_tpe, b=int(b_tpe), seed_for_algo=int(seed) + 2100))
        if b_grpo > 0:
            if verbose:
                print(f"[{algo_name}] stage=grpo_a2 budget={b_grpo}")
            algo_grpo = _make_algo("grpo_a2", seed_for_algo=int(seed) + 2101)
            train_trial_list.extend(_run_trials(algo_grpo, b=int(b_grpo), seed_for_algo=int(seed) + 2101))

    else:
        algo = _make_algo(algo_name, seed_for_algo=int(seed))
        train_trial_list = _run_trials(algo, b=int(train_trials), seed_for_algo=int(seed))

    best = best_trial(train_trial_list)
    if best is None:
        raise RuntimeError("No trials produced")

    # GraphRAG has been removed from this repo; keep these fields for backward-compatible
    # result schema (always None).
    best_global: Optional[Trial] = None

    # validate once with best config
    val_res, val_stats = evaluate_config(
        docs_val,
        qas_val,
        best.config,
        metrics_cfg=metrics_cfg,
        rag_factory=rag_factory,
        show_progress=show_eval_progress,
        split_name="validation",
        dump_predictions_path=dump_val_predictions_path,
        dump_item_prefix={
            "split": "validation",
            "algo": algo_name,
            "config_str": config_to_str(best.config),
        }
        if dump_val_predictions_path
        else None,
        dump_limit=int(dump_val_limit) if dump_val_limit else 0,
    )

    val_global = None

    return {
        "algo": algo_name,
        "train_trials": [
            {
                "reward": t.reward,
                "seconds": t.seconds,
                "metrics": t.metrics,
                "config": t.config,
                "config_str": config_to_str(t.config),
            }
            for t in train_trial_list
        ],
        "best_train": {
            "reward": best.reward,
            "seconds": best.seconds,
            "metrics": best.metrics,
            "config": best.config,
            "config_str": config_to_str(best.config),
        },
        "best_train_global": (
            {
                "reward": best_global.reward,
                "seconds": best_global.seconds,
                "metrics": best_global.metrics,
                "config": best_global.config,
                "config_str": config_to_str(best_global.config),
                "delta_vs_best": float(best.reward - best_global.reward),
            }
            if best_global is not None
            else None
        ),
        "validation": {
            "reward": val_res.weighted_reward,
            "seconds": val_stats.seconds,
            "metrics": val_res.per_metric,
            "config": best.config,
            "config_str": config_to_str(best.config),
        },
        "validation_global": val_global,
    }


def run_dataset(
    dataset_dir: str,
    *,
    out_dir: str,
    algos: Sequence[str],
    train_trials: int,
    metrics_cfg: MetricsConfig,
    seed: int = 42,
    grpo_group_size: int = 0,
    tpe_patience: int = 0,
    tpe_min_delta: float = 0.0,
    tpe_warmup: int = 0,
    debug_trace_n: int = 0,
    debug_trace_every: int = 0,
    debug_trace_max_chars: int = 400,
    debug_trace_split: str = "both",
    rag_plugin: str = "",
    space_plugin: str = "",
    cache_dir: str = "",
    llm_base_url: str = "",
    embedder_model: str = "",
    pipeline: str = "",
    generator_model: str = "",
    generator_max_tokens: int = 0,
    bm25_weight: float = -1.0,
    config_base: Optional[Dict] = None,
    verbose: bool = False,
    log_every: int = 1,
    show_eval_progress: bool = False,
    dump_val_generations: bool = False,
    dump_val_limit: int = 0,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    docs_train, qas_train = load_split(dataset_dir, "train")
    docs_val, qas_val = load_split(dataset_dir, "validation")

    # Load plugins (optional). If not provided, use built-ins.
    rag_factory: RagFactory
    if rag_plugin:
        rag_factory = load_object(rag_plugin)
    else:
        # New default: unified modular RAG (common/multimodal).
        from .plugins.rag import UnifiedRag

        rag_factory = UnifiedRag

    if space_plugin:
        space = load_object(space_plugin)
        # allow plugin to be a class or instance
        if isinstance(space, type):
            space = space()
    else:
        space = SearchSpace()

    # YAML config can provide:
    # - fixed base config (scalars): merged into each trial config
    # - space constraints (lists): restrict the search space candidates
    fixed_base: Dict[str, Any] = dict(config_base or {})
    space_overrides: Dict[str, Sequence] = {}

    # Optional structured form:
    #   base: { ... }
    #   space: { key: [candidates...] }
    if isinstance(fixed_base.get("base"), dict):
        base = dict(fixed_base.pop("base") or {})
        fixed_base = {**fixed_base, **base}
    if isinstance(fixed_base.get("space"), dict):
        space_overrides.update(dict(fixed_base.pop("space") or {}))

    # Backward-compatible shorthand: list values at top-level are treated as space constraints.
    for k in list(fixed_base.keys()):
        v = fixed_base.get(k)
        if isinstance(v, (list, tuple)):
            space_overrides[k] = list(v)
            fixed_base.pop(k, None)

    def _expand_range(v: Any) -> Optional[Sequence]:
        """
        Allow YAML to specify a numeric range and expand to a discrete grid.

        Examples:
          bm25_weight:
            low: 0
            high: 1
            steps: 11
        """
        if not isinstance(v, dict):
            return None
        # accept both low/high and min/max
        if ("low" in v and "high" in v) or ("min" in v and "max" in v):
            lo = float(v.get("low", v.get("min")))
            hi = float(v.get("high", v.get("max")))
            steps = int(v.get("steps", v.get("n", 11)))
            if steps <= 0:
                raise ValueError("range steps must be > 0")
            if steps == 1:
                return (lo,)
            # inclusive linspace without numpy
            return tuple(lo + (hi - lo) * i / (steps - 1) for i in range(steps))
        return None

    if space_overrides:
        pipeline_hint = str(fixed_base.get("pipeline") or "")
        # If user pins pipeline in base, also restrict the enumerated search space pipeline.
        # Otherwise, configs generated for other pipelines may leak in, and then get "re-labeled"
        # as multimodal via merge, causing model mismatches (e.g., qwen3 text model in multimodal).
        if pipeline_hint in {"common", "multimodal"} and hasattr(space, "pipeline"):
            space = dataclasses.replace(space, pipeline=(pipeline_hint,))
        for k, v in list(space_overrides.items()):
            expanded = _expand_range(v)
            if expanded is not None:
                vv = tuple(expanded)
            else:
                vv = tuple(v) if isinstance(v, (list, tuple)) else (v,)
            # pipeline-aware mapping for multimodal vs common
            if k == "embedder_model":
                if pipeline_hint == "multimodal" and hasattr(space, "multimodal_embedder_model"):
                    space = dataclasses.replace(space, multimodal_embedder_model=vv)
                elif hasattr(space, "embedder_model"):
                    space = dataclasses.replace(space, embedder_model=vv)
                continue
            if k == "reranker_model":
                if pipeline_hint == "multimodal" and hasattr(space, "multimodal_reranker_model"):
                    space = dataclasses.replace(space, multimodal_reranker_model=vv)
                elif hasattr(space, "reranker_model"):
                    space = dataclasses.replace(space, reranker_model=vv)
                continue
            if k == "rewriter_model":
                if pipeline_hint == "multimodal" and hasattr(space, "multimodal_rewriter_model"):
                    space = dataclasses.replace(space, multimodal_rewriter_model=vv)
                elif hasattr(space, "rewriter_model"):
                    space = dataclasses.replace(space, rewriter_model=vv)
                continue
            if k == "pruner_model":
                if pipeline_hint == "multimodal" and hasattr(space, "multimodal_pruner_model"):
                    space = dataclasses.replace(space, multimodal_pruner_model=vv)
                elif hasattr(space, "pruner_model"):
                    space = dataclasses.replace(space, pruner_model=vv)
                continue
            if hasattr(space, k):
                space = dataclasses.replace(space, **{k: vv})
                continue
    else:
        # No explicit space overrides, but still honor pinned pipeline in base (if any).
        pipeline_hint = str(fixed_base.get("pipeline") or "")
        if pipeline_hint in {"common", "multimodal"} and hasattr(space, "pipeline"):
            space = dataclasses.replace(space, pipeline=(pipeline_hint,))

    results: Dict[str, Dict] = {}
    inject = {
        "dataset_id": os.path.basename(dataset_dir.rstrip("/")),
        "cache_dir": cache_dir or os.path.join(out_dir, ".cache"),
    }
    if verbose:
        inject["verbose"] = True
    if llm_base_url:
        inject["llm_base_url"] = llm_base_url
    if pipeline:
        inject["pipeline"] = str(pipeline)
    if embedder_model:
        inject["embedder_model"] = embedder_model
    if generator_model:
        inject["generator_model"] = str(generator_model)
    if generator_max_tokens and int(generator_max_tokens) > 0:
        inject["generator_max_tokens"] = int(generator_max_tokens)
    if bm25_weight is not None and float(bm25_weight) >= 0.0:
        inject["bm25_weight"] = float(bm25_weight)

    # Multi-GPU Parallel execution
    # Determine GPU IDs to use
    gpu_list: List[Optional[int]] = [None]
    env_gpus = os.environ.get("AUTORAG_GPUS")
    if env_gpus:
        gpu_list = [int(g.strip()) for g in env_gpus.split(",")]

    num_workers = len(gpu_list)
    
    tasks = []
    for a in algos:
        kwargs = {
            "docs_train": docs_train,
            "qas_train": qas_train,
            "docs_val": docs_val,
            "qas_val": qas_val,
            "space": space,
            "metrics_cfg": metrics_cfg,
            "rag_factory": rag_factory,
            "train_trials": train_trials,
            "seed": seed,
            "grpo_group_size": grpo_group_size,
            "config_base": fixed_base,
            "config_inject": inject,
            "verbose": verbose,
            "log_every": log_every,
            "show_eval_progress": show_eval_progress,
        }
        tasks.append(a)

    if num_workers > 1:
        if verbose:
            print(f"[DATASET] Running {len(algos)} algorithms in parallel on GPUs: {gpu_list}")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # We need a way to cycle through GPUs
            futures = []
            for i, a in enumerate(algos):
                gid = gpu_list[i % num_workers]
                dump_path = os.path.join(out_dir, f"val_predictions_{a}.jsonl") if dump_val_generations else None
                futures.append(executor.submit(_worker_run_algorithm, gid, a, {
                    "docs_train": docs_train,
                    "qas_train": qas_train,
                    "docs_val": docs_val,
                    "qas_val": qas_val,
                    "space": space,
                    "metrics_cfg": metrics_cfg,
                    "rag_factory": rag_factory,
                    "train_trials": train_trials,
                    "seed": seed,
                    "grpo_group_size": grpo_group_size,
                    "tpe_patience": tpe_patience,
                    "tpe_min_delta": tpe_min_delta,
                    "tpe_warmup": tpe_warmup,
                    "debug_trace_n": debug_trace_n,
                    "debug_trace_every": debug_trace_every,
                    "debug_trace_max_chars": debug_trace_max_chars,
                    "debug_trace_split": debug_trace_split,
                    "config_base": fixed_base,
                    "config_inject": inject,
                    "verbose": verbose,
                    "log_every": log_every,
                    "show_eval_progress": show_eval_progress,
                    "dump_val_predictions_path": dump_path,
                    "dump_val_limit": int(dump_val_limit) if dump_val_limit else 0,
                }))
            
            for future in futures:
                name, res = future.result()
                results[name] = res
    else:
        # Sequential execution (original behavior)
        for a in algos:
            if verbose:
                print(f"[DATASET] algo={a} train_trials={train_trials}")
            dump_path = os.path.join(out_dir, f"val_predictions_{a}.jsonl") if dump_val_generations else None
            results[a] = run_one_algorithm(
                a,
                docs_train=docs_train,
                qas_train=qas_train,
                docs_val=docs_val,
                qas_val=qas_val,
                space=space,
                metrics_cfg=metrics_cfg,
                rag_factory=rag_factory,
                train_trials=train_trials,
                seed=seed,
                grpo_group_size=grpo_group_size,
                tpe_patience=tpe_patience,
                tpe_min_delta=tpe_min_delta,
                tpe_warmup=tpe_warmup,
                debug_trace_n=debug_trace_n,
                debug_trace_every=debug_trace_every,
                debug_trace_max_chars=debug_trace_max_chars,
                debug_trace_split=debug_trace_split,
                config_base=fixed_base,
                config_inject=inject,
                verbose=verbose,
                log_every=log_every,
                show_eval_progress=show_eval_progress,
                dump_val_predictions_path=dump_path,
                dump_val_limit=int(dump_val_limit) if dump_val_limit else 0,
            )

    summary = {
        "dataset_dir": dataset_dir,
        "algos": list(algos),
        "train_trials_budget": train_trials,
        "metrics_weights": metrics_cfg.weights,
        "bertscore_model": metrics_cfg.bertscore_model,
        "rag_plugin": rag_plugin or "autorag_offline_search.plugins.rag:UnifiedRag",
        "space_plugin": space_plugin or None,
        "results": results,
    }

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # write a small CSV-ish summary (tab-separated) for quick comparison
    m_keys = sorted(metrics_cfg.weights.keys())
    header = [
        "algo",
        "train_best_reward",
        "train_best_global_reward",
        "train_delta_vs_global",
        "val_reward",
        "val_global_reward",
        "val_delta_vs_global",
    ] + [f"val_{k}" for k in m_keys] + ["best_config"]
    lines = ["\t".join(header)]
    
    for a in algos:
        r = results[a]
        bt = r["best_train"]["reward"]
        bg = r.get("best_train_global") or None
        bg_reward = float(bg["reward"]) if isinstance(bg, dict) else 0.0
        # NOTE: we report +delta when best beats global baseline.
        best_minus_global = float(bt - bg_reward) if bg is not None else 0.0
        v = r["validation"]
        vg = r.get("validation_global") or None
        vg_reward = float(vg["reward"]) if isinstance(vg, dict) else 0.0
        val_minus_global = float(v["reward"] - vg_reward) if vg is not None else 0.0
        
        row = [
            a,
            f"{bt:.6f}",
            f"{bg_reward:.6f}",
            f"{best_minus_global:.6f}",
            f"{v['reward']:.6f}",
            f"{vg_reward:.6f}",
            f"{val_minus_global:.6f}",
        ]
        for k in m_keys:
            row.append(f"{v['metrics'].get(k, 0.0):.6f}")
        row.append(r["best_train"]["config_str"])
        
        lines.append("\t".join(row))
    
    with open(os.path.join(out_dir, "summary.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return summary


