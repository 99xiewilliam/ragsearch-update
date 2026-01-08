from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .experiment import run_dataset
from .metrics import MetricsConfig


@dataclass(frozen=True)
class AutoTrialsConfig:
    """
    Determine train budget (number of evaluated configs) as the search space grows.
    """

    mode: str = "linear"  # linear | sqrt | log2
    scale: float = 5.0
    min_trials: int = 10
    max_trials: int = 80


def auto_train_trials(bits: int, *, cfg: AutoTrialsConfig) -> int:
    """
    Default heuristic (safe-ish): grow budget with bits, capped.
    - linear: scale * bits
    - sqrt: scale * sqrt(2^bits) = scale * 2^(bits/2)
    - log2: scale * log2(2^bits) = scale * bits
    """
    b = int(bits)
    if b <= 0:
        return int(cfg.min_trials)

    if cfg.mode == "linear":
        raw = cfg.scale * b
    elif cfg.mode == "sqrt":
        raw = cfg.scale * (2 ** (b / 2.0))
    elif cfg.mode == "log2":
        raw = cfg.scale * b
    else:
        raise ValueError(f"Unknown auto trials mode: {cfg.mode}")

    t = int(round(raw))
    t = max(int(cfg.min_trials), t)
    t = min(int(cfg.max_trials), t)

    # CRITICAL FIX: Ensure search is meaningful by not covering the whole space
    # For small spaces, limit to 75% of total size so algorithms have to "search"
    space_size = 2**b
    if b <= 6:
        t = min(t, int(space_size * 0.75))
    else:
        # For larger spaces, just ensure we don't exceed the total (though unlikely)
        t = min(t, space_size)

    # Final safety: at least 2 trials to have some comparison, but not more than space_size
    t = max(min(2, space_size), t)

    return t


def run_sweep(
    *,
    datasets_root: str,
    out_dir: str,
    bits_list: Sequence[int],
    algos: Sequence[str],
    metrics_cfg: MetricsConfig,
    seed: int,
    auto_trials_cfg: AutoTrialsConfig,
    rag_plugin: str = "",
    space_plugin: str = "",
    grpo_group_size: int = 0,
    cache_dir: str = "",
    llm_base_url: str = "",
    openai_base_url: str = "",
    openai_api_key: str = "",
    gemini_api_key: str = "",
    embedder_backend: str = "",
    embedder_model: str = "",
    embedder_device: str = "",
    verbose: bool = False,
    log_every: int = 1,
    show_eval_progress: bool = False,
    dump_val_generations: bool = False,
    dump_val_limit: int = 0,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    datasets = []
    for name in sorted(os.listdir(datasets_root)):
        d = os.path.join(datasets_root, name)
        if os.path.isdir(d):
            datasets.append((name, d))

    sweep_rows: List[Dict] = []
    all_runs: Dict = {
        "datasets_root": datasets_root,
        "bits_list": list(bits_list),
        "algos": list(algos),
        "metrics_weights": metrics_cfg.weights,
        "bertscore_model": metrics_cfg.bertscore_model,
        "auto_trials": {
            "mode": auto_trials_cfg.mode,
            "scale": auto_trials_cfg.scale,
            "min_trials": auto_trials_cfg.min_trials,
            "max_trials": auto_trials_cfg.max_trials,
        },
        "runs": {},
    }

    for ds_name, ds_dir in datasets:
        for bits in bits_list:
            train_trials = auto_train_trials(int(bits), cfg=auto_trials_cfg)
            run_out = os.path.join(out_dir, ds_name, f"bits{int(bits)}")
            if verbose:
                print(f"[SWEEP] dataset={ds_name} bits={int(bits)} train_trials={train_trials} out={run_out}")
            summary = run_dataset(
                ds_dir,
                out_dir=run_out,
                algos=algos,
                train_trials=train_trials,
                metrics_cfg=metrics_cfg,
                seed=seed,
                space_bits=int(bits),
                grpo_group_size=grpo_group_size,
                rag_plugin=rag_plugin,
                space_plugin=space_plugin,
                cache_dir=cache_dir or os.path.join(out_dir, ".cache"),
                llm_base_url=llm_base_url,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
                embedder_backend=embedder_backend,
                embedder_model=embedder_model,
                embedder_device=embedder_device,
                verbose=verbose,
                log_every=log_every,
                show_eval_progress=show_eval_progress,
                dump_val_generations=bool(dump_val_generations),
                dump_val_limit=int(dump_val_limit) if dump_val_limit else 0,
            )

            all_runs["runs"].setdefault(ds_name, {})[f"bits{int(bits)}"] = {
                "train_trials": train_trials,
                "out_dir": run_out,
            }

            # flatten results for one big table
            # Write out per-metric columns in a way that matches metrics_weights.
            # This avoids misleading outputs like always writing val_meteor while
            # the reward is actually using chrf (or other metrics).
            metric_keys = list(metrics_cfg.weights.keys())
            for a in algos:
                r = summary["results"][a]
                v = r["validation"]
                metrics_map = v.get("metrics", {}) or {}
                per_metric_cols = {f"val_{k}": float(metrics_map.get(k, 0.0)) for k in metric_keys}
                sweep_rows.append(
                    {
                        "dataset": ds_name,
                        "bits": int(bits),
                        "space_size": int(2 ** int(bits)),
                        "algo": a,
                        "train_trials": int(train_trials),
                        "train_best_reward": float(r["best_train"]["reward"]),
                        "val_reward": float(v["reward"]),
                        **per_metric_cols,
                        "best_config": r["best_train"]["config_str"],
                        "run_dir": run_out,
                    }
                )

    all_runs["rows"] = sweep_rows

    # write master outputs
    with open(os.path.join(out_dir, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(all_runs, f, ensure_ascii=False, indent=2)

    # Dynamic metric columns based on metrics_weights, to match how reward is computed.
    metric_header = [f"val_{k}" for k in metrics_cfg.weights.keys()]
    header = [
        "dataset",
        "bits",
        "space_size",
        "algo",
        "train_trials",
        "train_best_reward",
        "val_reward",
        *metric_header,
        "best_config",
        "run_dir",
    ]
    lines = ["\t".join(header)]
    for row in sweep_rows:
        lines.append(
            "\t".join(
                [
                    str(row[k])
                    for k in header
                ]
            )
        )
    with open(os.path.join(out_dir, "sweep.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return all_runs


