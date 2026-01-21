#!/usr/bin/env python3
"""
分析 trial 结果里：
1) 模块级影响（控制变量思想）：只看模块 on/off 对 reward 的影响幅度
2) （可选）超参级重要性：RandomForest 的 feature_importances_

输入：train_trials_*.jsonl（每行包含 config + reward；可含一行 _type=meta）
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


MODULE_KEYS: List[str] = [
    "rewriter_enabled",
    "chunking_enabled",
    "reranker_enabled",
    "pruner_enabled",
]


def _load_trials(jsonl_path: str) -> pd.DataFrame:
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return pd.DataFrame()

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_type" in obj:
                continue
            row = dict(obj.get("config", {}))
            row["_reward"] = obj.get("reward", 0.0)
            data.append(row)

    if not data:
        print("No trial data found in file.")
        return pd.DataFrame()

    return pd.DataFrame(data)


def _ensure_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.fillna(False).astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y", "on"])


def analyze_module_effect(df: pd.DataFrame, *, source: str, reward_col: str = "_reward") -> None:
    """
    模块级影响分析（更接近“控制变量”）：
    - 直接对比：模块开 vs 关 的平均 reward 差
    - 分层对比：按“其他模块开关组合”分层，层内做 on-off，对各层按样本量加权平均
    """
    if df is None or df.empty:
        return
    if reward_col not in df.columns:
        print(f"[module] Missing reward column: {reward_col}")
        return

    work = df.copy()
    for k in MODULE_KEYS:
        if k not in work.columns:
            work[k] = False
        work[k] = _ensure_bool(work[k])

    y = pd.to_numeric(work[reward_col], errors="coerce").fillna(0.0)

    print(f"\n===== Module Effect Analysis (on/off impact) =====")
    print(f"Source: {source}")
    print(f"Total Trials: {len(work)}")
    print(f"Reward column: {reward_col}")
    print("-" * 86)
    print(f"{'Module':<18} | {'n(off)':>6} {'mean(off)':>10} || {'n(on)':>6} {'mean(on)':>10} || {'delta(on-off)':>14}")
    print("-" * 86)

    for m in MODULE_KEYS:
        on_mask = work[m] == True
        off_mask = ~on_mask
        n_on = int(on_mask.sum())
        n_off = int(off_mask.sum())
        mean_on = float(y[on_mask].mean()) if n_on > 0 else float("nan")
        mean_off = float(y[off_mask].mean()) if n_off > 0 else float("nan")
        delta = (mean_on - mean_off) if (n_on > 0 and n_off > 0) else float("nan")
        print(f"{m:<18} | {n_off:>6} {mean_off:>10.4f} || {n_on:>6} {mean_on:>10.4f} || {delta:>14.4f}")

    print("-" * 86)
    print("注：上表 delta 是“直接对比”，没有控制其他模块/超参（可能有混杂）。")

    print(f"\n===== Stratified Module Effect (control other modules) =====")
    print("做法：对某个模块 m，按其他模块(开关组合)分层；只在层内做 on-off 对比；再按层样本数加权平均。")
    print("-" * 86)
    print(f"{'Module':<18} | {'usable strata':>12} | {'weighted delta':>14}")
    print("-" * 86)

    for m in MODULE_KEYS:
        others = [k for k in MODULE_KEYS if k != m]
        strata = work[others].astype(int).agg(lambda r: "".join(map(str, r.values.tolist())), axis=1)

        deltas = []
        weights = []
        usable = 0
        for key in strata.unique():
            idx = strata == key
            m_s = work.loc[idx, m]
            if m_s.nunique(dropna=False) < 2:
                continue
            usable += 1
            y_s = y[idx]
            mean_on = float(y_s[m_s == True].mean())
            mean_off = float(y_s[m_s == False].mean())
            deltas.append(mean_on - mean_off)
            weights.append(int(idx.sum()))

        if weights:
            wdelta = float(np.average(np.asarray(deltas, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))
        else:
            wdelta = float("nan")

        print(f"{m:<18} | {usable:>12} | {wdelta:>14.4f}")

    print("-" * 86)
    print("解释：weighted delta 越大，说明“打开该模块”对 reward 的平均增益越大（在控制其他模块开关后）。")
    print("注意：这仍不控制模块内部子超参（如 topk/chunk_size），后续可做二阶段分析。\n")


def analyze_hparam_importance(df: pd.DataFrame, *, source: str) -> None:
    """
    原脚本逻辑：用 RandomForest 预测 reward，并输出 feature_importances_ 作为超参重要性。
    """
    if df is None or df.empty:
        return

    target = "_reward"
    features = [c for c in df.columns if c != target and not c.startswith("_")]
    if not features:
        print("[hparam] No features found.")
        return

    X = df[features].copy()
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0)

    for col in X.columns:
        if X[col].dtype == "object" or isinstance(X[col].iloc[0], bool):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n===== Hyperparameter Importance Analysis (RandomForest) =====")
    print(f"Source: {source}")
    print(f"Total Trials: {len(df)}")
    print("-" * 45)
    print(f"{'Parameter':<30} | {'Importance':<10}")
    print("-" * 45)
    for i in range(len(features)):
        print(f"{features[indices[i]]:<30} | {importances[indices[i]]:.4f}")
    print("-" * 45)
    print("Interpretation: Higher value means the parameter has a stronger influence on the final reward.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to train_trials_*.jsonl")
    parser.add_argument(
        "--mode",
        type=str,
        default="module",
        choices=["module", "hparam", "both"],
        help="module=模块开关影响；hparam=随机森林超参重要性；both=两者都跑。",
    )
    parser.add_argument(
        "--reward_col",
        type=str,
        default="_reward",
        help="Reward column name in the flattened dataframe (default: _reward).",
    )
    args = parser.parse_args()

    df = _load_trials(args.input)
    if df is None or df.empty:
        raise SystemExit(1)

    if args.mode in ("module", "both"):
        analyze_module_effect(df, source=args.input, reward_col=str(args.reward_col))
    if args.mode in ("hparam", "both"):
        analyze_hparam_importance(df, source=args.input)


if __name__ == "__main__":
    main()
