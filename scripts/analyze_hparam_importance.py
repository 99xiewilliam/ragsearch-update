#!/usr/bin/env python3
import json
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def analyze_importance(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return

    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if "_type" in obj: continue
            # Flatten config and reward
            row = dict(obj.get("config", {}))
            row["_reward"] = obj.get("reward", 0.0)
            data.append(row)

    if not data:
        print("No trial data found in file.")
        return

    df = pd.DataFrame(data)
    
    # Identify target and features
    target = "_reward"
    features = [c for c in df.columns if c != target and not c.startswith("_")]

    # Preprocessing
    X = df[features].copy()
    y = df[target]

    le_map = {}
    for col in X.columns:
        if X[col].dtype == 'object' or isinstance(X[col].iloc[0], bool):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_map[col] = le

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get Importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n===== Hyperparameter Importance Analysis =====")
    print(f"Source: {jsonl_path}")
    print(f"Total Trials: {len(df)}")
    print("-" * 45)
    print(f"{'Parameter':<30} | {'Importance':<10}")
    print("-" * 45)
    for i in range(len(features)):
        print(f"{features[indices[i]]:<30} | {importances[indices[i]]:.4f}")
    print("-" * 45)
    print("Interpretation: Higher value means the parameter has a stronger influence on the final reward.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to train_trials_*.jsonl")
    args = parser.parse_args()
    analyze_importance(args.input)
