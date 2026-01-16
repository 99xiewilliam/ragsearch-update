#!/usr/bin/env python3
"""Extract best TPE configuration from results.json.

This script supports both legacy and current results.json schemas.
"""
import json
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python extract_best_config.py <results.json>")
    sys.exit(1)

results_file = sys.argv[1]
if not os.path.exists(results_file):
    print(f"Error: {results_file} not found")
    sys.exit(1)

with open(results_file, 'r') as f:
    data = json.load(f)

# Find TPE results
tpe_data = None
for algo, result in data.get("results", {}).items():
    if algo == "tpe":
        tpe_data = result
        break

if not tpe_data:
    print("Error: No TPE results found in results.json")
    sys.exit(1)

# Schema v2 (current): { best_train: {reward, metrics, config}, validation: {reward, metrics, config} }
best_train = tpe_data.get("best_train") or {}
validation = tpe_data.get("validation") or {}

# Schema v1 (legacy): { best_config, train_best_reward, train_best_metrics, val_reward, val_metrics }
best_config = best_train.get("config") or tpe_data.get("best_config", {})
best_reward = float(best_train.get("reward", tpe_data.get("train_best_reward", 0.0)) or 0.0)
best_metrics = best_train.get("metrics") or tpe_data.get("train_best_metrics", {}) or {}

val_reward = float(validation.get("reward", tpe_data.get("val_reward", 0.0)) or 0.0)
val_metrics = validation.get("metrics") or tpe_data.get("val_metrics", {}) or {}

print("\n" + "=" * 80)
print("TPE Best Configuration (Converged)")
print("=" * 80)
print(f"\n📊 Training Performance:")
print(f"  Best Reward: {best_reward:.6f}")
# Keep output concise: only show commonly reported NLP metrics by default.
# (Raw metrics are still stored in results.json.)
METRIC_ORDER = ["meteor", "rougeL", "em", "bleu", "qa_f1"]
for k in METRIC_ORDER:
    if k in best_metrics:
        print(f"  {k}: {float(best_metrics[k]):.6f}")

print(f"\n📊 Validation Performance:")
print(f"  Reward: {val_reward:.6f}")
for k in METRIC_ORDER:
    if k in val_metrics:
        print(f"  {k}: {float(val_metrics[k]):.6f}")

print(f"\n⚙️  Best Configuration:")
print(json.dumps(best_config, indent=2, ensure_ascii=False))

# Save best config to a separate JSON file
out_dir = os.path.dirname(results_file)
best_config_file = os.path.join(out_dir, "best_config.json")
with open(best_config_file, 'w') as f:
    json.dump(best_config, f, indent=2, ensure_ascii=False)
print(f"\n💾 Best configuration saved to: {best_config_file}")

print("=" * 80)
