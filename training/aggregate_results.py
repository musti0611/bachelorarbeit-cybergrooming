"""
Aggregate multi-seed results: reads every models/*/summary.json and reports
mean +/- std of the best F1/Precision/Recall per condition (baseline vs
with_labels). This is the core comparison for the thesis.

    py aggregate_results.py
"""

import json
import glob
import statistics
from collections import defaultdict

groups = defaultdict(list)
for path in glob.glob("models/*/summary.json"):
    s = json.load(open(path, encoding="utf-8"))
    if s.get("best"):
        groups[(s["dataset"], s["variant"])].append(s)

if not groups:
    raise SystemExit("Keine summary.json gefunden. Erst train.py laufen lassen.")


def ms(values):
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{m:.4f} +/- {sd:.4f}"


print(f"\n{'Bedingung':<28} {'Seeds':>5}  {'F1':>18} {'Precision':>18} {'Recall':>18}")
print("-" * 92)
for (dataset, variant), runs in sorted(groups.items()):
    seeds = sorted(r["seed"] for r in runs)
    f1 = [r["best"]["f1"] for r in runs]
    pr = [r["best"]["precision"] for r in runs]
    rc = [r["best"]["recall"] for r in runs]
    name = f"{variant} ({dataset})"
    print(f"{name:<28} {len(runs):>5}  {ms(f1):>18} {ms(pr):>18} {ms(rc):>18}")
    print(f"{'  seeds: ' + str(seeds):<28}")
print("-" * 92)
print("Erwartung: with_labels sollte hoeher liegen, wenn die Verhaltens-Labels helfen.")
