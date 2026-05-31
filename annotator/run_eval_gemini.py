"""Evaluate Gemini 2.0 Flash annotator against the manual gold standard."""

import json
import sys
from pathlib import Path
from datetime import datetime

from parse_goldstandard import parse, LABELS
from prompt_builder import build_system_prompt
from annotator_gemini import annotate, MODEL
from evaluate import compute_metrics, print_report

GOLD_PATH = r"C:\Users\mugur\bachelorarbeit\annotate_results_manual.txt"
RESULTS_DIR = Path(r"C:\Users\mugur\bachelorarbeit\annotator\results")
N_FEWSHOT = 3
PROMPT_VERSION = "v2"


def main():
    print("Parsing gold standard...")
    data = parse(GOLD_PATH)
    for label in LABELS:
        print(f"  {label}: {len(data[label])} examples ({N_FEWSHOT} few-shot, {len(data[label]) - N_FEWSHOT} eval)")

    print("\nBuilding prompt...")
    system_prompt = build_system_prompt(data, n_fewshot=N_FEWSHOT)

    eval_items = []
    for label in LABELS:
        for ex in data[label][N_FEWSHOT:]:
            eval_items.append({"gold": label, "text": ex["text"], "conv_id": ex["conv_id"]})

    total = len(eval_items)
    print(f"\nRunning {MODEL} on {total} examples...\n")

    results = []
    errors = 0
    for i, item in enumerate(eval_items):
        try:
            pred = annotate(item["text"], system_prompt)
        except Exception as e:
            print(f"  [ERROR] {e}")
            pred = "UNKNOWN"
            errors += 1

        correct = item["gold"] == pred
        results.append({
            "gold": item["gold"],
            "pred": pred,
            "text": item["text"],
            "conv_id": item["conv_id"],
            "correct": correct,
        })

        mark = "OK" if correct else "--"
        print(f"  [{i+1:3d}/{total}] {mark}  gold={item['gold']:<30} pred={pred}")

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS -- {MODEL}")
    print(f"{'='*60}\n")
    metrics = compute_metrics(results)
    print_report(metrics)

    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    print(f"\nOverall accuracy: {accuracy:.3f}  ({errors} API errors)")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{MODEL.replace('-', '_')}_{PROMPT_VERSION}_{ts}.json"
    out_path.write_text(
        json.dumps({
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "timestamp": ts,
            "n_fewshot": N_FEWSHOT,
            "total_evaluated": total,
            "accuracy": round(accuracy, 3),
            "api_errors": errors,
            "metrics": metrics,
            "predictions": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
