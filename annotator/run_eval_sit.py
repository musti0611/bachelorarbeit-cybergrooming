"""
Evaluate GLM 5.2 (Fraunhofer SIT) annotator against the manual gold standard.
- First 3 examples per label -> few-shot in the prompt
- Remaining examples per label -> evaluation set

Requests are sent concurrently (WORKERS threads) for speed. Requires the
environment variable SIT_API_KEY and an active VPN connection to reach
automat.sit.fraunhofer.de.
"""

import json
from pathlib import Path
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from parse_goldstandard import parse, LABELS
from prompt_builder import build_system_prompt
from annotator_sit import annotate, MODEL
from evaluate import compute_metrics, print_report

GOLD_PATH = r"C:\Users\mugur\bachelorarbeit\annotate_results_manual.txt"
RESULTS_DIR = Path(r"C:\Users\mugur\bachelorarbeit\annotator\results")
N_FEWSHOT = 3
PROMPT_VERSION = "v1"
WORKERS = 4


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
    print(f"\nRunning {MODEL} on {total} examples ({WORKERS} parallel workers)...\n")

    errors = {"n": 0}

    def worker(item):
        try:
            return annotate(item["text"], system_prompt)
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            errors["n"] += 1
            return "UNKNOWN"

    preds = [None] * total
    done = 0
    t0 = datetime.now()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, item): i for i, item in enumerate(eval_items)}
        for fut in as_completed(futures):
            i = futures[fut]
            preds[i] = fut.result()
            done += 1
            # Live counter on a single, overwriting line
            print(f"\r  {done}/{total} annotiert...", end="", flush=True)
    print()  # newline after the counter
    elapsed = (datetime.now() - t0).total_seconds()

    results = []
    for item, pred in zip(eval_items, preds):
        correct = item["gold"] == pred
        results.append({
            "gold": item["gold"],
            "pred": pred,
            "text": item["text"],
            "conv_id": item["conv_id"],
            "correct": correct,
        })
        mark = "OK" if correct else "--"
        print(f"  {mark}  gold={item['gold']:<30} pred={pred}")

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — {MODEL}")
    print(f"{'='*60}\n")
    metrics = compute_metrics(results)
    print_report(metrics)

    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    print(f"\nOverall accuracy: {accuracy:.3f}  ({errors['n']} API errors)")
    print(f"Time: {elapsed:.1f}s for {total} examples ({elapsed/total:.2f}s/example wall-clock)")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = MODEL.replace("-", "_").replace(".", "_").replace("/", "_")
    out_path = RESULTS_DIR / f"eval_{safe_model}_{PROMPT_VERSION}_{ts}.json"
    out_path.write_text(
        json.dumps({
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "timestamp": ts,
            "n_fewshot": N_FEWSHOT,
            "workers": WORKERS,
            "total_evaluated": total,
            "accuracy": round(accuracy, 3),
            "api_errors": errors["n"],
            "elapsed_seconds": round(elapsed, 1),
            "metrics": metrics,
            "predictions": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
