"""
Validate the BATCHED GLM 5.2 annotator on the manual gold standard.

Goal: confirm that batching (many messages per API call) yields the SAME quality
as one-message-per-call. Compare the Macro-F1 here against the single-message
run (~0.848). If they match, batching is safe to use on the full PAN12.

Examples are shuffled before batching so each batch mixes labels -- a strong
test that the model really labels each message independently.
"""

import json
import random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from parse_goldstandard import parse, LABELS
from prompt_builder import build_system_prompt
from annotator_sit_batch import annotate_batch
from annotator_sit import MODEL
from evaluate import compute_metrics, print_report

GOLD_PATH = r"C:\Users\mugur\bachelorarbeit\annotate_results_manual.txt"
RESULTS_DIR = Path(r"C:\Users\mugur\bachelorarbeit\annotator\results")
N_FEWSHOT = 3
BATCH_SIZE = 20      # messages per API call
WORKERS = 4          # parallel API calls
SEED = 42


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def main():
    print("Parsing gold standard...")
    data = parse(GOLD_PATH)
    system_prompt = build_system_prompt(data, n_fewshot=N_FEWSHOT)

    eval_items = []
    for label in LABELS:
        for ex in data[label][N_FEWSHOT:]:
            eval_items.append({"gold": label, "text": ex["text"], "conv_id": ex["conv_id"]})

    random.Random(SEED).shuffle(eval_items)  # mix labels across batches
    batches = list(chunk(eval_items, BATCH_SIZE))
    total = len(eval_items)
    print(f"\n{total} examples -> {len(batches)} batches of up to {BATCH_SIZE} "
          f"({WORKERS} parallel workers)\n")

    def worker(batch):
        try:
            return annotate_batch([it["text"] for it in batch], system_prompt)
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            return ["UNKNOWN"] * len(batch)

    preds_by_batch = [None] * len(batches)
    done = 0
    print(f"Sende {len(batches)} Batches an GLM (erster Batch kann ~30 s dauern)...", flush=True)
    t0 = datetime.now()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            preds_by_batch[futures[fut]] = fut.result()
            done += 1
            secs = (datetime.now() - t0).total_seconds()
            print(f"  Batch {done}/{len(batches)} fertig  ({secs:.0f}s)", flush=True)
    elapsed = (datetime.now() - t0).total_seconds()

    results = []
    length_mismatch = 0
    for batch, preds in zip(batches, preds_by_batch):
        if len(preds) != len(batch):
            length_mismatch += 1
            preds = (preds + ["UNKNOWN"] * len(batch))[:len(batch)]
        for item, pred in zip(batch, preds):
            results.append({
                "gold": item["gold"], "pred": pred,
                "text": item["text"], "conv_id": item["conv_id"],
                "correct": item["gold"] == pred,
            })

    print(f"\n{'='*60}")
    print(f"BATCHED EVALUATION — {MODEL} (batch_size={BATCH_SIZE})")
    print(f"{'='*60}\n")
    metrics = compute_metrics(results)
    print_report(metrics)

    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    unknowns = sum(1 for r in results if r["pred"] == "UNKNOWN")
    print(f"\nOverall accuracy: {accuracy:.3f}")
    print(f"UNKNOWN predictions: {unknowns}   |   batches with wrong length: {length_mismatch}")
    print(f"Time: {elapsed:.1f}s for {total} examples ({elapsed/total:.3f}s/example wall-clock)")
    print(f"-> projected for 2,952,960 PAN12 messages: "
          f"{elapsed/total * 2952960 / 3600:.1f} h at these settings")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_batch_bs{BATCH_SIZE}_{ts}.json"
    out_path.write_text(json.dumps({
        "model": MODEL, "batch_size": BATCH_SIZE, "workers": WORKERS,
        "accuracy": round(accuracy, 3), "unknowns": unknowns,
        "length_mismatch_batches": length_mismatch,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics, "predictions": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
