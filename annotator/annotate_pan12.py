"""
Annotate the full PAN12 datapack with behavior labels (GLM 5.2, batched).

Robust for long / overnight runs:
- CHECKPOINTED: every label is written to a .jsonl file immediately. If the run
  is interrupted (VPN drop, server change, Ctrl+C), just start it again -- it
  skips everything already done and resumes.
- AUTO-MODEL: picks whichever GLM is currently available (the SIT list changes).
- BATCHED + PARALLEL: many independent messages per call, several calls at once.

Usage (PowerShell):
    py .\annotate_pan12.py --dataset train
    py .\annotate_pan12.py --dataset test
    py .\annotate_pan12.py --dataset train --limit 2000      # small dry run
    py .\annotate_pan12.py --dataset train --workers 8 --batch-size 40

The labels are written to  pan12_labels/<dataset>.jsonl .
Run build_labeled_datapack.py afterwards to fold them into a datapack copy.
"""

import argparse
import json
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import annotator_sit
from annotator_sit import _get_client
from annotator_sit_batch import annotate_batch

# PINNED: train and test MUST use the exact same model. Do NOT auto-switch.
PINNED_MODEL = "MSF.PhalaCloud/GLM-5.2-W4AFP8"
from parse_goldstandard import parse
from prompt_builder import build_system_prompt

BASE = Path(r"C:\Users\mugur\bachelorarbeit")
DATAPACKS = {
    "train": BASE / "eSPD-datasets/PAN12/datapacks/datapack-PAN12-train.json",
    "test":  BASE / "eSPD-datasets/PAN12/datapacks/datapack-PAN12-test.json",
}
GOLD_PATH = BASE / "annotate_results_manual.txt"
OUT_DIR = BASE / "annotator" / "pan12_labels"


def collect_tasks(datapack):
    """Return [(key, body), ...] for every non-empty message. key = chatID#index."""
    tasks = []
    for chat_id, chat in datapack["chats"].items():
        for i, entry in enumerate(chat.get("content", [])):
            if isinstance(entry, dict) and entry.get("type") == "message":
                body = (entry.get("body") or "").strip()
                if body:
                    tasks.append((f"{chat_id}#{i}", body))
    return tasks


def load_done(ckpt_path):
    done = set()
    if ckpt_path.exists():
        with ckpt_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec["l"] != "UNKNOWN":   # UNKNOWNs are re-tried on the next run
                            done.add(rec["k"])
                    except Exception:
                        pass
    return done


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["train", "test"], default="train")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="only annotate the first N pending messages (dry run)")
    args = ap.parse_args()

    available = [m.id for m in _get_client().models.list().data]
    if PINNED_MODEL not in available:
        raise SystemExit(
            f"\nFEHLER: Das festgelegte Modell '{PINNED_MODEL}' ist gerade NICHT verfuegbar.\n"
            f"Verfuegbar sind: {available}\n"
            f"Abbruch, damit train und test NICHT versehentlich auf verschiedenen Modellen laufen.\n"
            f"Warte, bis das Modell wieder da ist, oder frage bei Fraunhofer nach."
        )
    annotator_sit.MODEL = PINNED_MODEL
    print(f"Model (pinned): {PINNED_MODEL}")

    print("Building prompt...")
    system_prompt = build_system_prompt(parse(str(GOLD_PATH)), n_fewshot=3)

    print(f"Loading datapack '{args.dataset}' (this can take a moment)...")
    datapack = json.loads(DATAPACKS[args.dataset].read_text(encoding="utf-8"))
    tasks = collect_tasks(datapack)
    del datapack  # free memory; we only needed the messages

    OUT_DIR.mkdir(exist_ok=True)
    ckpt_path = OUT_DIR / f"{args.dataset}.jsonl"
    done = load_done(ckpt_path)

    pending = [t for t in tasks if t[0] not in done]
    if args.limit:
        pending = pending[:args.limit]

    print(f"Total messages: {len(tasks):,} | already done: {len(done):,} | "
          f"to do now: {len(pending):,}")
    if not pending:
        print("Nothing to do. All messages already annotated.")
        return

    batches = list(chunk(pending, args.batch_size))
    print(f"-> {len(batches):,} batches of {args.batch_size}, {args.workers} parallel workers\n")

    lock = threading.Lock()
    ckpt_file = ckpt_path.open("a", encoding="utf-8")
    state = {"done": 0, "errors": 0}
    total = len(pending)
    t0 = time.time()

    def worker(batch):
        keys = [k for k, _ in batch]
        try:
            labels = annotate_batch([b for _, b in batch], system_prompt)
        except Exception as e:
            state["errors"] += 1
            labels = ["UNKNOWN"] * len(batch)
            print(f"\n  [ERROR] {e}")
        if len(labels) != len(batch):
            labels = (labels + ["UNKNOWN"] * len(batch))[:len(batch)]
        with lock:
            for k, lab in zip(keys, labels):
                ckpt_file.write(json.dumps({"k": k, "l": lab}) + "\n")
            ckpt_file.flush()
            state["done"] += len(batch)
            d = state["done"]
            el = time.time() - t0
            rate = d / el if el else 0
            eta = (total - d) / rate if rate else 0
            print(f"  {d:,}/{total:,}  ({rate:.0f} msg/s, ETA {eta/3600:.1f} h, "
                  f"{state['errors']} errors)")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, b) for b in batches]
            for _ in as_completed(futures):
                pass
    finally:
        ckpt_file.close()

    el = time.time() - t0
    print(f"\nDone this session: {state['done']:,} messages in {el/60:.1f} min "
          f"({state['errors']} batch errors). Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
