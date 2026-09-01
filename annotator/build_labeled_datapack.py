"""
Fold the annotated labels (pan12_labels/<dataset>.jsonl) into a COPY of the
datapack, writing each message's behavior label into its `labels` field.

The original datapack is never modified. Output:
    datapacks/datapack-PAN12-<dataset>-labeled.json

Usage:  py .\build_labeled_datapack.py --dataset train
"""

import argparse
import json
from pathlib import Path

BASE = Path(r"C:\Users\mugur\bachelorarbeit")
DP_DIR = BASE / "eSPD-datasets/PAN12/datapacks"
OUT_DIR = BASE / "annotator" / "pan12_labels"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["train", "test"], default="train")
    args = ap.parse_args()

    ckpt = OUT_DIR / f"{args.dataset}.jsonl"
    print("Lade Labels aus Checkpoint...")
    labels = {}
    with ckpt.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec["l"] != "UNKNOWN":   # skip stale UNKNOWN lines; real labels are authoritative
                    labels[rec["k"]] = rec["l"]
    print(f"  {len(labels):,} Labels geladen.")

    src = DP_DIR / f"datapack-PAN12-{args.dataset}.json"
    print(f"Lade Original-Datapack ({src.name}, kann ~1 Min dauern)...")
    datapack = json.loads(src.read_text(encoding="utf-8"))
    n_chats = len(datapack["chats"])
    print(f"  {n_chats:,} Konversationen geladen. Fuelle Labels ein...")

    filled, missing = 0, 0
    for ci, (chat_id, chat) in enumerate(datapack["chats"].items(), 1):
        for i, entry in enumerate(chat.get("content", [])):
            if isinstance(entry, dict) and entry.get("type") == "message" and (entry.get("body") or "").strip():
                lab = labels.get(f"{chat_id}#{i}")
                if lab:
                    entry["labels"] = [lab]
                    filled += 1
                else:
                    missing += 1
        if ci % 20000 == 0:
            print(f"  {ci:,}/{n_chats:,} Konversationen verarbeitet...")

    out = DP_DIR / f"datapack-PAN12-{args.dataset}-labeled.json"
    print(f"Schreibe {out.name} (grosse Datei, kann ~1-2 Min dauern)...")
    out.write_text(json.dumps(datapack, ensure_ascii=False), encoding="utf-8")
    print(f"FERTIG: {filled:,} Nachrichten gelabelt ({missing:,} ohne Label). ORIGINAL unveraendert.")
    print(f"Geschrieben: {out}")


if __name__ == "__main__":
    main()
