"""Verify that the gold-standard examples actually come from the PAN12 corpus.

For each gold example (label, conv_id, text) we check:
  - does its conversation id exist in PAN12?
  - does its text appear as a message body in that conversation?
  - (fallback) does its text appear anywhere in PAN12?
"""

import json
from pathlib import Path
from parse_goldstandard import parse, LABELS

BASE = Path(r"C:\Users\mugur\bachelorarbeit")
GOLD = BASE / "annotate_results_manual.txt"
DPS = {
    "train": BASE / "eSPD-datasets/PAN12/datapacks/datapack-PAN12-train.json",
    "test":  BASE / "eSPD-datasets/PAN12/datapacks/datapack-PAN12-test.json",
}


def norm(s):
    return " ".join((s or "").split()).lower()


# --- gold examples ---
data = parse(str(GOLD))
gold = []
for label in LABELS:
    for ex in data[label]:
        gold.append({"label": label, "conv_id": ex["conv_id"], "text": ex["text"]})
print(f"Gold examples: {len(gold)}\n")

# --- build lookup from PAN12 ---
conv_bodies = {}   # conv_id -> set of normalized bodies
all_bodies = set()
for name, path in DPS.items():
    print(f"Loading {name} datapack...")
    dp = json.loads(path.read_text(encoding="utf-8"))
    for cid, chat in dp["chats"].items():
        s = conv_bodies.setdefault(cid, set())
        for entry in chat.get("content", []):
            if isinstance(entry, dict) and entry.get("type") == "message":
                b = norm(entry.get("body"))
                if b:
                    s.add(b)
                    all_bodies.add(b)
    del dp

# --- check ---
conv_ok = txt_in_conv = txt_anywhere = 0
misses = []
for g in gold:
    nt = norm(g["text"])
    has_conv = g["conv_id"] in conv_bodies
    in_conv = has_conv and nt in conv_bodies[g["conv_id"]]
    anywhere = nt in all_bodies
    conv_ok += has_conv
    txt_in_conv += in_conv
    txt_anywhere += anywhere
    if not in_conv:
        misses.append(g)

n = len(gold)
print(f"\n=== RESULT ({n} gold examples) ===")
print(f"conv_id exists in PAN12:            {conv_ok}/{n}")
print(f"text found in ITS conversation:     {txt_in_conv}/{n}")
print(f"text found ANYWHERE in PAN12:        {txt_anywhere}/{n}")

if misses:
    print(f"\n{len(misses)} examples whose text was NOT found in their conversation (first 15):")
    for g in misses[:15]:
        print(f"  [{g['label']}] conv={g['conv_id']} exists={g['conv_id'] in conv_bodies}")
        print(f"     text: {g['text'][:90]}")
