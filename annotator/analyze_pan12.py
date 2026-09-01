"""Count total vs. unique messages in the PAN12 datapacks.

Decides feasibility of full annotation: unique count = number of LLM calls we
actually need if we deduplicate identical message bodies.
"""

import re
from pathlib import Path

DATAPACKS = [
    r"C:\Users\mugur\bachelorarbeit\eSPD-datasets\PAN12\datapacks\datapack-PAN12-train.json",
    r"C:\Users\mugur\bachelorarbeit\eSPD-datasets\PAN12\datapacks\datapack-PAN12-test.json",
]

# matches "body": "...."  handling \" and \\ escapes
BODY_RE = re.compile(r'"body":\s*"((?:[^"\\]|\\.)*)"')

total = 0
exact = set()
norm = set()
short = 0  # <= 15 chars, the trivial repeated stuff

for path in DATAPACKS:
    text = Path(path).read_text(encoding="utf-8")
    n = 0
    for m in BODY_RE.finditer(text):
        body = m.group(1)
        n += 1
        exact.add(body)
        norm.add(body.strip().lower())
        if len(body) <= 15:
            short += 1
    print(f"{Path(path).name}: {n:,} messages")
    total += n
    del text

print("\n=== TOTALS (train + test) ===")
print(f"Total messages:          {total:,}")
print(f"Unique (exact):          {len(exact):,}  ({len(exact)/total:.1%} of total)")
print(f"Unique (lower+stripped): {len(norm):,}  ({len(norm)/total:.1%} of total)")
print(f"Very short (<=15 chars): {short:,}  ({short/total:.1%} of total)")
