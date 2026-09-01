"""Measure the REAL per-call latency of batched annotation, single-threaded.

Uses its own client with a long timeout and NO retries, so the measured time is
the true latency (not masked by retry loops). Tells us how long one batch really
takes and whether the output format is correct, so we can pick a sane batch size
and worker count.
"""

import os
import time
from openai import OpenAI
from parse_goldstandard import parse, LABELS
from prompt_builder import build_system_prompt
from annotator_sit import MODEL, EXTRA_BODY
from annotator_sit_batch import _build_user_prompt, _parse_batch

GOLD_PATH = r"C:\Users\mugur\bachelorarbeit\annotate_results_manual.txt"

client = OpenAI(
    base_url="https://automat.sit.fraunhofer.de/api",
    api_key=os.environ["SIT_API_KEY"],
    timeout=300.0,
    max_retries=0,
)

data = parse(GOLD_PATH)
system_prompt = build_system_prompt(data, n_fewshot=3)
print(f"System prompt: {len(system_prompt)} chars (~{len(system_prompt)//4} tokens)\n")

texts = []
for label in LABELS:
    for ex in data[label][3:]:
        texts.append(ex["text"])


def timed_batch(batch):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_prompt(batch)},
        ],
        temperature=0,
        max_tokens=16 * len(batch) + 100,
        extra_body=EXTRA_BODY,
    )
    dt = time.time() - t0
    labels = _parse_batch(resp.choices[0].message.content, len(batch))
    return dt, labels


for size in [1, 5, 10, 20]:
    try:
        dt, labels = timed_batch(texts[:size])
        ok = sum(1 for l in labels if l != "UNKNOWN")
        print(f"batch_size={size:2d}: {dt:6.1f}s  ->  {dt/size:5.2f}s/msg  "
              f"| {ok}/{size} parsed")
    except Exception as e:
        print(f"batch_size={size:2d}: ERROR: {e}")
