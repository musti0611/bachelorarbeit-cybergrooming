"""Batched GLM 5.2 annotator: label many INDEPENDENT messages per API call.

Each message is labelled on its own surface content only (same methodology as
the single-message annotator) -- messages are explicitly presented as unrelated,
so the model does not use conversation context (avoids intent/leakage).
"""

import re
import annotator_sit
from annotator_sit import _get_client, EXTRA_BODY, _normalize
from parse_goldstandard import LABELS

# matches "12. LABEL" / "12) LABEL" / "12 - LABEL" etc.
_LINE_RE = re.compile(r'^\s*(\d+)\s*[.)\-:]\s*(.+?)\s*$')


def _build_user_prompt(texts) -> str:
    lines = [
        "Classify EACH message below INDEPENDENTLY, based only on its own surface content.",
        "The messages are UNRELATED to each other. Do NOT use any message as context for another.",
        f"Return EXACTLY {len(texts)} lines, one label per message, in this exact format:",
        "<number>. <LABEL>",
        "Output ONLY these numbered lines. No explanation, no extra text.",
        "",
    ]
    for i, t in enumerate(texts, 1):
        one_line = " ".join(str(t).split())  # collapse newlines/whitespace
        if len(one_line) > 4000:             # cap giant pasted logs so a batch can't blow the context limit
            one_line = one_line[:4000]
        lines.append(f'{i}. "{one_line}"')
    return "\n".join(lines)


def _parse_batch(raw: str, n: int) -> list:
    by_idx = {}
    ordered = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if m:
            lab = _normalize(m.group(2))
            by_idx[int(m.group(1))] = lab
            ordered.append(lab)
        else:
            lab = _normalize(line)
            if lab != "UNKNOWN":
                ordered.append(lab)

    # Prefer the explicit numbering; fall back to positional order.
    if len(by_idx) >= 0.8 * n:
        return [by_idx.get(i, "UNKNOWN") for i in range(1, n + 1)]
    return [ordered[i] if i < len(ordered) else "UNKNOWN" for i in range(n)]


def annotate_batch(texts, system_prompt) -> list:
    """Label a list of messages in a single API call. Returns len(texts) labels."""
    if not texts:
        return []
    response = _get_client().chat.completions.create(
        model=annotator_sit.MODEL,  # read live so resolve_model() can switch it
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_prompt(texts)},
        ],
        temperature=0,
        max_tokens=16 * len(texts) + 100,  # ~1 short label line per message
        extra_body=EXTRA_BODY,
    )
    raw = response.choices[0].message.content
    return _parse_batch(raw, len(texts))
