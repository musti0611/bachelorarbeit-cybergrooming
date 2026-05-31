"""Parse annotate_results_manual.txt into a structured dict."""

import re
from pathlib import Path
from typing import Dict, List

LABELS = [
    "NEUTRAL",
    "VERTRAUENSAUFBAU",
    "INFORMATIONSGEWINNUNG",
    "GEHEIMHALTUNG/ISOLATION",
    "SEXUALISIERUNG",
    "KONTROLLE/NOETIGUNG",
    "OFFLINE-ESKALATION",
]


def parse(filepath: str) -> Dict[str, List[dict]]:
    """
    Returns {label: [{"conv_id": str, "text": str}, ...]}
    First 3 examples per label = few-shot; rest = evaluation set.
    """
    content = Path(filepath).read_text(encoding="utf-8")
    data = {label: [] for label in LABELS}
    current_label = None
    current_entry = None

    for line in content.splitlines():
        # Detect label header: "  LABEL: NEUTRAL  (50 Beispiele)"
        m = re.match(r'\s+LABEL:\s+(\S+.*?)\s+\(\d+', line)
        if m:
            raw = m.group(1).strip()
            label = raw.replace("NÖTIGUNG", "NOETIGUNG")
            if label in data:
                current_label = label
            current_entry = None
            continue

        if current_label is None:
            continue

        # New example: "  [01] Conv-ID: abc123"
        m = re.match(r'\s+\[\d+\]\s+Conv-ID:\s+(\w+)', line)
        if m:
            if current_entry and current_entry.get("text"):
                data[current_label].append(current_entry)
            current_entry = {"conv_id": m.group(1), "text": ""}
            continue

        if current_entry is None:
            continue

        # Skip Turn-Nr line
        if re.match(r'\s+Turn-Nr:', line):
            continue

        # Text line: "       Text:    some text here"
        m = re.match(r'\s+Text:\s+(.*)', line)
        if m:
            current_entry["text"] = m.group(1).strip()
            continue

        # Text continuation (wrapped lines)
        stripped = line.strip()
        if (current_entry["text"] and stripped
                and not re.match(r'={5,}', stripped)
                and not re.match(r'\[\d+\]', stripped)):
            current_entry["text"] += " " + stripped

    # Flush last entry
    if current_label and current_entry and current_entry.get("text"):
        data[current_label].append(current_entry)

    return data
