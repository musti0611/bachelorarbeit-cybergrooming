"""GPT-4o-mini annotator."""

import os
import time
from openai import OpenAI
from parse_goldstandard import LABELS

MODEL = "gpt-4o-mini"
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _normalize(response: str) -> str:
    """Extract canonical label from model response."""
    text = response.strip().upper()
    if text in LABELS:
        return text
    for label in LABELS:
        if label in text:
            return label
    return "UNKNOWN"


def annotate(text: str, system_prompt: str, sleep: float = 0.2) -> str:
    """Call GPT-4o-mini and return the predicted label."""
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Text: "{text}"\nLabel:'},
        ],
        temperature=0,
        max_tokens=20,
    )
    time.sleep(sleep)
    raw = response.choices[0].message.content
    return _normalize(raw)
