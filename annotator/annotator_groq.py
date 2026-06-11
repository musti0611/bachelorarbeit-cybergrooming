"""Llama 3.3 70B annotator via Together AI."""

import os
import time
from openai import OpenAI
from parse_goldstandard import LABELS

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
        )
    return _client


def _normalize(response: str) -> str:
    text = response.strip().upper()
    if text in LABELS:
        return text
    for label in LABELS:
        if label in text:
            return label
    return "UNKNOWN"


def annotate(text: str, system_prompt: str, sleep: float = 4.0) -> str:
    """Call Llama 3.3 70B via Groq and return the predicted label."""
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
