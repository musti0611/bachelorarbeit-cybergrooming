"""Llama 3.1 8B annotator via Groq."""

import os
import re
import time
from groq import Groq
from parse_goldstandard import LABELS

MODEL = "llama-3.1-8b-instant"
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _normalize(response: str) -> str:
    text = response.strip().upper()
    if text in LABELS:
        return text
    for label in LABELS:
        if label in text:
            return label
    return "UNKNOWN"


def _parse_retry_delay(error_str: str, default: float = 20.0) -> float:
    """Extract suggested retry delay from Groq rate limit error message."""
    m = re.search(r'Please try again in ([\d.]+)s', error_str)
    return float(m.group(1)) + 1.5 if m else default


def annotate(text: str, system_prompt: str, retries: int = 8) -> str:
    """Call Llama 3.1 8B via Groq and return the predicted label."""
    for attempt in range(retries):
        try:
            response = _get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Text: "{text}"\nLabel:'},
                ],
                temperature=0,
                max_tokens=20,
            )
            return _normalize(response.choices[0].message.content)
        except Exception as e:
            wait = _parse_retry_delay(str(e))
            print(f"  [RATE LIMIT attempt {attempt+1}] waiting {wait:.0f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Llama API failed after {retries} retries")
