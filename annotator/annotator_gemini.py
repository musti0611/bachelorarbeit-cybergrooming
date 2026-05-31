"""Gemini 2.0 Flash annotator (google-genai SDK)."""

import os
import time
from google import genai
from google.genai import types
from parse_goldstandard import LABELS

MODEL = "gemini-2.0-flash"
# Free tier: 15 RPM → 4s between requests to stay safe
SLEEP_BETWEEN_REQUESTS = 4.0
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _normalize(response: str) -> str:
    text = response.strip().upper()
    if text in LABELS:
        return text
    for label in LABELS:
        if label in text:
            return label
    return "UNKNOWN"


def annotate(text: str, system_prompt: str, retries: int = 3) -> str:
    """Call Gemini 2.0 Flash and return the predicted label."""
    prompt = f"{system_prompt}\n\nText: \"{text}\"\nLabel:"
    for attempt in range(retries):
        try:
            response = _get_client().models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=20,
                ),
            )
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            return _normalize(response.text)
        except Exception as e:
            wait = SLEEP_BETWEEN_REQUESTS * (2 ** attempt)
            print(f"  [RETRY {attempt+1}/{retries}] {e} — waiting {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"Gemini API failed after {retries} retries")
