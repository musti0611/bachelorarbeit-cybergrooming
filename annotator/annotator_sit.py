"""GLM 5.2 annotator via Fraunhofer SIT (automat.sit.fraunhofer.de, OpenAI-compatible)."""

import os
import time
from openai import OpenAI
from parse_goldstandard import LABELS

MODEL = "MSF.PhalaCloud/GLM-5.2-W4AFP8"  # SSE twin was removed; MSF = same GLM 5.2, other server
BASE_URL = "https://automat.sit.fraunhofer.de/api"
# Disable GLM's reasoning trace -> ~4.5x faster per call, label stays correct.
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=BASE_URL,
            api_key=os.environ["SIT_API_KEY"],
            timeout=60.0,     # calls are ~1-3s; 60s tolerates load spikes without masking real hangs
            max_retries=3,    # auto-retry transient throttling / timeouts
        )
    return _client


def resolve_model(prefer: str = "GLM") -> str:
    """Pick a currently-available model id (the SIT model list changes often).

    Prefers one whose id contains `prefer` (default GLM); otherwise the first
    available model. Sets and returns the module-global MODEL.
    """
    global MODEL
    ids = [m.id for m in _get_client().models.list().data]
    chosen = next((i for i in ids if prefer.lower() in i.lower()), None) or (ids[0] if ids else MODEL)
    MODEL = chosen
    return chosen


def _normalize(response: str) -> str:
    """Extract canonical label from model response."""
    text = (response or "").strip().upper()
    if text in LABELS:
        return text
    for label in LABELS:
        if label in text:
            return label
    return "UNKNOWN"


def annotate(text: str, system_prompt: str, sleep: float = 0.0) -> str:
    """Call GLM 5.2 (SIT) and return the predicted label."""
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Text: "{text}"\nLabel:'},
        ],
        temperature=0,
        max_tokens=32,  # thinking disabled -> only the short label is generated
        extra_body=EXTRA_BODY,
    )
    if sleep:
        time.sleep(sleep)
    raw = response.choices[0].message.content
    return _normalize(raw)
