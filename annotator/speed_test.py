"""Find the fastest working config for GLM on the SIT endpoint.

Tries several ways to disable the model's 'thinking' trace and times each one,
so we know which config to use for the annotator + the full PAN12 run.
"""

import os
import time
from openai import OpenAI

client = OpenAI(
    base_url="https://automat.sit.fraunhofer.de/api",
    api_key=os.environ["SIT_API_KEY"],
)
MODEL = "SSE.PhalaCloud/GLM-5.2-W4AFP8"

SYS = "Reply with ONLY one word: NEUTRAL"
USER = 'Text: "hey how was school today"\nLabel:'


def try_config(name, **extra):
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": USER}],
            temperature=0,
            max_tokens=512,
            **extra,
        )
        dt = time.time() - t0
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        r_len = len(reasoning) if reasoning else 0
        print(f"[{name:28}] {dt:6.2f}s  content={msg.content!r:20}  reasoning_chars={r_len}")
    except Exception as e:
        print(f"[{name:28}] ERROR: {e}")


print(f"Model: {MODEL}\n")
try_config("baseline (thinking on)")
try_config("chat_template enable_thinking", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
try_config("thinking type disabled", extra_body={"thinking": {"type": "disabled"}})
try_config("reasoning effort none", extra_body={"reasoning": {"enabled": False}})
try_config("chat_template thinking", extra_body={"chat_template_kwargs": {"thinking": False}})
