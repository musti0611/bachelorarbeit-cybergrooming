"""Quick connectivity + behavior check for the SIT GLM endpoint.

Shows the RAW model output and finish_reason so we can tell whether GLM emits
extra 'thinking' text (which would require a higher max_tokens) or answers
cleanly with just the label.
"""

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://automat.sit.fraunhofer.de/api",
    api_key=os.environ["SIT_API_KEY"],
)

resp = client.chat.completions.create(
    model="SSE.PhalaCloud/GLM-5.2-W4AFP8",
    messages=[
        {"role": "system", "content": "Reply with ONLY one word: TEST_OK"},
        {"role": "user", "content": "Say the word."},
    ],
    temperature=0,
    max_tokens=64,
)

msg = resp.choices[0].message
print("RAW content:", repr(msg.content))
print("finish_reason:", resp.choices[0].finish_reason)
# Some servers put reasoning in a separate field:
print("reasoning:", repr(getattr(msg, "reasoning_content", None)))
