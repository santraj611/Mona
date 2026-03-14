# Mona - Local Personal Assistant
# Copyright (C) 2025
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


# llm.py – Model loading and streaming inference.
#
# Qwen2.5 uses the ChatML prompt format:
#   <|im_start|>system\n…<|im_end|>
#   <|im_start|>user\n…<|im_end|>
#   <|im_start|>assistant\n

from llama_cpp import Llama
from config import (
    MODEL_PATH, N_CTX, N_THREADS, N_BATCH,
    MAX_TOKENS, TEMPERATURE, TOP_P, SYSTEM_PROMPT,
)

# Set to True temporarily if you want to see what the router actually outputs
_ROUTER_DEBUG = False
 
 
def load_model() -> Llama:
    """Load the GGUF model. Called once at startup."""
    print("⚙️  Loading model…", end=" ", flush=True)
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        verbose=False,
    )
    print("done.")
    return llm

 

def _build_prompt(history: list[dict], context: str = None) -> str:
    """
    Converts conversation history into Qwen2.5 ChatML format.
 
    Context injection strategy: inject into the LAST user message, not the
    system prompt. Small models (3B) pay far more attention to the most recent
    user turn than to a long system block -- putting context there ensures it
    is not silently ignored.
 
    history = [{"role": "user"|"assistant", "content": str}, ...]
    """
    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
 
    for i, turn in enumerate(history):
        role    = turn["role"]
        content = turn["content"]
 
        # Inject context directly into the last user message only
        is_last_user = (role == "user" and i == len(history) - 1)
        if is_last_user and context:
            content = (
                f"Use the following information to answer my question. "
                f"Cite sources by number e.g. [1] where relevant.\n\n"
                f"{context}\n\n"
                f"My question: {content}"
            )
 
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
 
    # Leave assistant turn open so the model continues from here
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)
 
 
def stream_response(llm: Llama, history: list[dict], context: str = None) -> str:
    """
    Streams the model's reply token by token.
    Returns the full response string when done.
    """
    prompt   = _build_prompt(history, context)
    response = []
 
    for chunk in llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stream=True,
        stop=["<|im_end|>", "<|im_start|>"],
    ):
        token = chunk["choices"][0]["text"]
        print(token, end="", flush=True)
        response.append(token)
 
    print()  # newline after streamed reply
    return "".join(response).strip()
 
