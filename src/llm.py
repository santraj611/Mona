# Mona - Local Personal Assistant
# Copyright (C) 2026
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

import os
import inspect
from typing import TypedDict

from llama_cpp import Llama
from config import (
    MODEL_PATH, N_CTX, N_THREADS, N_BATCH,
    MAX_TOKENS, TEMPERATURE, TOP_P, SYSTEM_PROMPT, PROMPT_FORMAT,
    LLAMA_FLASH_ATTN, FALLBACK_MODEL_PATHS,
)

# Set to True temporarily if you want to see what the router actually outputs
_ROUTER_DEBUG = False
class Turn(TypedDict):
    role: str
    content: str


def _resolve_prompt_format() -> str:
    """
    Resolve prompt format from config.
    Supported formats: chatml, gemma, llama3, mistral.
    """
    fmt = PROMPT_FORMAT.strip().lower()
    if fmt and fmt != "auto":
        if fmt in {"chatml", "gemma", "llama3", "mistral"}:
            return fmt
        print(f"⚠️  Unknown PROMPT_FORMAT='{PROMPT_FORMAT}', falling back to chatml.")
        return "chatml"

    model_path = MODEL_PATH.lower()
    if "gemma" in model_path:
        return "gemma"
    if "llama-3" in model_path or "llama3" in model_path or "llama_3" in model_path:
        return "llama3"
    if "mistral" in model_path:
        return "mistral"
    if "qwen" in model_path:
        return "chatml"
    return "chatml"


def _get_stop_tokens(prompt_format: str) -> list[str]:
    if prompt_format == "gemma":
        return ["<end_of_turn>", "<start_of_turn>"]
    if prompt_format == "llama3":
        return ["<|eot_id|>", "<|start_header_id|>"]
    if prompt_format == "mistral":
        return ["</s>", "[INST]"]
    return ["<|im_end|>", "<|im_start|>"]


def _build_prompt_chatml(history: list[Turn], context: str | None = None) -> str:
    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]

    for i, turn in enumerate(history):
        role = turn["role"]
        content = turn["content"]

        is_last_user = (role == "user" and i == len(history) - 1)
        if is_last_user and context:
            content = (
                "Use the following information to answer my question. "
                "Cite sources by number e.g. [1] where relevant.\n\n"
                f"{context}\n\n"
                f"My question: {content}"
            )

        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _build_prompt_gemma(history: list[Turn], context: str | None = None) -> str:
    """
    Build prompt for Gemma instruct/chat models.
    Gemma uses: <start_of_turn>user/model ... <end_of_turn>
    """
    parts: list[str] = []

    # Gemma generally does not use a dedicated system role in the template.
    # We fold system instructions into the first user turn for better adherence.
    first_user_idx = next((i for i, t in enumerate(history) if t["role"] == "user"), None)

    for i, turn in enumerate(history):
        role = turn["role"]
        content = turn["content"]

        if role == "assistant":
            gemma_role = "model"
        elif role == "user":
            gemma_role = "user"
        else:
            continue

        if i == first_user_idx:
            content = f"{SYSTEM_PROMPT}\n\n{content}"

        is_last_user = (role == "user" and i == len(history) - 1)
        if is_last_user and context:
            content = (
                "Use the following information to answer my question. "
                "Cite sources by number e.g. [1] where relevant.\n\n"
                f"{context}\n\n"
                f"My question: {content}"
            )

        parts.append(f"<start_of_turn>{gemma_role}\n{content}<end_of_turn>\n")

    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def _build_prompt_llama3(history: list[Turn], context: str | None = None) -> str:
    parts = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + f"{SYSTEM_PROMPT}<|eot_id|>"
    ]

    for i, turn in enumerate(history):
        role = turn["role"]
        content = turn["content"]
        if role not in {"user", "assistant"}:
            continue

        is_last_user = (role == "user" and i == len(history) - 1)
        if is_last_user and context:
            content = (
                "Use the following information to answer my question. "
                "Cite sources by number e.g. [1] where relevant.\n\n"
                f"{context}\n\n"
                f"My question: {content}"
            )

        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
        )

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "".join(parts)


def _build_prompt_mistral(history: list[Turn], context: str | None = None) -> str:
    # Mistral Instruct style: <s>[INST] user [/INST] assistant</s>...
    system_prefix = f"<<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
    turns: list[str] = []
    i = 0
    while i < len(history):
        turn = history[i]
        if turn["role"] != "user":
            i += 1
            continue

        user_content = turn["content"]
        if i == 0:
            user_content = system_prefix + user_content

        is_last_user = (i == len(history) - 1)
        if is_last_user and context:
            user_content = (
                "Use the following information to answer my question. "
                "Cite sources by number e.g. [1] where relevant.\n\n"
                f"{context}\n\n"
                f"My question: {user_content}"
            )

        assistant_content = ""
        if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
            assistant_content = history[i + 1]["content"]
            i += 1

        turns.append(f"<s>[INST] {user_content} [/INST]{assistant_content}</s>")
        i += 1

    # Open assistant generation turn.
    if not turns:
        turns.append(f"<s>[INST] {system_prefix}Hello [/INST]")
    elif history and history[-1]["role"] == "user":
        turns[-1] = turns[-1].removesuffix("</s>")
    else:
        turns.append("<s>[INST] Continue. [/INST]")

    return "".join(turns)

 
def load_model() -> Llama:
    """Load the GGUF model. Called once at startup."""
    model_path = MODEL_PATH
    print("⚙️  Loading model…", end=" ", flush=True)

    if not os.path.exists(model_path):
        print("failed.")
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}\n"
            + "Set MODEL_PATH in config.py to an existing .gguf file."
        )

    def _make_llama(path: str) -> Llama:
        # flash_attn exists only on newer llama-cpp-python builds.
        if "flash_attn" in inspect.signature(Llama).parameters:
            return Llama(
                model_path=path,
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_batch=N_BATCH,
                flash_attn=LLAMA_FLASH_ATTN,
                verbose=False,
            )
        return Llama(
            model_path=path,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=N_BATCH,
            verbose=False,
        )

    try:
        llm = _make_llama(model_path)
        print("done.")
        print(f"🧠 Loaded model: {model_path} (primary)")
        return llm
    except Exception as e:
        # Some GGUFs may not be supported by the local llama-cpp runtime.
        # Try configured fallback models in order.
        for fallback_path in FALLBACK_MODEL_PATHS:
            if model_path == fallback_path or not os.path.exists(fallback_path):
                continue
            print("failed.")
            print(
                "⚠️  Primary model failed to load. "
                + f"Trying fallback model: {fallback_path}"
            )
            print(f"   Loader error: {e}")
            try:
                llm = _make_llama(fallback_path)
                print("✅ Fallback model loaded.")
                print(f"🧠 Loaded model: {fallback_path} (fallback)")
                return llm
            except Exception as fallback_error:
                print(f"   Fallback also failed: {fallback_error}")

        raise RuntimeError(
            "Failed to load GGUF model. "
            + "If this is a Gemma model, update llama-cpp-python to a newer version "
            + "that supports this architecture, or use a known-compatible GGUF build."
        ) from e

 

def _build_prompt(history: list[Turn], context: str | None = None) -> str:
    """
    Converts conversation history into the selected chat prompt format.
 
    Context injection strategy: inject into the LAST user message, not the
    system prompt. Small models (3B) pay far more attention to the most recent
    user turn than to a long system block -- putting context there ensures it
    is not silently ignored.
 
    history = [{"role": "user"|"assistant", "content": str}, ...]
    """
    prompt_format = _resolve_prompt_format()
    if prompt_format == "gemma":
        return _build_prompt_gemma(history, context)
    if prompt_format == "llama3":
        return _build_prompt_llama3(history, context)
    if prompt_format == "mistral":
        return _build_prompt_mistral(history, context)
    return _build_prompt_chatml(history, context)
 
 
def stream_response(llm: Llama, history: list[Turn], context: str | None = None) -> str:
    """
    Streams the model's reply token by token.
    Returns the full response string when done.
    """
    prompt_format = _resolve_prompt_format()
    prompt = _build_prompt(history, context)
    response: list[str] = []
 
    for chunk in llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stream=True,
        stop=_get_stop_tokens(prompt_format),
    ):
        if isinstance(chunk, str):
            continue
        if not chunk["choices"]:
            continue
        token = chunk["choices"][0]["text"]
        print(token, end="", flush=True)
        response.append(token)
 
    print()  # newline after streamed reply
    return "".join(response).strip()
 
