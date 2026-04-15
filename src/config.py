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


# config.py – All settings in one place. Edit here, not in code.

import os

# ── Model ──────────────────────────────────────────────────────────────────────
# Primary model to load first.
MODEL_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-Q4_K_M.gguf")
# Tried only if primary model fails to load.
FALLBACK_MODEL_PATHS = [
    os.path.expanduser("~/models/tinyllama_1B.gguf"),
    os.path.expanduser("~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
]
# Prompt format used to build chat prompts.
# Options: "auto", "chatml", "gemma", "llama3", "mistral"
# - auto: infer from MODEL_PATH (qwen/gemma/llama3/mistral -> matching format)
PROMPT_FORMAT = "auto"

# ── Runtime ────────────────────────────────────────────────────────────────────
# CPU-friendly preset for low-end systems (no GPU/VRAM).
LOW_RESOURCE_MODE = True
# N_CTX: smaller context = less RAM and faster prompt processing.
N_CTX      = 4096 if LOW_RESOURCE_MODE else 32768
# Use a conservative thread count by default to avoid freezing the system.
N_THREADS  = max(1, (os.cpu_count() or 2) * 3 // 4)
# Lower batch size significantly reduces peak memory on CPU.
N_BATCH    = 192 if LOW_RESOURCE_MODE else 512
# Try to enable llama.cpp Flash Attention (reduces KV-cache warnings and memory).
LLAMA_FLASH_ATTN = True

# ── Generation ─────────────────────────────────────────────────────────────────
MAX_TOKENS  = 192 if LOW_RESOURCE_MODE else 1024
TEMPERATURE = 0.7
TOP_P       = 0.9

# ── Persona ────────────────────────────────────────────────────────────────────
ASSISTANT_NAME = "Mona"
SYSTEM_PROMPT = (
    "You are a friendly assistant named Mona. "
    "You are helpful, kind, clever, and very friendly. "
    "For coding tasks, write clean code with brief explanations. "
    "For creative tasks like stories, write multiple paragraphs. "
    "Match your response length to what the question actually needs."
)

# ── Memory ─────────────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10  # how many back-and-forth exchanges to remember

# ── Web Search ─────────────────────────────────────────────────────────────────
SEARCH_MAX_RESULTS  = 5      # how many pages to fetch per search
SEARCH_MAX_CHARS    = 600    # max characters to extract per page
SEARCH_TIMEOUT      = 6      # seconds before giving up on a page fetch
# Keywords that hint the user wants fresh/online info
SEARCH_TRIGGER_WORDS = [
    "latest", "current", "today", "news", "recent", "now",
    "price", "weather", "score", "who is", "what is", "search",
    "look up", "find", "2024", "2025",
]
 
# ── RAG ────────────────────────────────────────────────────────────────────────
DOCUMENTS_DIR     = "documents"   # folder to drop your .txt / .pdf files into
CHUNK_SIZE        = 400           # characters per chunk
CHUNK_OVERLAP     = 80            # overlap between chunks to preserve context
TOP_K_RESULTS     = 3             # how many chunks to inject into the prompt
MIN_SCORE         = 0.15          # minimum similarity score to bother injecting
 
