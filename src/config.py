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
MODEL_PATH = os.path.expanduser("~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf")

# ── Runtime ────────────────────────────────────────────────────────────────────
N_CTX      = 2048   # context window (lower = faster, less memory)
N_THREADS  = 4      # match your physical CPU core count
N_BATCH    = 512    # tokens processed per batch

# ── Generation ─────────────────────────────────────────────────────────────────
MAX_TOKENS  = 256   # max tokens per reply (keep low for a 3B model)
TEMPERATURE = 0.7
TOP_P       = 0.9

# ── Persona ────────────────────────────────────────────────────────────────────
ASSISTANT_NAME = "Mona"
SYSTEM_PROMPT  = (
    "You are a friendly assistant named Mona. "
    "You answer as concisely as possible. "
    "You are helpful, kind, clever, and very friendly."
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
 
