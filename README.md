# Mona — Local Personal Assistant

A modular, offline-first personal assistant powered by a local GGUF model via `llama-cpp-python`. Inspired by JARVIS — Mona runs entirely on your machine, searches the web when needed, and can answer from your own documents.

> **Status: Paused** — The current hardware (CPU-only) makes inference too slow for daily use (~10 min/reply on a 3B model). The codebase is complete and ready to resume when better hardware is available.

---

## Project Structure

```
mona/
├── documents/     # Drop .txt / .md / .pdf files here for RAG
├── src/           # Source code
    ├── main.py        # Entry point — CLI chat loop
    ├── config.py      # All settings in one place
    ├── llm.py         # Model loading, prompt building, streaming
    ├── memory.py      # Conversation history (sliding window)
    ├── router.py      # Instant rule-based search/skip classifier
    ├── search.py      # DuckDuckGo web search + page scraping
    ├── rag.py         # Local document ingestion + TF-IDF retrieval
    ├── tests.py       # Full test suite (51 tests)
```

---

## Features

- **Streaming replies** — tokens print as they're generated
- **Conversation memory** — remembers the last N exchanges (configurable)
- **Web search** — automatically searches DuckDuckGo for live/current queries
- **Local RAG** — indexes your own documents and retrieves relevant chunks
- **Smart routing** — rule-based classifier decides search vs model knowledge instantly, no extra LLM call
- **Context injection** — search/RAG results injected into the last user message (not the system prompt) so small models actually use them

---

## Setup

### 1. Install dependencies

```bash
pip install llama-cpp-python scikit-learn duckduckgo-search requests beautifulsoup4 pypdf
```

### 2. Download a model

Place a GGUF model file in `~/models/`. The project is configured for:

```
~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
```

You can change this in `config.py`.

### 3. Run

```bash
python main.py
```

---

## Usage

```
You: what's the latest news on AI?
  🌐 Searching the web...
  ✅ Got 5 result(s)

Mona: [answers from live search results, citing [1][2] etc]

You: explain how transformers work
Mona: [answers from model knowledge, no search needed]

You: search: current bitcoin price
  🌐 Searching the web...
```

**Commands:**
- `clear` — wipe conversation memory
- `reindex` — reload documents from the `documents/` folder
- `search: <query>` — force a web search regardless of routing decision
- `exit` / `quit` — exit

---

## Configuration

All tunable settings live in `config.py`:

| Setting | Default | Notes |
|---|---|---|
| `MODEL_PATH` | `~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf` | Path to your GGUF file |
| `N_CTX` | `2048` | Context window size |
| `N_THREADS` | `4` | Set to your physical CPU core count |
| `N_BATCH` | `512` | Tokens per batch — higher is faster |
| `MAX_TOKENS` | `256` | Max tokens per reply |
| `MAX_HISTORY_TURNS` | `10` | How many exchanges Mona remembers |
| `TOP_K_RESULTS` | `3` | RAG chunks injected per query |

---

## Tests

```bash
# Unit tests (no model or network needed, runs in <1s)
python tests.py

# or with pytest
pytest tests.py -v

# Include live network tests
pytest tests.py -v -k integration
```

**51 tests** covering: prompt builder, conversation memory, search router, context formatting, RAG chunking, and web search integration.

---

## Known Issues / Limitations

- **Slow on CPU** — a 3B model on CPU takes several minutes per reply. A GPU with 8GB+ VRAM would bring this down to seconds. See *Resuming* below.
- **GGUF models can't be fine-tuned** — the quantized format is inference-only. Fine-tuning requires the original fp16 weights and a GPU.
- **Router uses regex rules** — the search/skip classifier covers common cases well but may miss niche queries. Add patterns to `_SEARCH_PATTERNS` / `_SKIP_PATTERNS` in `router.py` as needed.

---

## Resuming on Better Hardware

When you have a GPU available, only two things need to change:

**1. `config.py` — update model path and unlock context/tokens:**
```python
MODEL_PATH = "~/models/your-new-model.gguf"
N_CTX      = 8192
MAX_TOKENS = 1024
N_BATCH    = 512
```

**2. `llm.py` — add GPU layers:**
```python
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_batch=N_BATCH,
    n_gpu_layers=-1,   # -1 = offload all layers to GPU
    verbose=False,
)
```

**Recommended models to try:**
- `Qwen2.5-7B-Instruct-Q4_K_M` — great balance of speed and quality on 8GB VRAM
- `Llama-3.1-8B-Instruct-Q4_K_M` — strong general assistant
- `Mistral-7B-Instruct-v0.3-Q4_K_M` — fast, good for conversation

All use the same ChatML-compatible prompt format already implemented.

---

## Prompt Format

Mona uses Qwen2.5's ChatML format:

```
<|im_start|>system
You are a friendly assistant named Mona...
<|im_end|>
<|im_start|>user
[context block if search/RAG found something]

My question: what is X?
<|im_end|>
<|im_start|>assistant
```

Context is injected into the user message (not the system prompt) because small models pay more attention to the most recent user turn.
