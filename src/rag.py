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


# rag.py – Document ingestion and retrieval for Mona's knowledge base.
#
# No GPU needed. Uses TF-IDF (sklearn) for retrieval — fast, lightweight,
# works entirely on CPU. Drop files into documents/ and Mona will learn from them.
#
# Supported formats: .txt, .md, .pdf

import os
import math
import glob
import pickle
from pathlib import Path

from config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, MIN_SCORE

# Cache file so we don't re-index on every startup
INDEX_CACHE = ".rag_index.pkl"


# ── Text extraction ────────────────────────────────────────────────────────────

def _read_txt(path: Path) -> str:
    with open(path.absolute(), "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print(f"  ⚠️  Skipping {path.absolute()} — install pypdf: pip install pypdf")
        return ""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    ext = p.suffix.lower()

    if ext in {".txt", ".md"}:
        return _read_txt(p)
    elif ext == ".pdf":
        return _read_pdf(p)
    return ""


# ── Chunking ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks, tagging each with its source file."""
    chunks = []
    start  = 0
    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": os.path.basename(source)})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── TF-IDF index ───────────────────────────────────────────────────────────────

class RAGIndex:
    def __init__(self):
        self.chunks:      list[dict] = []
        self.vectorizer = None
        self.matrix     = None

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, docs_dir: str) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        patterns = ["*.txt", "*.md", "*.pdf"]
        files    = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(docs_dir, pat)))

        if not files:
            print(f"  ℹ️  No documents found in '{docs_dir}/' — RAG disabled.")
            return

        print(f"  📚 Indexing {len(files)} file(s)…")
        self.chunks = []
        for path in files:
            text = _extract_text(path)
            if text.strip():
                new_chunks = _chunk_text(text, path)
                self.chunks.extend(new_chunks)
                print(f"     ✓ {os.path.basename(path)} ({len(new_chunks)} chunks)")

        if not self.chunks:
            return

        corpus          = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix     = self.vectorizer.fit_transform(corpus)
        print(f"  ✅ Index ready — {len(self.chunks)} total chunks.")

    # ── Retrieve ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """Return the top_k most relevant chunks for a query."""
        if self.vectorizer is None or self.matrix is None:
            return []

        import numpy as np
        qvec   = self.vectorizer.transform([query])
        # cosine similarity via dot product (TF-IDF rows are not unit-normed, so we normalise)
        scores = (self.matrix * qvec.T).toarray().flatten()

        # normalise
        norms  = np.sqrt(self.matrix.power(2).sum(axis=1)).A1
        qnorm  = math.sqrt(qvec.power(2).sum())
        if qnorm > 0:
            scores = scores / (norms * qnorm + 1e-9)

        top_indices = scores.argsort()[::-1][:top_k]
        results     = []
        for i in top_indices:
            if scores[i] >= MIN_SCORE:
                results.append({**self.chunks[i], "score": float(scores[i])})
        return results

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self) -> None:
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump({"chunks": self.chunks, "vectorizer": self.vectorizer, "matrix": self.matrix}, f)

    def load(self) -> bool:
        if not os.path.exists(INDEX_CACHE):
            return False
        with open(INDEX_CACHE, "rb") as f:
            data = pickle.load(f)
        self.chunks      = data["chunks"]
        self.vectorizer  = data["vectorizer"]
        self.matrix      = data["matrix"]
        return True


# ── Public helpers ─────────────────────────────────────────────────────────────

def load_rag(docs_dir: str = DOCUMENTS_DIR, force_reindex: bool = False) -> RAGIndex:
    """
    Load or build the RAG index.
    Pass force_reindex=True to rebuild after adding new documents.
    """
    index = RAGIndex()

    if not force_reindex and index.load():
        print(f"  📂 RAG index loaded from cache ({len(index.chunks)} chunks).")
        return index

    os.makedirs(docs_dir, exist_ok=True)
    index.build(docs_dir)
    if index.chunks:
        index.save()
    return index


def build_context_block(results: list[dict]) -> str | None:
    """Format retrieved chunks into a context block for the prompt."""
    if not results:
        return None
    parts = ["Relevant information from your knowledge base:\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] (from {r['source']})\n{r['text']}")
    return "\n\n".join(parts)
