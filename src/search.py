# search.py – Web search for Mona using DuckDuckGo (no API key needed).
#
# Flow:
#   1. Query DuckDuckGo for top N results
#   2. Fetch and strip each page to plain text
#   3. Return a context block Mona can answer from

import re
import textwrap

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from router import strip_prefix
from config import (
    SEARCH_MAX_RESULTS,
    SEARCH_MAX_CHARS,
    SEARCH_TIMEOUT,
)


# ── Fetching ───────────────────────────────────────────────────────────────────
def _fetch_page(url: str) -> str:
    """Download a page and return cleaned plain text."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MonaBot/1.0)"}
        resp    = requests.get(url, headers=headers, timeout=SEARCH_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:SEARCH_MAX_CHARS]

    except Exception:
        return ""


# ── Main search function ───────────────────────────────────────────────────────

def web_search(query: str) -> list[dict]:
    """
    Search DuckDuckGo and return a list of result dicts:
      {"title": str, "url": str, "snippet": str, "body": str}
    """
    clean_query = strip_prefix(query)
    results     = []

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(clean_query, max_results=SEARCH_MAX_RESULTS))
    except Exception as e:
        print(f"  ⚠️  Search failed: {e}")
        return []

    for hit in hits:
        url     = hit.get("href", "")
        snippet = hit.get("body", "")[:300]
        body    = _fetch_page(url) if url else ""

        results.append({
            "title":   hit.get("title", ""),
            "url":     url,
            "snippet": snippet,
            "body":    body or snippet,   # fallback to snippet if fetch fails
        })

    return results


# ── Context formatting ─────────────────────────────────────────────────────────

def build_search_context(results: list[dict], query: str) -> str | None:
    """Format search results into a context block for the prompt."""
    if not results:
        return None

    parts = [f'Web search results for: "{query}"\n']
    for i, r in enumerate(results, 1):
        title = r["title"]
        url   = r["url"]
        body  = textwrap.shorten(r["body"], width=SEARCH_MAX_CHARS, placeholder="…")
        parts.append(f"[{i}] {title}\nSource: {url}\n{body}")

    parts.append(
        "\nUsing the sources above, answer the user's question accurately and concisely. "
        "Cite sources by number e.g. [1] when relevant."
    )
    return "\n\n".join(parts)
