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


# router.py - Fast rule-based search router. Zero LLM calls, instant decision.
#
# Design:
#   1. Manual override prefixes always win ("search:", "find:")
#   2. Conversational patterns are blocked first (prevent false positives)
#   3. Temporal/live-data signals trigger search
#   4. Everything else goes to model knowledge

import re

# -- Patterns that NEVER need a search ----------------------------------------
# Checked first so they can't be overridden by trigger patterns below.

_SKIP_PATTERNS = [
    r"^(hi|hey|hello|yo|sup)\b",
    r"^how are you",
    r"^what('s| is) your (name|purpose|goal|job|role)",
    r"^who are you",
    r"^(can|could|will|would) you\b",
    r"^do you (know|think|believe|like|have|remember)",
    r"^are you\b",
    r"^(tell|show) me (a joke|about yourself|something fun)",
    r"^(thanks|thank you|thx|ty)\b",
    r"^(ok|okay|got it|understood|makes sense|cool|nice|great)\b",
    r"^(yes|no|sure|maybe|perhaps)\b",
    r"^(explain|describe|define|what does .* mean)",
    r"^(how do(es)? .* work)",
    r"^(what is the difference between)",
    r"^(summari[sz]e|recap|review) (this|our|the conversation|what)",
]

# -- Patterns that DO need a search -------------------------------------------
# Ordered from most to least specific to avoid false matches.

_SEARCH_PATTERNS = [
    # Explicit temporal references
    r"\b(today|tonight|this (morning|afternoon|evening|week|month|year))\b",
    r"\b(right now|at the moment|currently|as of (now|today|late|recently))\b",
    r"\b(latest|newest|most recent|just released|just announced|breaking)\b",
    r"\b(yesterday|last (night|week|month|year|monday|tuesday|wednesday|thursday|friday))\b",

    # Live data categories
    r"\b(weather|forecast|temperature|rain|snow|humidity)\b",
    r"\b(stock|share) (price|market|value)\b",
    r"\b(price of|cost of|how much (is|does|do))\b",
    r"\b(exchange rate|currency|bitcoin|crypto|ethereum)\b",
    r"\b(news|headline|update|announcement|event)\b",
    r"\b(score|result|match|game|fixture|standings)\b",
    r"\b(traffic|delay|outage|downtime|status)\b",

    # People / places in present tense (CEO, president, etc.)
    r"\bwho (is|are) (the )?(current |new )?(ceo|president|prime minister|chancellor|mayor|head|leader|director|founder|owner)\b",
    r"\bwho (leads|runs|owns|manages|heads)\b",
    r"\bis .* still\b",
    r"\bdoes .* still\b",

    # Explicit search intent
    r"\b(search|look up|find|google|check)\b.*(for|about|on|if|whether)",
    r"\bwhat('s| is) (happening|going on|new|the situation)\b",
    r"\bhave (they|he|she|it) (released|launched|announced|said|done|published)\b",
]

# Pre-compile for speed
_SKIP_RE   = [re.compile(p, re.IGNORECASE) for p in _SKIP_PATTERNS]
_SEARCH_RE = [re.compile(p, re.IGNORECASE) for p in _SEARCH_PATTERNS]


def should_search(query: str) -> bool:
    """
    Instantly decide if a query needs a web search.
    No LLM call. Runs in microseconds.

    Returns True  → do a web search
    Returns False → answer from model knowledge / RAG
    """
    q = query.strip()

    # 1. Manual override always wins
    if q.lower().startswith(("search:", "find:", "look up:")):
        return True

    # 2. Conversational guard — these never need search
    for pattern in _SKIP_RE:
        if pattern.search(q):
            return False

    # 3. Live/temporal signals
    for pattern in _SEARCH_RE:
        if pattern.search(q):
            return True

    return False


def strip_prefix(query: str) -> str:
    """Remove manual search prefixes before sending to DDG."""
    for prefix in ("search:", "find:", "look up:"):
        if query.lower().startswith(prefix):
            return query[len(prefix):].strip()
    return query
