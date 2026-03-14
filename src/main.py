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

# main.py – Entry point. Runs the CLI chat loop.

from config import ASSISTANT_NAME
from llm import load_model, stream_response
from router import should_search
from memory import ConversationMemory
from rag import load_rag, build_context_block
from search import web_search, build_search_context


def main() -> None:
    llm    = load_model()
    memory = ConversationMemory()
 
    print("\n📚 Setting up knowledge base…")
    rag = load_rag()
 
    print(f"\n🌙 {ASSISTANT_NAME} is ready.")
    print("  Commands: 'exit' · 'clear' (memory) · 'reindex' (reload documents)")
    print("  Tip: start with 'search:' to force a web search on any query\n")
 
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
 
        if not user_input:
            continue
 
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
 
        if user_input.lower() == "clear":
            memory.clear()
            continue
 
        if user_input.lower() == "reindex":
            print("🔄 Reindexing documents…")
            rag = load_rag(force_reindex=True)
            continue
 
        context = None
 
        # ── Web search (takes priority over local RAG) ─────────────────────
        if should_search(user_input):
            print("🌐 Searching the web…", flush=True)
            results = web_search(user_input)
            if results:
                context = build_search_context(results, user_input)
                print(f"✅ Got {len(results)} result(s)")
            else:
                print("⚠️  No results found, falling back to local knowledge.")
 
        # ── Local RAG fallback ─────────────────────────────────────────────
        if context is None:
            rag_results = rag.retrieve(user_input)
            context     = build_context_block(rag_results)
            if context:
                sources = {r["source"] for r in rag_results}
                print(f"📂 Using local context from: {', '.join(sources)}")
 
        # ── Generate response ──────────────────────────────────────────────
        memory.add("user", user_input)
 
        print(f"\n{ASSISTANT_NAME}: ", end="", flush=True)
        reply = stream_response(llm, memory.get_history(), context=context)
 
        memory.add("assistant", reply)
        print()
 
 
if __name__ == "__main__":
    main()
