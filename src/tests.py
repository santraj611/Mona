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


# tests.py - Test suite for Mona.
#
# Works with both:
#   python tests.py
#   pytest tests.py -v
#   pytest tests.py -v -k integration   (live network tests)

import sys
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_history(*pairs):
    h = []
    for user, asst in pairs:
        h.append({"role": "user",      "content": user})
        h.append({"role": "assistant", "content": asst})
    return h


def _import_llm():
    sys.modules.setdefault("llama_cpp", MagicMock())
    import importlib, llm as llm_mod
    importlib.reload(llm_mod)
    return llm_mod


# ---------------------------------------------------------------------------
# router.py - the fast classifier
# ---------------------------------------------------------------------------

class TestRouter(unittest.TestCase):

    def setUp(self):
        from router import should_search
        self.should_search = should_search

    # -- Should search --------------------------------------------------------

    def test_temporal_today(self):
        self.assertTrue(self.should_search("what's the weather today"))

    def test_temporal_latest(self):
        self.assertTrue(self.should_search("latest news on AI"))

    def test_temporal_right_now(self):
        self.assertTrue(self.should_search("what is happening right now in Gaza"))

    def test_live_price(self):
        self.assertTrue(self.should_search("price of bitcoin"))

    def test_live_stock(self):
        self.assertTrue(self.should_search("Apple stock price today"))

    def test_live_weather(self):
        self.assertTrue(self.should_search("will it rain tomorrow"))

    def test_live_score(self):
        self.assertTrue(self.should_search("what was the score of last night's game"))

    def test_current_role(self):
        self.assertTrue(self.should_search("who is the CEO of OpenAI"))

    def test_still_query(self):
        self.assertTrue(self.should_search("is Elon Musk still at Tesla"))

    def test_recent_release(self):
        self.assertTrue(self.should_search("have they released GPT-5 yet"))

    def test_force_prefix_search(self):
        self.assertTrue(self.should_search("search: best pizza recipe"))

    def test_force_prefix_find(self):
        self.assertTrue(self.should_search("find: open source models"))

    # -- Should NOT search ----------------------------------------------------

    def test_skip_greeting(self):
        self.assertFalse(self.should_search("hey how are you"))

    def test_skip_what_is_your_name(self):
        self.assertFalse(self.should_search("what is your name"))

    def test_skip_who_are_you(self):
        self.assertFalse(self.should_search("who are you"))

    def test_skip_explain(self):
        self.assertFalse(self.should_search("explain how neural networks work"))

    def test_skip_how_does_work(self):
        self.assertFalse(self.should_search("how does a transformer work"))

    def test_skip_difference_between(self):
        self.assertFalse(self.should_search("what is the difference between RAG and fine-tuning"))

    def test_skip_thanks(self):
        self.assertFalse(self.should_search("thanks!"))

    def test_skip_affirmation(self):
        self.assertFalse(self.should_search("ok got it"))

    def test_skip_joke(self):
        self.assertFalse(self.should_search("tell me a joke"))

    def test_skip_summarize(self):
        self.assertFalse(self.should_search("summarize our conversation"))

    def test_case_insensitive(self):
        self.assertTrue(self.should_search("LATEST NEWS ON PYTHON"))


class TestStripPrefix(unittest.TestCase):

    def setUp(self):
        from router import strip_prefix
        self.strip = strip_prefix

    def test_search_stripped(self):
        self.assertEqual(self.strip("search: AI news"), "AI news")

    def test_find_stripped(self):
        self.assertEqual(self.strip("find: something"), "something")

    def test_look_up_stripped(self):
        self.assertEqual(self.strip("look up: trains"), "trains")

    def test_no_prefix_unchanged(self):
        self.assertEqual(self.strip("normal query"), "normal query")

    def test_case_insensitive(self):
        self.assertEqual(self.strip("SEARCH: test"), "test")


# ---------------------------------------------------------------------------
# llm.py - prompt builder
# ---------------------------------------------------------------------------

class TestBuildPrompt(unittest.TestCase):

    def setUp(self):
        self.llm_mod = _import_llm()
        self.llm_mod.PROMPT_FORMAT = "chatml"
        self.build   = self.llm_mod._build_prompt

    def test_system_block_present(self):
        self.assertIn("<|im_start|>system",
                      self.build([{"role": "user", "content": "hi"}]))

    def test_ends_with_open_assistant_tag(self):
        prompt = self.build([{"role": "user", "content": "hi"}])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_user_content_present(self):
        self.assertIn("hello there",
                      self.build([{"role": "user", "content": "hello there"}]))

    def test_context_in_last_user_message(self):
        history = [{"role": "user", "content": "what is X?"}]
        prompt  = self.build(history, "UNIQUE_CTX_MARKER")
        user_start = prompt.index("<|im_start|>user")
        self.assertIn("UNIQUE_CTX_MARKER", prompt[user_start:])

    def test_context_not_in_system_block(self):
        history = [{"role": "user", "content": "q"}]
        prompt  = self.build(history, "UNIQUE_CTX_MARKER")
        sys_end = prompt.index("<|im_end|>")
        self.assertNotIn("UNIQUE_CTX_MARKER", prompt[:sys_end])

    def test_earlier_turns_untouched(self):
        history = make_history(("first q", "first a"))
        history.append({"role": "user", "content": "second q"})
        self.assertIn("<|im_start|>user\nfirst q<|im_end|>",
                      self.build(history, "CTX"))

    def test_no_context_no_injection_text(self):
        prompt = self.build([{"role": "user", "content": "hi"}], context=None)
        self.assertNotIn("Use the following information", prompt)

    def test_multi_turn_order(self):
        history = make_history(("q1", "a1"), ("q2", "a2"))
        history.append({"role": "user", "content": "q3"})
        prompt  = self.build(history)
        positions = [prompt.index(t) for t in ["q1", "a1", "q2", "a2", "q3"]]
        self.assertEqual(positions, sorted(positions))

    def test_gemma_prompt_format_tags(self):
        self.llm_mod.PROMPT_FORMAT = "gemma"
        prompt = self.build([{"role": "user", "content": "hi"}])
        self.assertIn("<start_of_turn>user", prompt)
        self.assertIn("<start_of_turn>model\n", prompt)
        self.assertNotIn("<|im_start|>", prompt)

    def test_gemma_maps_assistant_role_to_model(self):
        self.llm_mod.PROMPT_FORMAT = "gemma"
        history = make_history(("q1", "a1"))
        prompt = self.build(history)
        self.assertIn("<start_of_turn>model\na1<end_of_turn>", prompt)

    def test_auto_prompt_format_uses_model_path(self):
        self.llm_mod.PROMPT_FORMAT = "auto"
        self.llm_mod.MODEL_PATH = "/tmp/gemma-3-4b-it-q4_k_m.gguf"
        self.assertEqual(self.llm_mod._resolve_prompt_format(), "gemma")

    def test_auto_prompt_format_detects_llama3(self):
        self.llm_mod.PROMPT_FORMAT = "auto"
        self.llm_mod.MODEL_PATH = "/tmp/llama-3.1-8b-instruct-q4_k_m.gguf"
        self.assertEqual(self.llm_mod._resolve_prompt_format(), "llama3")

    def test_auto_prompt_format_detects_mistral(self):
        self.llm_mod.PROMPT_FORMAT = "auto"
        self.llm_mod.MODEL_PATH = "/tmp/mistral-7b-instruct-v0.3-q4_k_m.gguf"
        self.assertEqual(self.llm_mod._resolve_prompt_format(), "mistral")

    def test_llama3_prompt_uses_header_tokens(self):
        self.llm_mod.PROMPT_FORMAT = "llama3"
        prompt = self.build([{"role": "user", "content": "hi"}])
        self.assertIn("<|start_header_id|>user<|end_header_id|>", prompt)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", prompt)

    def test_mistral_prompt_uses_inst_tags(self):
        self.llm_mod.PROMPT_FORMAT = "mistral"
        prompt = self.build([{"role": "user", "content": "hi"}])
        self.assertIn("[INST]", prompt)
        self.assertIn("[/INST]", prompt)


# ---------------------------------------------------------------------------
# memory.py
# ---------------------------------------------------------------------------

class TestConversationMemory(unittest.TestCase):

    def setUp(self):
        from memory import ConversationMemory
        self.Memory = ConversationMemory

    def test_add_and_retrieve(self):
        mem = self.Memory(max_turns=10)
        mem.add("user", "hello")
        mem.add("assistant", "hi")
        h = mem.get_history()
        self.assertEqual(h[0]["content"], "hello")
        self.assertEqual(h[1]["content"], "hi")

    def test_max_turns_evicts_oldest(self):
        mem = self.Memory(max_turns=2)
        for i in range(3):
            mem.add("user", f"q{i}")
            mem.add("assistant", f"a{i}")
        contents = [h["content"] for h in mem.get_history()]
        self.assertNotIn("q0", contents)
        self.assertIn("q2", contents)

    def test_clear(self):
        mem = self.Memory()
        mem.add("user", "hello")
        mem.clear()
        self.assertEqual(mem.get_history(), [])

    def test_empty_on_init(self):
        self.assertEqual(self.Memory().get_history(), [])


# ---------------------------------------------------------------------------
# search.py
# ---------------------------------------------------------------------------

class TestBuildSearchContext(unittest.TestCase):

    def setUp(self):
        from search import build_search_context
        self.build = build_search_context

    def test_empty_returns_none(self):
        self.assertIsNone(self.build([], "query"))

    def test_content_present(self):
        results = [{"title": "Test", "url": "https://t.com", "body": "Some content"}]
        ctx = self.build(results, "query")
        self.assertIn("Test", ctx)
        self.assertIn("https://t.com", ctx)
        self.assertIn("Some content", ctx)

    def test_numbered(self):
        results = [
            {"title": "A", "url": "https://a.com", "body": "Body A"},
            {"title": "B", "url": "https://b.com", "body": "Body B"},
        ]
        ctx = self.build(results, "query")
        self.assertIn("[1]", ctx)
        self.assertIn("[2]", ctx)

    def test_query_in_context(self):
        results = [{"title": "X", "url": "https://x.com", "body": "body"}]
        self.assertIn("my query", self.build(results, "my query"))


# ---------------------------------------------------------------------------
# rag.py - chunking
# ---------------------------------------------------------------------------

class TestChunking(unittest.TestCase):

    def setUp(self):
        from rag import _chunk_text
        self.chunk = _chunk_text

    def test_short_text_single_chunk(self):
        chunks = self.chunk("Hello world", "test.txt")
        self.assertEqual(len(chunks), 1)

    def test_long_text_multiple_chunks(self):
        self.assertGreater(len(self.chunk("A" * 1000, "big.txt")), 1)

    def test_empty_text_no_chunks(self):
        self.assertEqual(self.chunk("", "empty.txt"), [])

    def test_source_is_basename(self):
        self.assertEqual(self.chunk("text", "/path/to/file.txt")[0]["source"], "file.txt")

    def test_all_chunks_have_keys(self):
        for c in self.chunk("word " * 200, "doc.txt"):
            self.assertIn("text",   c)
            self.assertIn("source", c)


# ---------------------------------------------------------------------------
# Integration (opt-in)
# ---------------------------------------------------------------------------

class TestWebSearchIntegration(unittest.TestCase):

    def setUp(self):
        if "integration" not in " ".join(sys.argv):
            self.skipTest("Pass --integration or -k integration to run")

    def test_ddg_returns_results(self):
        from search import web_search
        results = web_search("Python programming language")
        self.assertGreater(len(results), 0)
        self.assertIn("url", results[0])

    def test_paris_in_results(self):
        from search import web_search
        bodies = " ".join(r["body"] for r in web_search("capital of France")).lower()
        self.assertIn("paris", bodies)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    argv = [a for a in sys.argv if a != "--integration"]
    unittest.main(argv=argv, verbosity=2)
