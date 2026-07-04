import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from semantic_ants.engine import EngineConfig, SemanticEngine


class KnowledgeChatTest(unittest.TestCase):
    def test_builtin_greeting_routes_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(
                config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=8, max_depth=3)
            )
            result = engine.analyze("привет", lang="ru")
            self.assertTrue(result.response)
            self.assertTrue(any(item["uri"] == "/m/dialogue/greeting" for item in result.activated_concepts))

    def test_builtin_alphabet_routes_without_template_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp), allow_network=False))
            result = engine.analyze("покажи русский алфавит", lang="ru")
            self.assertTrue(result.response)
            self.assertTrue(any(item["uri"] == "/m/language/alphabet" for item in result.activated_concepts))

    def test_builtin_basic_concept_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp), allow_network=False))
            result = engine.analyze("что такое солнце", lang="ru")
            self.assertTrue(any(item["uri"] == "/m/basic/sun" for item in result.activated_concepts))
            self.assertTrue(result.response)
            self.assertNotIn("Солнце относится к области природы и природных явлений.", result.response)
            self.assertTrue(any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in result.response))

    def test_answer_language_matches_question_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp), allow_network=False))
            ru_result = engine.analyze("что такое солнце", lang="ru")
            en_result = engine.analyze("what is the sun", lang="en")
            self.assertTrue(any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in ru_result.response))
            self.assertFalse(any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in en_result.response))

    def test_chat_context_is_persisted_by_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp), allow_network=False))
            first = engine.analyze("hello", lang="en", session_id="test")
            second = engine.analyze("what now", lang="en", session_id="test")
            self.assertEqual(first.context_turns, [])
            self.assertTrue(any(turn["text"] == "hello" for turn in second.context_turns))
            self.assertGreaterEqual(len(engine.checkpoint.chat_sessions["test"]), 4)

    def test_chat_reset_session_clears_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp), allow_network=False))
            engine.analyze("hello", lang="en", session_id="test")
            result = engine.analyze("fresh start", lang="en", session_id="test", reset_session=True)
            self.assertEqual(result.context_turns, [])

    def test_chat_once_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "semantic_ants",
                    "--state-dir",
                    tmp,
                    "chat",
                    "--once",
                    "кто ты",
                    "--no-cache-refresh",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["response"])
            self.assertEqual(payload["session_id"], "default")


if __name__ == "__main__":
    unittest.main()
