import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from semantic_ants.engine import EngineConfig, SemanticEngine


def _has_cyrillic(value: str) -> bool:
    return any("\u0430" <= ch.lower() <= "\u044f" or ch.lower() == "\u0451" for ch in value)


class KnowledgeChatTest(unittest.TestCase):
    def test_builtin_greeting_routes_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(
                config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=8, max_depth=3)
            )
            result = engine.analyze("\u043f\u0440\u0438\u0432\u0435\u0442", lang="ru")
            self.assertTrue(result.response)
            self.assertIn("\u043f\u0440\u0438\u0432\u0435\u0442", result.response.lower())
            self.assertEqual(result.response_source, "exact_memory")
            self.assertTrue(any(item["uri"] == "/m/top/dialogue" for item in result.activated_concepts))

    def test_builtin_wellbeing_question_uses_graph_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(
                config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=8, max_depth=3)
            )
            result = engine.analyze("\u043a\u0430\u043a \u0434\u0435\u043b\u0430?", lang="ru")
            self.assertTrue(result.response)
            self.assertIn("\u0442\u0435\u0431", result.response.lower())
            self.assertEqual(result.response_source, "exact_memory")
            self.assertTrue(any(item["uri"] == "/m/top/dialogue" for item in result.activated_concepts))

    def test_unknown_input_stays_graph_driven(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(
                config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=8, max_depth=3)
            )
            result = engine.analyze("\u043f\u0432\u0430\u043f", lang="ru")
            self.assertTrue(result.response)
            self.assertIn("\u043f\u0432\u0430\u043f", result.response.lower())
            self.assertTrue(result.activated_concepts)
            self.assertEqual(result.activated_concepts[0]["uri"], "/m/concept/\u043f\u0432\u0430\u043f")

    def test_unseen_content_word_gets_domain_route_and_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(
                config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=8, max_depth=3)
            )
            result, graph = engine.analyze_with_graph("\u043a\u043d\u0438\u0433\u0430", lang="ru", strength_vector=(3,))
            self.assertTrue(result.response)
            self.assertIn("\u043a\u043d\u0438\u0433\u0430", result.response.lower())
            self.assertIn("\u043f\u0440\u0435\u0434\u043c\u0435\u0442", result.response.lower())
            self.assertTrue(
                any(
                    edge.start == "/m/concept/\u043a\u043d\u0438\u0433\u0430"
                    and edge.end == "/m/top/object"
                    and edge.relation == "InTopDomain"
                    and edge.edge_type == "domain"
                    for edge in graph.edges()
                )
            )
            self.assertTrue(any(step["relation"] == "InTopDomain" for step in result.signal_trace))

    def test_new_content_turn_does_not_answer_previous_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(
                config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=8, max_depth=3)
            )
            engine.analyze("\u041f\u0440\u0438\u0432\u0435\u0442!", lang="ru", session_id="test", strength_vector=(3,))
            result = engine.analyze("\u041a\u043e\u043c\u043f\u044c\u044e\u0442\u0435\u0440", lang="ru", session_id="test", strength_vector=(3,))

            self.assertNotEqual(result.response_source, "contextual_follow_up")
            self.assertNotIn("\u0422\u044b \u0441\u043f\u0440\u0430\u0448\u0438\u0432\u0430\u043b", result.response)
            self.assertNotIn("\u041f\u0440\u0438\u0432\u0435\u0442", result.response)
            self.assertIn("\u043a\u043e\u043c\u043f\u044c\u044e\u0442\u0435\u0440", result.response.lower())

    def test_builtin_alphabet_routes_without_template_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp), allow_network=False))
            result = engine.analyze("\u043f\u043e\u043a\u0430\u0436\u0438 \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0430\u043b\u0444\u0430\u0432\u0438\u0442", lang="ru")
            self.assertTrue(result.response)
            self.assertIn("\u0430\u043b\u0444\u0430\u0432\u0438\u0442", result.response.lower())

    def test_builtin_basic_concept_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp), allow_network=False))
            result = engine.analyze("\u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435 \u0441\u043e\u043b\u043d\u0446\u0435", lang="ru")
            self.assertTrue(result.response)
            self.assertIn("\u0441\u043e\u043b\u043d\u0446\u0435", result.response.lower())
            self.assertIn("\u0437\u0432\u0435\u0437\u0434\u0430", result.response.lower())
            self.assertTrue(any(item["uri"] == "/m/top/nature" for item in result.activated_concepts))

    def test_content_word_outweighs_question_word(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp), allow_network=False))
            result = engine.analyze("\u0410 \u0434\u0435\u0440\u0435\u0432\u043e \u044d\u0442\u043e \u0447\u0442\u043e ?", lang="ru")
            self.assertTrue(result.response)
            self.assertEqual(result.activated_concepts[0]["uri"], "/m/concept/\u0434\u0435\u0440\u0435\u0432\u043e")
            self.assertIn("\u0434\u0435\u0440\u0435\u0432\u043e", result.response.lower())
            self.assertNotIn("\u0447\u0442\u043e \u2014", result.response.lower())

    def test_answer_language_matches_question_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp), allow_network=False))
            ru_result = engine.analyze("\u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435 \u0441\u043e\u043b\u043d\u0446\u0435", lang="ru")
            en_result = engine.analyze("what is the sun", lang="en")
            self.assertTrue(_has_cyrillic(ru_result.response))
            self.assertTrue(en_result.response)

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
                    "\u043a\u0442\u043e \u0442\u044b",
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
