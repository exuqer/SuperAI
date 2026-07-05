import importlib.util
import tempfile
import unittest
from pathlib import Path


PYMORPHY_AVAILABLE = bool(importlib.util.find_spec("pymorphy3"))


@unittest.skipUnless(PYMORPHY_AVAILABLE, 'install dependencies with: pip install -e ".[web]"')
class SimpleTrainingTest(unittest.TestCase):
    def make_engine(self, tmp: str):
        from semantic_ants.engine import EngineConfig, SemanticEngine
        from tests.fixtures import FakeConceptNetClient

        return SemanticEngine(
            config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=4, max_depth=2),
            client=FakeConceptNetClient(),
        )

    def test_simple_training_builds_token_edges_and_memory(self):
        from semantic_ants.learning import SimpleQATrainer

        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            report = SimpleQATrainer(engine, engine.store).train_payload(
                {
                    "question": "что делает программист?",
                    "expected_answer": "Программист пишет код на компьютере.",
                    "lang": "ru",
                    "concept_meanings": [
                        {
                            "concept": "/c/ru/программист",
                            "label": "программист",
                            "meaning": "человек, который пишет код",
                        }
                    ],
                }
            )

            self.assertEqual(report.errors, [])
            self.assertIn("программист", report.question_tokens)
            self.assertIn("писать", report.answer_tokens)
            self.assertNotIn("что", report.question_tokens)
            self.assertTrue(engine.checkpoint.accepted_answers)
            relations = {edge["relation"] for edge in engine.checkpoint.custom_edges}
            self.assertIn("ExpectedAnswerToken", relations)
            self.assertIn("AnswerNextToken", relations)
            self.assertIn("MeaningHint", relations)
            self.assertIn("DescribedByToken", relations)
            definitions = engine.checkpoint.metadata["concept_definitions"]
            self.assertEqual(definitions["/c/ru/программист"]["meaning"], "человек, который пишет код")

    def test_simple_training_teaches_decoder_role_edges(self):
        from semantic_ants.decoding import decode_words
        from semantic_ants.learning import SimpleQATrainer

        with tempfile.TemporaryDirectory() as tmp:
            from semantic_ants.engine import EngineConfig, SemanticEngine

            engine = SemanticEngine(
                config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=8, max_depth=3)
            )
            SimpleQATrainer(engine, engine.store).train_payload(
                {
                    "question": "что делает программист?",
                    "expected_answer": "Программист пишет код на компьютере.",
                    "lang": "ru",
                }
            )

            result = decode_words(
                "",
                lang="ru",
                tokens=["компьютер", "код", "писать", "программист"],
                checkpoint=engine.checkpoint,
            )

            self.assertEqual(result.sentence, "программист пишет код на компьютере")
            self.assertEqual([token.role for token in result.tokens], ["subject", "verb", "object", "instrument"])

    def test_simple_training_answer_is_used_by_chat(self):
        from semantic_ants.learning import SimpleQATrainer

        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            SimpleQATrainer(engine, engine.store).train_payload(
                {
                    "question": "КАК ДУМАЕШЬ ЧТО ТАКОЕ ОСЕНЬ",
                    "expected_answer": "Осень - это время года, когда желтеют листья",
                    "lang": "ru",
                    "concept_meanings": [
                        {
                            "concept": "/c/ru/осень",
                            "label": "осень",
                            "meaning": "время года",
                        }
                    ],
                }
            )

            result = engine.analyze("Осень это что?", lang="ru")

            self.assertEqual(result.response, "Осень - это время года, когда желтеют листья")

    def test_learned_meaning_can_render_as_chat_answer(self):
        from semantic_ants.learning import SimpleQATrainer

        with tempfile.TemporaryDirectory() as tmp:
            from semantic_ants.engine import EngineConfig, SemanticEngine

            engine = SemanticEngine(
                config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=8, max_depth=3)
            )
            SimpleQATrainer(engine, engine.store).train_payload(
                {
                    "question": "что такое осень",
                    "expected_answer": "Осень - это время года",
                    "lang": "ru",
                    "concept_meanings": [
                        {
                            "concept": "/c/ru/осень",
                            "label": "осень",
                            "meaning": "время года",
                        }
                    ],
                }
            )

            result = engine.analyze("осень", lang="ru", strength_vector=(3,))

            self.assertIn("осень", result.response.lower())
            self.assertIn("время года", result.response.lower())


if __name__ == "__main__":
    unittest.main()
