import json
import tempfile
import unittest
from pathlib import Path

from semantic_ants.engine import EngineConfig, SemanticEngine
from semantic_ants.learning import CheckpointStore, FeedbackTrainer, Trainer
from tests.fixtures import FakeConceptNetClient


class EngineTrainingTest(unittest.TestCase):
    def make_engine(self, tmp: str) -> SemanticEngine:
        store = CheckpointStore(Path(tmp) / "model.bin")
        return SemanticEngine(
            config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=4, max_depth=2),
            client=FakeConceptNetClient(),
            store=store,
        )

    def test_engine_depth_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result = engine.analyze("apple", lang="en")
            self.assertTrue(result.routes)
            self.assertTrue(all(len(route.steps) <= 2 for route in result.routes))

    def test_analyze_with_graph_keeps_result_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result, graph = engine.analyze_with_graph("apple", lang="en")
            plain = engine.analyze("apple", lang="en")
            self.assertEqual(result.tokens, ["apple"])
            self.assertEqual(plain.tokens, result.tokens)
            self.assertTrue(graph.nodes)
            self.assertTrue(graph.edges())

    def test_strength_vector_limits_top_layer_and_records_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result = engine.analyze("apple", lang="en", strength_vector=(3,))
            self.assertTrue(result.routes)
            self.assertEqual(result.semantic_vector["strength_vector"], [3])
            self.assertTrue(result.signal_trace)
            self.assertTrue(all(step["layer"] == 0 for step in result.signal_trace))
            self.assertTrue(all(len(route.steps) <= 3 for route in result.routes))
            self.assertTrue(
                any(item["uri"] == "/m/top/object" for item in result.semantic_vector.get("items", []))
            )

    def test_strength_vector_falls_back_when_no_configured_layer_edge_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result = engine.analyze("кот", lang="ru", strength_vector=(3,))

            self.assertTrue(result.routes)
            self.assertTrue(any(route.steps for route in result.routes))
            self.assertTrue(result.signal_trace)

    def test_training_reinforces_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            path = Path(tmp) / "examples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "text": "apple",
                        "lang": "en",
                        "target_concepts": ["/c/en/apple_meaning"],
                        "target_response": "learned response",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = Trainer(engine, engine.store).train_file(path, epochs=1)
            self.assertEqual(report.examples, 1)
            self.assertGreater(report.reinforced_edges, 0)
            self.assertTrue(engine.checkpoint.response_memory)

    def test_training_aliases_question_and_expected_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            path = Path(tmp) / "aliases.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question": "как дела?",
                        "lang": "ru",
                        "target_concepts": ["/m/top/dialogue"],
                        "expected_answer": "Нормально, спасибо. А у тебя?",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = Trainer(engine, engine.store).train_file(path, epochs=1)
            self.assertEqual(report.errors, [])
            learned = [
                item
                for item in engine.checkpoint.accepted_answers
                if item.get("stimulus") == "как дела?" and item.get("answer") == "Нормально, спасибо. А у тебя?"
            ][0]
            self.assertEqual(learned["lang"], "ru")
            self.assertTrue(learned["answer_concepts"])

    def test_training_records_translation_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            path = Path(tmp) / "translation.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "text": "яблоко",
                        "lang": "ru",
                        "answer_lang": "en",
                        "target_concepts": ["/c/ru/яблоко"],
                        "target_response": "apple",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = Trainer(engine, engine.store).train_file(path, epochs=1)
            self.assertEqual(report.errors, [])
            learned = [
                item
                for item in engine.checkpoint.accepted_answers
                if item.get("stimulus") == "яблоко" and item.get("answer") == "apple"
            ][0]
            self.assertEqual(learned["lang"], "en")
            self.assertEqual(learned["source_lang"], "ru")
            self.assertTrue(
                any(
                    item.get("answer") == "apple"
                    and item.get("lang") == "en"
                    and item.get("source_lang") == "ru"
                    for item in engine.checkpoint.response_memory.values()
                )
            )

    def test_top_layer_training_reinforces_layer_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            path = Path(tmp) / "top.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "text": "apple",
                        "lang": "en",
                        "strength_vector": [3],
                        "layer_targets": {"0": ["/m/top/object"]},
                        "target_concepts": ["/m/top/object"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            before = engine.checkpoint.concept_pheromone_for("/m/top/object")
            report = Trainer(engine, engine.store).train_file(path, epochs=1)
            self.assertEqual(report.errors, [])
            self.assertGreater(engine.checkpoint.concept_pheromone_for("/m/top/object"), before)

    def test_layered_qa_example_tracks_layer_targets_and_strength_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            path = Path(tmp) / "layered.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question": "как дела?",
                        "lang": "ru",
                        "strength_vector": [3, 8, 8],
                        "layer_targets": {
                            "0": ["/m/top/dialogue"],
                            "1": ["/m/user/ru/вопрос"],
                            "2": ["/m/user/ru/дела"],
                        },
                        "target_concepts": [
                            "/m/top/dialogue",
                            "/m/user/ru/вопрос",
                            "/m/user/ru/дела",
                        ],
                        "concept_labels": {
                            "/m/top/dialogue": "Общение",
                            "/m/user/ru/вопрос": "Вопрос",
                            "/m/user/ru/дела": "дела",
                        },
                        "expected_answer": "Нормально, спасибо. А у тебя?",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = Trainer(engine, engine.store).train_file(path, epochs=1)
            self.assertEqual(report.errors, [])
            self.assertTrue(engine.checkpoint.accepted_answers)
            self.assertEqual(engine.checkpoint.accepted_answers[0]["lang"], "ru")
            self.assertTrue(engine.checkpoint.accepted_answers[0]["answer_concepts"])
            layer_targets = {
                edge.get("metadata", {}).get("layer_target")
                for edge in engine.checkpoint.custom_edges
                if edge.get("metadata", {}).get("layer_target") is not None
            }
            self.assertEqual(layer_targets, {"0", "1", "2"})
            result = engine.analyze("как дела?", lang="ru", strength_vector=(3, 8, 8))
            self.assertEqual(result.semantic_vector["strength_vector"], [3, 8, 8])

    def test_layer_target_creates_top_edge_for_new_word(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            path = Path(tmp) / "new_top.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "text": "нейроштука",
                        "lang": "ru",
                        "strength_vector": [3],
                        "layer_targets": {"0": ["/m/top/artifact"]},
                        "target_concepts": ["/m/top/artifact"],
                        "concept_labels": {"/m/top/artifact": "артефакт"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = Trainer(engine, engine.store).train_file(path, epochs=1)
            self.assertEqual(report.errors, [])
            result = engine.analyze("нейроштука", lang="ru", strength_vector=(3,))
            learned = [item for item in result.semantic_vector["items"] if item["uri"] == "/m/top/artifact"]
            self.assertTrue(learned)
            self.assertEqual(learned[0]["label"], "артефакт")

    def test_feedback_penalizes_bad_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result = engine.analyze("apple", lang="en")
            before = dict(engine.checkpoint.suppressed_concepts)
            feedback = FeedbackTrainer(engine, engine.store).apply(result.result_id, score=1)
            self.assertGreater(feedback["changed_edges"], 0)
            self.assertEqual(feedback["trained_dialogues"], 0)
            self.assertEqual(feedback["rejected_dialogues"], 1)
            self.assertNotEqual(before, engine.checkpoint.suppressed_concepts)

    def test_feedback_trains_positive_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result = engine.analyze("apple", lang="en")
            before_answers = len(engine.checkpoint.accepted_answers)
            before_responses = len(engine.checkpoint.response_memory)
            before_patterns = len(engine.checkpoint.mini_generator.get("dialogue_patterns", []))
            feedback = FeedbackTrainer(engine, engine.store).apply(result.result_id, score=5)
            self.assertGreater(feedback["changed_edges"], 0)
            self.assertEqual(feedback["trained_dialogues"], 1)
            self.assertEqual(feedback["rejected_dialogues"], 0)
            self.assertGreater(len(engine.checkpoint.accepted_answers), before_answers)
            self.assertGreater(len(engine.checkpoint.response_memory), before_responses)
            self.assertGreater(len(engine.checkpoint.mini_generator.get("dialogue_patterns", [])), before_patterns)


if __name__ == "__main__":
    unittest.main()
