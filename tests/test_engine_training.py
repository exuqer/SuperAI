import json
import tempfile
import unittest
from pathlib import Path

from semantic_ants.engine import EngineConfig, SemanticEngine
from semantic_ants.learning import CheckpointStore, FeedbackTrainer, Trainer
from tests.fixtures import FakeConceptNetClient


class EngineTrainingTest(unittest.TestCase):
    def make_engine(self, tmp: str) -> SemanticEngine:
        store = CheckpointStore(Path(tmp) / "model.json")
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

    def test_feedback_penalizes_bad_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result = engine.analyze("apple", lang="en")
            before = dict(engine.checkpoint.suppressed_concepts)
            feedback = FeedbackTrainer(engine, engine.store).apply(result.result_id, score=1)
            self.assertGreater(feedback["changed_edges"], 0)
            self.assertNotEqual(before, engine.checkpoint.suppressed_concepts)


if __name__ == "__main__":
    unittest.main()
