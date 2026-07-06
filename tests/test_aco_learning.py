import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from semantic_ants.core.models import SemanticEdge
from semantic_ants.engine import EngineConfig, SemanticEngine
from semantic_ants.learning import ACOTrainer, CheckpointStore, Experience, FeedbackTrainer
from tests.fixtures import FakeConceptNetClient


class ACOLearningTest(unittest.TestCase):
    def make_engine(self, tmp: str) -> SemanticEngine:
        store = CheckpointStore(Path(tmp) / "model.bin")
        return SemanticEngine(
            config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=6, max_depth=2),
            client=FakeConceptNetClient(),
            store=store,
        )

    def test_good_experience_reinforces_pheromones(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            trainer = ACOTrainer(engine, engine.store)
            edge = SemanticEdge("/c/en/apple", "/c/en/apple_meaning", "RelatedTo")
            before = engine.checkpoint.pheromone_for(edge)
            trainer.learn_experience(
                Experience(
                    stimulus="apple",
                    lang="en",
                    target_concepts=["/c/en/apple_meaning"],
                    accepted_answer="Apple points to the learned meaning.",
                    positive_edges=[("/c/en/apple", "RelatedTo", "/c/en/apple_meaning")],
                    reward=1.0,
                )
            )
            self.assertGreater(engine.checkpoint.pheromone_for(edge), before)
            self.assertTrue(engine.checkpoint.accepted_answers)

    def test_positive_edge_dict_preserves_layer_metadata(self):
        experience = Experience.from_dict(
            {
                "stimulus": "apple",
                "lang": "en",
                "positive_edges": [
                    {
                        "start": "/c/en/apple",
                        "relation": "InTopDomain",
                        "end": "/m/top/object",
                        "layer": 0,
                        "distance": 8,
                        "edge_type": "domain",
                        "metadata": {"source": "web_training_form"},
                    }
                ],
            }
        )
        self.assertIsInstance(experience.positive_edges[0], dict)
        edge = experience.positive_edges[0]
        self.assertEqual(edge["layer"], 0)
        self.assertEqual(edge["distance"], 8.0)
        self.assertEqual(edge["edge_type"], "domain")
        self.assertEqual(edge["metadata"]["source"], "web_training_form")
        self.assertEqual(experience.to_dict()["positive_edges"][0]["metadata"]["source"], "web_training_form")

    def test_bad_experience_evaporates_false_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            trainer = ACOTrainer(engine, engine.store)
            edge = SemanticEdge("/c/en/apple", "/c/en/apple_meaning", "RelatedTo")
            before = engine.checkpoint.pheromone_for(edge)
            trainer.learn_experience(
                Experience(
                    stimulus="apple",
                    lang="en",
                    rejected_answers=["Apple points to the learned meaning."],
                    negative_concepts=["/c/en/apple_meaning"],
                    reward=-1.0,
                )
            )
            self.assertLess(engine.checkpoint.pheromone_for(edge), before)
            self.assertIn("/c/en/apple_meaning", engine.checkpoint.suppressed_concepts)
            self.assertTrue(engine.checkpoint.negative_memory)

    def test_feedback_changes_hybrid_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result = engine.analyze("apple", lang="en")
            before_patterns = len(engine.checkpoint.mini_generator.get("dialogue_patterns", []))
            FeedbackTrainer(engine, engine.store).apply(
                result.result_id,
                score=5,
                corrected_concepts=["/c/en/apple_meaning"],
                corrected_response="Corrected apple answer.",
            )
            self.assertGreater(len(engine.checkpoint.mini_generator.get("dialogue_patterns", [])), before_patterns)
            hybrid = engine.analyze("apple", lang="en", mode="hybrid")
            self.assertEqual(hybrid.response, result.response)
            self.assertIn("apple", hybrid.response.lower())
            self.assertNotIn("corrected", hybrid.response.lower())

    def test_dream_creates_only_weak_bridges(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            engine.checkpoint.add_learned_bridge("/c/en/a", "Confirmed", "/c/en/b", weight=2.0, confirmed=True)
            engine.checkpoint.add_learned_bridge("/c/en/b", "Confirmed", "/c/en/c", weight=2.0, confirmed=True)
            report = ACOTrainer(engine, engine.store).dream(steps=12)
            dream_bridges = [
                bridge for bridge in engine.checkpoint.learned_bridges if bridge.get("metadata", {}).get("dream")
            ]
            self.assertEqual(report["bridges"], len(dream_bridges))
            self.assertTrue(all(float(bridge["weight"]) <= 0.12 for bridge in dream_bridges))
            confirmed = [
                bridge
                for bridge in engine.checkpoint.learned_bridges
                if bridge.get("start") == "/c/en/a" and bridge.get("end") == "/c/en/b"
            ][0]
            self.assertTrue(confirmed["confirmed"])
            self.assertEqual(confirmed["weight"], 2.0)

    def test_learn_cli_supports_experience_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiences.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "stimulus": "apple",
                        "lang": "en",
                        "target_concepts": ["/c/en/apple"],
                        "accepted_answer": "Apple accepted.",
                        "reward": 1.0,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "semantic_ants",
                    "--state-dir",
                    tmp,
                    "learn",
                    str(path),
                    "--epochs",
                    "1",
                    "--no-cache-refresh",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("examples=1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
