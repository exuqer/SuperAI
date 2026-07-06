import tempfile
import unittest
from pathlib import Path

from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import SemanticEdge
from semantic_ants.engine import EngineConfig, SemanticEngine
from semantic_ants.learning import CheckpointStore, FeedbackTrainer
from semantic_ants.server.graph import concept_detail, graph_from_checkpoint, graph_snapshot, trace_interpretation
from tests.fixtures import FakeConceptNetClient


class ServerGraphTest(unittest.TestCase):
    def make_engine(self, tmp: str) -> SemanticEngine:
        store = CheckpointStore(Path(tmp) / "model.bin")
        return SemanticEngine(
            config=EngineConfig(state_dir=Path(tmp), allow_network=False, ant_count=4, max_depth=2),
            client=FakeConceptNetClient(),
            store=store,
        )

    def test_analyze_graph_snapshot_marks_signal_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result, graph = engine.analyze_with_graph("apple", lang="en", strength_vector=(3,))
            snapshot = graph_snapshot(graph, engine.checkpoint, result)
            self.assertTrue(snapshot["nodes"])
            self.assertTrue(snapshot["edges"])
            self.assertTrue(any(edge["signal"]["active"] for edge in snapshot["edges"]))
            self.assertTrue(any(node["signal"]["active"] for node in snapshot["nodes"]))

    def test_focused_graph_keeps_coverage_edges_for_active_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            graph = SemanticGraph()
            graph.add_edge(SemanticEdge("a", "b", "r1"))
            graph.add_edge(SemanticEdge("c", "d", "r2"))
            result = {
                "routes": [
                    {
                        "ant_id": 1,
                        "start": "a",
                        "concepts": ["a", "b"],
                        "total_score": 1.0,
                        "steps": [
                            {
                                "start": "a",
                                "end": "b",
                                "relation": "r1",
                                "edge_weight": 1.0,
                                "pheromone": 1.0,
                                "score": 1.0,
                                "source": "checkpoint",
                                "layer": 1,
                                "distance": 1.0,
                                "remaining_strength": None,
                                "edge_type": "semantic",
                            }
                        ],
                    },
                    {
                        "ant_id": 2,
                        "start": "c",
                        "concepts": ["c", "d"],
                        "total_score": 1.0,
                        "steps": [
                            {
                                "start": "c",
                                "end": "d",
                                "relation": "r2",
                                "edge_weight": 1.0,
                                "pheromone": 1.0,
                                "score": 1.0,
                                "source": "checkpoint",
                                "layer": 1,
                                "distance": 1.0,
                                "remaining_strength": None,
                                "edge_type": "semantic",
                            }
                        ],
                    },
                ],
                "signal_trace": [],
                "semantic_vector": {"items": []},
            }

            snapshot = graph_snapshot(graph, engine.checkpoint, result, limit=1)
            edge_ids = [edge["id"] for edge in snapshot["edges"]]
            self.assertIn("a|r1|b", edge_ids)
            self.assertIn("c|r2|d", edge_ids)

    def test_checkpoint_graph_filter_by_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            graph = graph_from_checkpoint(engine.checkpoint)
            snapshot = graph_snapshot(graph, engine.checkpoint, layer=0, limit=50)
            self.assertTrue(snapshot["edges"])
            self.assertTrue(all(edge["layer"] == 0 for edge in snapshot["edges"]))

    def test_concept_detail_contains_edges_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            detail = concept_detail(engine.checkpoint, "/m/top/object")
            self.assertEqual(detail["node"]["uri"], "/m/top/object")
            self.assertTrue(detail["incoming"] or detail["outgoing"])

    def test_query_snapshot_links_unseen_concept_to_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            graph = graph_from_checkpoint(engine.checkpoint)
            snapshot = graph_snapshot(
                graph,
                engine.checkpoint,
                layer=1,
                relation="InLanguage",
                edge_type="language",
                min_pheromone=1.0,
                query="/c/ru/\u043a\u043d\u0438\u0433\u0430",
                limit=50,
            )
            self.assertTrue(snapshot["edges"])
            self.assertTrue(
                any(
                    edge["start"] == "/c/ru/\u043a\u043d\u0438\u0433\u0430"
                    and edge["end"] == "/m/language/ru"
                    for edge in snapshot["edges"]
                )
            )

    def test_concept_detail_links_unseen_concept_to_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            detail = concept_detail(engine.checkpoint, "/c/ru/\u043a\u043d\u0438\u0433\u0430")
            self.assertEqual(detail["node"]["uri"], "/c/ru/\u043a\u043d\u0438\u0433\u0430")
            self.assertTrue(any(edge["relation"] == "InLanguage" for edge in detail["outgoing"]))

    def test_trace_interpretation_lists_active_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result = engine.analyze("apple", lang="en", strength_vector=(3,))
            payload = trace_interpretation(result)
            self.assertTrue(payload["active_edge_ids"])
            self.assertTrue(payload["chains"])

    def test_feedback_changes_checkpoint_for_api_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            result = engine.analyze("apple", lang="en")
            before = dict(engine.checkpoint.suppressed_concepts)
            feedback = FeedbackTrainer(engine, engine.store).apply(result.result_id, score=1)
            self.assertGreater(feedback["changed_edges"], 0)
            self.assertNotEqual(before, engine.checkpoint.suppressed_concepts)


if __name__ == "__main__":
    unittest.main()
