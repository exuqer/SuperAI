from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import mock
import unittest

from semantic_ants.engine import DEFAULT_VECTOR_DIM, EngineConfig, SemanticEngine, token_id, zero_vector
from semantic_ants.state import Checkpoint, save_checkpoint


def vec(first: float, second: float = 0.0, third: float = 0.0) -> list[float]:
    values = [0.0] * DEFAULT_VECTOR_DIM
    values[0] = first
    values[1] = second
    values[2] = third
    return values


class FakeEmbeddingBackend:
    dim = DEFAULT_VECTOR_DIM

    def ensure(self) -> bool:
        return True

    def encode(self, text: str) -> list[float]:
        mapping = {
            "транспорт": vec(1.0),
            "самокат": vec(0.9, 0.1),
            "молоко": vec(0.1, 0.9),
            "далекий": vec(-1.0),
            "ответ": vec(0.8),
            "привет": vec(0.7),
            "дальше": vec(0.0, 0.1),
        }
        return list(mapping.get(text.casefold(), vec(0.0, 0.0, 1.0)))

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(text) for text in texts]


class EngineVectorTests(unittest.TestCase):
    def test_missing_sentence_transformers_falls_back_to_zero_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
                engine = SemanticEngine(config=EngineConfig(state_dir=state_dir))
                vector = engine._embedding_vector("транспорт")

            self.assertEqual(len(vector), DEFAULT_VECTOR_DIM)
            self.assertEqual(vector, zero_vector())
            self.assertFalse(engine._embedding_available())

    def test_sqlite_checkpoint_round_trip_preserves_tokens_edges_and_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = state_dir / "checkpoint.sqlite"
            checkpoint = Checkpoint(
                tokens={
                    "велосипед": {
                        "id": "token:велосипед",
                        "type": "token",
                        "token": "велосипед",
                        "label": "велосипед",
                        "count": 3,
                        "vector": vec(0.25, 0.75),
                        "created_at": 1.0,
                        "updated_at": 2.0,
                    },
                    "__assistant__": {
                        "id": "token:__assistant__",
                        "type": "token",
                        "token": "__assistant__",
                        "label": "__assistant__",
                        "count": 1,
                        "vector": vec(1.0),
                        "created_at": 1.0,
                        "updated_at": 2.0,
                    },
                },
                sessions={"unit": [{"role": "user", "text": "hello"}]},
                results={
                    "result:stored": {
                        "result_id": "result:stored",
                        "trace": {},
                        "backpack": {"node_ids": ["token:велосипед"]},
                    }
                },
                meta={
                    "hypernodes": {
                        "hyper:__root__": {
                            "id": "hyper:__root__",
                            "type": "hypernode",
                            "label": "__root__",
                            "hierarchy": [],
                            "parent": None,
                            "depth": 0,
                            "count": 1,
                            "vector": vec(0.1),
                            "created_at": 1.0,
                            "updated_at": 2.0,
                            "subgraph": {
                                "tokens": {
                                    "token:велосипед": {
                                        "id": "token:велосипед",
                                        "type": "token",
                                        "token": "велосипед",
                                        "label": "велосипед",
                                        "count": 3,
                                    }
                                },
                                "edges": {
                                    "token:__assistant__|next|token:велосипед": {
                                        "weight": 2.0,
                                        "pheromone": 1.5,
                                    }
                                },
                            },
                        }
                    },
                    "transition_memory": {
                        "token:__assistant__|then|token:велосипед": {"token:ответ": 4}
                    },
                },
            )
            save_checkpoint(checkpoint_path, checkpoint)

            engine = SemanticEngine(config=EngineConfig(state_dir=state_dir))
            engine.save()
            reloaded = SemanticEngine(config=EngineConfig(state_dir=state_dir))

            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(reloaded.checkpoint.tokens["велосипед"]["count"], 3)
            self.assertEqual(reloaded.checkpoint.tokens["велосипед"]["vector"][:2], [0.25, 0.75])
            self.assertIn("hyper:__root__", reloaded.checkpoint.meta["hypernodes"])
            self.assertIn("token:__assistant__|next|token:велосипед", reloaded.checkpoint.meta["hypernodes"]["hyper:__root__"]["subgraph"]["edges"])
            self.assertEqual(reloaded.checkpoint.meta["transition_memory"]["token:__assistant__|then|token:велосипед"]["token:ответ"], 4)

    def test_training_writes_lowercase_token_label_and_embedding_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()

            engine.train_text("Самокат едет быстро.", session_id="unit")

            record = engine.checkpoint.tokens["самокат"]
            self.assertEqual(record["label"], "самокат")
            self.assertEqual(record["vector"], vec(0.9, 0.1))

    def test_training_batches_new_tokens_and_skips_repeat_subgraph_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()

            with mock.patch.object(engine, "_embedding_many", wraps=engine._embedding_many) as embed_many:
                engine.train_text("Самокат самокат самокат.", session_id="unit")

            self.assertEqual(embed_many.call_count, 1)
            self.assertEqual(embed_many.call_args.args[0], ["самокат", "."])

    def test_training_writes_only_next_token_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()

            report = engine.train_text("Привет как дела? => Все хорошо спасибо", session_id="unit")

            self.assertGreaterEqual(report["source_pairs"], 1)
            self.assertGreaterEqual(report["source_sequences"], 1)
            root_edges = engine.checkpoint.meta["hypernodes"]["hyper:__root__"]["subgraph"]["edges"]
            root_tokens = engine.checkpoint.meta["hypernodes"]["hyper:__root__"]["subgraph"]["tokens"]
            self.assertTrue(root_edges)
            self.assertIn("token:__user__|next|token:привет", root_edges)
            self.assertIn("token:__assistant__|next|token:все", root_edges)
            self.assertTrue(all(set(edge) <= {"weight", "pheromone"} for edge in root_edges.values()))
            self.assertIn("token:привет", root_tokens)
            self.assertIn("token:все", root_tokens)

    def test_graph_payload_contains_only_circle_token_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()
            engine.train_text("Привет как дела? => Все хорошо спасибо", session_id="unit")

            graph = engine.graph(query="привет", limit=24)

            self.assertTrue(graph["nodes"])
            self.assertTrue(all(node["type"] == "token" for node in graph["nodes"]))
            self.assertTrue(all(node["shape"] == "circle" for node in graph["nodes"]))

    def test_chat_backpack_omits_recursive_layers_and_graph_data_until_lazy_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()
            engine.train_text("Привет! Расскажи анекдот -- Блин! - сказал слон, наступив на колобка.", session_id="unit")

            payload = engine.chat("Привет! Расскажи анекдот", session_id="unit")
            backpack = payload["backpack"]

            self.assertNotIn("layers", backpack)
            self.assertNotIn("graph_data", backpack)
            lazy_backpack = engine.backpack(result_id=payload["result"]["result_id"])
            self.assertIn("graph_data", lazy_backpack)
            self.assertTrue(lazy_backpack["graph_data"]["nodes"])
            self.assertTrue(lazy_backpack["graph_data"]["edges"])

    def test_chat_response_uses_lazy_backpack_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()
            engine.train_text("Привет! Расскажи анекдот -- Блин! - сказал слон.", session_id="unit")

            response = engine.chat("Привет! Расскажи анекдот", session_id="unit")
            self.assertNotIn("graph_data", response["backpack"])

            backpack = engine.backpack(result_id=response["result"]["result_id"])
            self.assertIn("graph_data", backpack)
            self.assertTrue(backpack["graph_data"]["nodes"])
            self.assertTrue(backpack["graph_data"]["edges"])

    def test_chat_uses_deferred_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()
            engine.train_text("Привет! Расскажи анекдот.", session_id="unit")

            with mock.patch.object(engine, "_schedule_persist", wraps=engine._schedule_persist) as schedule_persist, \
                mock.patch.object(engine, "_persist", wraps=engine._persist) as persist:
                engine.chat("Привет!", session_id="unit")

            self.assertGreaterEqual(schedule_persist.call_count, 1)
            self.assertEqual(persist.call_count, 0)

    def test_generation_candidates_apply_semantic_suppression_repetition_and_terminal_gravity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()
            now = 1.0
            report = engine._empty_train_report(session_id="unit", epochs=1)
            for token in ["__assistant__", "самокат", "молоко", "далекий", "привет", "ответ", "дальше", "."]:
                engine._ensure_token(token, report, now)
            engine.checkpoint.tokens["самокат"]["vector"] = vec(0.9, 0.1)
            engine.checkpoint.tokens["молоко"]["vector"] = vec(0.1, 0.9)
            engine.checkpoint.tokens["далекий"]["vector"] = vec(-1.0)
            engine.checkpoint.tokens["привет"]["vector"] = vec(0.7)
            engine.checkpoint.tokens["ответ"]["vector"] = vec(0.8)
            engine.checkpoint.tokens["дальше"]["vector"] = vec(0.0, 0.1)
            for target in ["самокат", "молоко", "далекий", "привет", "ответ"]:
                engine._add_edge(token_id("__assistant__"), token_id(target), 1.0, report, now)
            engine._add_edge(token_id("ответ"), token_id("."), 1.0, report, now)
            engine._add_edge(token_id("ответ"), token_id("дальше"), 1.0, report, now)

            semantic_scores = engine._generation_candidates(
                token_id("__assistant__"),
                vec(1.0),
                query_tokens=set(),
                generated_tokens=[],
            )
            self.assertGreater(semantic_scores[token_id("самокат")], semantic_scores[token_id("молоко")])
            self.assertNotIn(token_id("далекий"), semantic_scores)

            suppressed_scores = engine._generation_candidates(
                token_id("__assistant__"),
                vec(1.0),
                query_tokens={"привет"},
                generated_tokens=[],
            )
            self.assertNotIn(token_id("привет"), suppressed_scores)

            repeated_scores = engine._generation_candidates(
                token_id("__assistant__"),
                vec(1.0),
                query_tokens=set(),
                generated_tokens=[token_id("ответ")],
            )
            self.assertLess(repeated_scores[token_id("ответ")], semantic_scores[token_id("ответ")])

            terminal_scores = engine._generation_candidates(
                token_id("ответ"),
                vec(0.0),
                query_tokens=set(),
                generated_tokens=[token_id(str(index)) for index in range(6)],
            )
            self.assertGreater(terminal_scores[token_id(".")], terminal_scores[token_id("дальше")])

    def test_generation_candidates_use_raw_weight_cosine_and_reinforcement_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()
            now = 1.0
            report = engine._empty_train_report(session_id="unit", epochs=1)
            for token in ["__assistant__", "самокат", "ответ", "дальше"]:
                engine._ensure_token(token, report, now)
            engine.checkpoint.tokens["самокат"]["vector"] = vec(1.0)
            engine.checkpoint.tokens["ответ"]["vector"] = vec(1.0)
            engine.checkpoint.tokens["дальше"]["vector"] = vec(1.0)
            engine._add_edge(token_id("__assistant__"), token_id("самокат"), 2.0, report, now)
            engine._add_edge(token_id("__assistant__"), token_id("ответ"), 2.0, report, now)
            engine._add_edge(token_id("__assistant__"), token_id("дальше"), 2.0, report, now)
            engine._add_edge(token_id("самокат"), token_id("дальше"), 1.0, report, now)

            scores = engine._generation_candidates(
                token_id("__assistant__"),
                vec(1.0),
                query_tokens=set(),
                generated_tokens=[token_id("ответ")],
                next_milestone=token_id("дальше"),
            )

            self.assertAlmostEqual(scores[token_id("дальше")], 2.0 * 2.0 * 1.3, places=5)
            self.assertAlmostEqual(scores[token_id("самокат")], 2.0 * 2.0 * 1.3, places=5)
            self.assertAlmostEqual(scores[token_id("ответ")], 2.0 * 2.0 * 0.01, places=5)

    def test_generation_candidates_rank_by_weight_times_one_plus_cosine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()
            now = 1.0
            report = engine._empty_train_report(session_id="unit", epochs=1)
            for token in ["__assistant__", "close", "far"]:
                engine._ensure_token(token, report, now)
            engine.checkpoint.tokens["close"]["vector"] = vec(1.0)
            engine.checkpoint.tokens["far"]["vector"] = vec(0.0, 1.0)
            engine._add_edge(token_id("__assistant__"), token_id("close"), 3.0, report, now)
            engine._add_edge(token_id("__assistant__"), token_id("far"), 3.0, report, now)

            scores = engine._generation_candidates(token_id("__assistant__"), vec(1.0), query_tokens=set(), generated_tokens=[])

            self.assertGreater(scores[token_id("close")], scores[token_id("far")])
            self.assertAlmostEqual(scores[token_id("close")], 6.0, places=5)
            self.assertAlmostEqual(scores[token_id("far")], 3.0, places=5)

    def test_outgoing_edges_are_limited_to_top_hundred_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()
            now = 1.0
            report = engine._empty_train_report(session_id="unit", epochs=1)
            engine._ensure_token("source", report, now)
            for index in range(150):
                token = f"cand{index}"
                engine._ensure_token(token, report, now)
                engine._add_edge(token_id("source"), token_id(token), float(index), report, now)

            edges = engine._outgoing_edges(token_id("source"))

            self.assertEqual(len(edges), 100)
            self.assertEqual(edges[0]["target"], token_id("cand149"))
            self.assertEqual(edges[-1]["target"], token_id("cand50"))

    def test_root_synthesis_uses_sqlite_outgoing_edges_without_global_graph_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()
            now = 1.0
            report = engine._empty_train_report(session_id="unit", epochs=1)
            for token in ["__assistant__", "ответ", "дальше"]:
                engine._ensure_token(token, report, now)
            engine.checkpoint.tokens["ответ"]["vector"] = vec(1.0)
            engine.checkpoint.tokens["дальше"]["vector"] = vec(0.0, 1.0)
            engine._add_edge(token_id("__assistant__"), token_id("ответ"), 2.0, report, now)
            engine._add_edge(token_id("ответ"), token_id("дальше"), 1.0, report, now)
            backpack = engine._build_dense_backpack("привет", session_id="unit")

            with mock.patch.object(engine, "_active_graph_edges", side_effect=AssertionError("active graph should not be loaded")), \
                mock.patch.object(engine, "_all_edges", side_effect=AssertionError("global edge scan should not happen")):
                response, source = engine._synthesize_response(
                    ["привет"],
                    backpack,
                    session_id="unit",
                    milestones=[],
                    prompt_tail_token="привет",
                )

            self.assertEqual(source, "graph")
            self.assertTrue(response)

    def test_runtime_caches_rebuild_after_training_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()

            with mock.patch.object(engine, "_rebuild_runtime_caches", wraps=engine._rebuild_runtime_caches) as rebuild_train:
                engine.train_text("Самокат едет быстро.", session_id="unit")

            self.assertGreaterEqual(rebuild_train.call_count, 1)

            result = engine.chat("Привет", session_id="unit")["result"]
            with mock.patch.object(engine, "_rebuild_runtime_caches", wraps=engine._rebuild_runtime_caches) as rebuild_feedback:
                engine.feedback(result_id=result["result_id"], score=1, corrected_response="Отлично")

            self.assertGreaterEqual(rebuild_feedback.call_count, 1)


if __name__ == "__main__":
    unittest.main()
