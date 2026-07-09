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

    def test_chat_backpack_includes_recursive_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SemanticEngine(config=EngineConfig(state_dir=Path(tmp) / "state"))
            engine._embedding_backend = FakeEmbeddingBackend()
            engine.train_text("Привет! Расскажи анекдот -- Блин! - сказал слон, наступив на колобка.", session_id="unit")

            payload = engine.chat("Привет! Расскажи анекдот", session_id="unit")
            backpack = payload["backpack"]
            layers = backpack["layers"]

            self.assertGreaterEqual(len(layers), 2)
            self.assertTrue(all("nodes" in layer and "edges" in layer for layer in layers))
            self.assertTrue(all(layer["nodes"] for layer in layers))
            self.assertNotEqual(layers[0]["focus_ids"], layers[1]["focus_ids"])

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


if __name__ == "__main__":
    unittest.main()
