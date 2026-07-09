from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from semantic_ants.engine import DEFAULT_VECTOR_DIM, EngineConfig, SemanticEngine
from semantic_ants.preprocess import preprocess_dataset


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


class FakeEmbeddingBackend:
    dim = DEFAULT_VECTOR_DIM

    def ensure(self) -> bool:
        return True

    def encode(self, text: str) -> list[float]:
        values = [0.0] * DEFAULT_VECTOR_DIM
        values[0] = float(len(text.split())) or 1.0
        values[1] = float(sum(ord(char) for char in text) % 17)
        return values

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(text) for text in texts]


class HierarchyBackpackTests(unittest.TestCase):
    def test_preprocess_directory_emits_hierarchy_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output_path = Path(tmp) / "out.jsonl"
            write_text(root / "src" / "module.py", "print('hello')\n")
            write_text(root / "docs" / "guide.md", "# Intro\nBody text here.\n\n## Details\nMore text.\n")
            write_text(root / "node_modules" / "pkg" / "index.js", "console.log('skip me')\n")

            stats = preprocess_dataset(root, output_path)
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(stats["accepted"], len(records))
            self.assertTrue(records)
            self.assertTrue(all("hierarchy" in record and "text" in record for record in records))
            self.assertTrue(any(record["hierarchy"][:2] == ["src", "module"] for record in records))
            self.assertTrue(any(record["hierarchy"][:3] == ["docs", "guide", "Intro"] for record in records))
            self.assertFalse(any("node_modules" in record["hierarchy"] for record in records))

    def test_hierarchy_training_builds_hypernodes_and_edge_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            input_path = Path(tmp) / "hierarchy.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "hierarchy": ["docs", "intro"],
                        "text": "Привет как дела сегодня. Следующий шаг.",
                    }
                ],
            )

            engine = SemanticEngine(config=EngineConfig(state_dir=state_dir))
            engine._embedding_backend = FakeEmbeddingBackend()
            report = engine.train_jsonl(input_path, session_id="unit", epochs=1)

            node_id = engine._hierarchy_node_id(["docs", "intro"])
            hypernodes = engine.checkpoint.meta["hypernodes"]
            self.assertGreaterEqual(report["dataset_records"], 1)
            self.assertIn(node_id, hypernodes)
            self.assertIn("vector", hypernodes[node_id])
            self.assertIn("subgraph", hypernodes[node_id])
            leaf_graph = engine._subgraph_to_graph_payload(node_id, hypernodes[node_id]["subgraph"], label=hypernodes[node_id]["label"])
            self.assertTrue(leaf_graph["edges"])
            self.assertTrue(all(edge["relation"] == "next" for edge in leaf_graph["edges"]))
            self.assertTrue(all(edge["type"] == "transition_edge" for edge in leaf_graph["edges"]))

            parent_id = engine._hierarchy_node_id(["docs"])
            parent_graph = engine._build_hypernode_graph(parent_id, limit=24, highlight_result_id=None)
            self.assertTrue(any(node["id"] == node_id and node["type"] == "hypernode" for node in parent_graph["nodes"]))
            self.assertTrue(any(edge["source"] == parent_id and edge["target"] == node_id for edge in parent_graph["edges"]))
            self.assertTrue(any(edge["relation"] == "hierarchical_edge" for edge in parent_graph["edges"]))

    def test_stack_depth_and_backpack_schema_follow_drill_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            input_path = Path(tmp) / "hierarchy.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "hierarchy": ["docs", "intro"],
                        "text": "Привет как дела сегодня. Следующий шаг.",
                    }
                ],
            )

            engine = SemanticEngine(config=EngineConfig(state_dir=state_dir))
            engine._embedding_backend = FakeEmbeddingBackend()
            engine.train_jsonl(input_path, session_id="unit", epochs=1)

            node_id = engine._hierarchy_node_id(["docs", "intro"])
            drill_down = engine.drill_down(node_id, session_id="unit")
            response = engine.chat("Привет как дела", session_id="unit")
            drill_up = engine.drill_up(session_id="unit")
            reset = engine.reset_session("unit")

            backpack = response["backpack"]
            graph_data = backpack["graph_data"]
            self.assertEqual(drill_down["current_depth"], 1)
            self.assertEqual(backpack["current_depth"], 1)
            self.assertEqual(backpack["active_focus_label"], "intro")
            self.assertGreaterEqual(backpack["total_depth_layers"], 1)
            self.assertIn("nodes", graph_data)
            self.assertIn("edges", graph_data)
            self.assertTrue(graph_data["nodes"])
            self.assertTrue(graph_data["edges"])
            self.assertTrue(all(edge["relation"] == "next" for edge in graph_data["edges"]))
            self.assertEqual(drill_up["current_depth"], 0)
            self.assertTrue(reset["reset"])
            self.assertEqual(engine._stack_depth("unit"), 0)

    def test_parent_hypernode_focus_generates_child_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            input_path = Path(tmp) / "hierarchy.jsonl"
            write_jsonl(
                input_path,
                [
                    {"hierarchy": ["docs", "intro"], "text": "Первый раздел описывает вход."},
                    {"hierarchy": ["docs", "details"], "text": "Второй раздел описывает детали."},
                ],
            )

            engine = SemanticEngine(config=EngineConfig(state_dir=state_dir))
            engine._embedding_backend = FakeEmbeddingBackend()
            engine.train_jsonl(input_path, session_id="unit", epochs=1)

            parent_id = engine._hierarchy_node_id(["docs"])
            engine.drill_down(parent_id, session_id="unit")
            response = engine.chat("что внутри", session_id="unit")

            self.assertEqual(response["result"]["response_source"], "hypernode_plan")
            self.assertIn("intro", response["result"]["response"])
            self.assertIn("details", response["result"]["response"])


if __name__ == "__main__":
    unittest.main()
