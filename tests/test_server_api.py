import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path


WEB_DEPS = all(importlib.util.find_spec(name) for name in ["fastapi", "httpx", "pydantic"])


@unittest.skipUnless(WEB_DEPS, 'install web dependencies with: pip install -e ".[web]"')
class ServerApiTest(unittest.TestCase):
    def make_client(self, tmp: str):
        from fastapi.testclient import TestClient

        from semantic_ants.server.app import create_app
        from semantic_ants.server.service import ServerConfig

        app = create_app(ServerConfig(state_dir=Path(tmp), allow_network=False))
        return TestClient(app)

    def test_analyze_api_returns_graph_and_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.make_client(tmp)
            response = client.post(
                "/api/analyze",
                json={"text": "apple", "lang": "en", "strength_vector": [3]},
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["result"]["semantic_vector"])
            self.assertTrue(payload["result"]["signal_trace"])
            self.assertTrue(payload["graph"]["nodes"])
            self.assertTrue(any(edge["signal"]["active"] for edge in payload["graph"]["edges"]))
            self.assertTrue(payload["trace_interpretation"]["active_edge_ids"])

    def test_understand_api_returns_tokens_without_writing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.make_client(tmp)
            checkpoint_path = Path(tmp) / "checkpoints" / "model.json"
            before = checkpoint_path.read_text(encoding="utf-8") if checkpoint_path.exists() else ""

            response = client.post(
                "/api/understand",
                json={
                    "text": "котики едят",
                    "lang": "ru",
                    "session_id": "diag-session",
                    "turn_id": "turn-1",
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["input_text"], "котики едят")
            self.assertEqual(payload["lang"], "ru")
            self.assertEqual(payload["session_id"], "diag-session")
            self.assertEqual(payload["turn_id"], "turn-1")
            self.assertEqual([token["search_token"] for token in payload["tokens"]], ["кот", "есть"])
            self.assertEqual(payload["tokens"][0]["match_status"], "candidate")
            self.assertEqual(payload["tokens"][1]["match_status"], "candidate")

            after = checkpoint_path.read_text(encoding="utf-8") if checkpoint_path.exists() else ""
            self.assertEqual(before, after)
            self.assertEqual(client.get("/api/memory/results").json(), [])
            self.assertEqual(client.get("/api/chat/sessions").json(), [])

    def test_decode_api_returns_sentence_without_writing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.make_client(tmp)
            checkpoint_path = Path(tmp) / "checkpoints" / "model.json"
            before = checkpoint_path.read_text(encoding="utf-8") if checkpoint_path.exists() else ""

            response = client.post(
                "/api/decode",
                json={
                    "text": "кот есть рыба мясо",
                    "tokens": ["кот", "есть", "рыба", "мясо"],
                    "lang": "ru",
                    "session_id": "decode-session",
                    "turn_id": "turn-9",
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["input_text"], "кот есть рыба мясо")
            self.assertEqual(payload["input_tokens"], ["кот", "есть", "рыба", "мясо"])
            self.assertEqual(payload["lang"], "ru")
            self.assertEqual(payload["sentence"], "кот ест рыбу и мясо")
            self.assertEqual(payload["pattern"], "svo")
            self.assertEqual(payload["session_id"], "decode-session")
            self.assertEqual(payload["turn_id"], "turn-9")
            self.assertEqual([token["role"] for token in payload["tokens"]], ["subject", "verb", "object", "object"])
            self.assertEqual(payload["summary"]["objects"], 2)
            self.assertEqual(payload["summary"]["fallbacks"], 0)

            after = checkpoint_path.read_text(encoding="utf-8") if checkpoint_path.exists() else ""
            self.assertEqual(before, after)
            self.assertEqual(client.get("/api/memory/results").json(), [])
            self.assertEqual(client.get("/api/chat/sessions").json(), [])

    def test_graph_and_concept_detail_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.make_client(tmp)
            response = client.get("/api/graph", params={"layer": 0, "limit": 50})
            self.assertEqual(response.status_code, 200)
            graph = response.json()
            self.assertTrue(graph["edges"])
            self.assertTrue(all(edge["layer"] == 0 for edge in graph["edges"]))

            detail = client.get("/api/concepts/detail", params={"uri": "/m/top/object"})
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(detail.json()["node"]["uri"], "/m/top/object")

    def test_feedback_api_changes_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.make_client(tmp)
            analyzed = client.post("/api/analyze", json={"text": "apple", "lang": "en"}).json()
            result_id = analyzed["result"]["result_id"]
            response = client.post("/api/feedback", json={"result_id": result_id, "score": 1})
            self.assertEqual(response.status_code, 200)
            self.assertGreater(response.json()["changed_edges"], 0)

    def test_training_job_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.make_client(tmp)
            jsonl = json.dumps({"text": "apple", "lang": "en", "target_concepts": ["/m/top/object"]})
            response = client.post("/api/training/train", json={"jsonl": jsonl, "epochs": 1})
            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]
            for _ in range(50):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["result"]["examples"], 1)


if __name__ == "__main__":
    unittest.main()
