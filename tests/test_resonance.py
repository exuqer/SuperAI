import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_ants.resonance import effective_distance
from semantic_ants.server.graph import graph_from_checkpoint, graph_snapshot
from semantic_ants.server.service import EngineService, ServerConfig


class ResonanceExperimentTest(unittest.TestCase):
    def make_service(self, tmp: str) -> EngineService:
        return EngineService(ServerConfig(state_dir=Path(tmp), allow_network=False))

    def test_reset_starts_from_clean_v7_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            checkpoint = service.engine.checkpoint
            checkpoint.custom_edges.append({"start": "a", "relation": "r", "end": "b"})
            checkpoint.learned_bridges.append({"start": "a", "relation": "r", "end": "c"})
            checkpoint.accepted_answers.append({"answer": "old"})

            report = service._resonance_reset({"seed": False})

            self.assertEqual(report["version"], 7)
            self.assertTrue(report["resonance_experiment"])
            self.assertEqual(report["custom_edges"], 0)
            self.assertEqual(report["accepted_answers"], 0)
            self.assertEqual(service.engine.checkpoint.learned_bridges, [])

    def test_seed_and_generate_tree_plural(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})

            response = service.resonance_generate(
                {"text": "дерево subject plur", "lang": "ru", "session_id": "default"}
            )

            self.assertEqual(response["result"]["response"], "деревья")
            self.assertEqual(response["result"]["semantic_vector"]["active_plane"], "language:ru")
            self.assertGreater(response["graph"]["stats"]["edges"], 0)

    def test_seed_does_not_duplicate_base_form_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})
            snapshot = graph_snapshot(graph_from_checkpoint(service.engine.checkpoint), service.engine.checkpoint)
            tree_nodes = [node for node in snapshot["nodes"] if node["label"] == "дерево"]

            self.assertEqual(len(tree_nodes), 1)
            self.assertEqual(tree_nodes[0]["uri"], "/m/concept/дерево")

    def test_training_existing_form_reinforces_without_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})
            before_forms = len(service.engine.checkpoint.morph_forms["ru:дерево"])
            before_pheromone = next(
                form["pheromone"]
                for form in service.engine.checkpoint.morph_forms["ru:дерево"]
                if form["surface"] == "деревья"
            )

            service._resonance_train_form(
                {
                    "lang": "ru",
                    "lemma": "дерево",
                    "surface": "деревья",
                    "pos": "NOUN",
                    "role": "subject",
                    "gram": {"case": "nomn", "number": "plur", "gender": "neut"},
                    "reward": 2.0,
                }
            )

            forms = service.engine.checkpoint.morph_forms["ru:дерево"]
            after_pheromone = next(form["pheromone"] for form in forms if form["surface"] == "деревья")
            self.assertEqual(len(forms), before_forms)
            self.assertGreater(after_pheromone, before_pheromone)

    def test_question_answer_training_splits_answer_and_reinforces_existing_forms(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})
            before_forms = sum(len(values) for values in service.engine.checkpoint.morph_forms.values())
            before_pheromone = next(
                form["pheromone"]
                for form in service.engine.checkpoint.morph_forms["ru:писать"]
                if form["surface"] == "пишет"
            )

            report = service._resonance_train_qa(
                {
                    "question": "что делает программист?",
                    "expected_answer": "Программист пишет код на компьютере.",
                    "lang": "ru",
                    "epochs": 2,
                    "reward": 1.5,
                }
            )

            after_forms = sum(len(values) for values in service.engine.checkpoint.morph_forms.values())
            after_pheromone = next(
                form["pheromone"]
                for form in service.engine.checkpoint.morph_forms["ru:писать"]
                if form["surface"] == "пишет"
            )
            generated = service.resonance_generate({"text": "что делает программист?", "lang": "ru", "session_id": "default"})
            self.assertEqual(report["mode"], "question_answer")
            self.assertIn("программист", report["lemmas"])
            self.assertIn("писать", report["lemmas"])
            self.assertIn("код", report["lemmas"])
            self.assertIn("компьютер", report["lemmas"])
            self.assertEqual(after_forms, before_forms)
            self.assertGreater(after_pheromone, before_pheromone)
            self.assertEqual(generated["result"]["response"], "Программист пишет код на компьютере.")
            service.resonance_feedback({"result_id": generated["result"]["result_id"], "score": 1, "session_id": "default"})
            rerendered = service.resonance_generate(
                {"text": "что делает программист?", "lang": "ru", "session_id": "default"}
            )
            self.assertNotEqual(rerendered["result"]["response"], "Программист пишет код на компьютере.")
            snapshot = graph_snapshot(graph_from_checkpoint(service.engine.checkpoint), service.engine.checkpoint)
            programmer = next(node for node in snapshot["nodes"] if node["label"] == "программист")
            area_ids = programmer["metadata"].get("area_ids", [])
            self.assertTrue(any(str(area).startswith("area:language:ru/") for area in area_ids))
            self.assertFalse(any("tree_forms" in str(area) for area in area_ids))

    def test_question_answer_training_blends_multiple_answers_for_same_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})
            answers = [
                "Муж купает ребёнка, а он ест пену.",
                "Кот сидит на клавиатуре и пишет код.",
                "Сосед обещал похудеть, но весы его узнали.",
            ]

            for answer in answers:
                service._resonance_train_qa(
                    {
                        "question": "расскажи анекдот",
                        "expected_answer": answer,
                        "lang": "ru",
                        "reward": 1.0,
                    }
                )

            responses = [
                service.resonance_generate(
                    {
                        "text": "расскажи анекдот",
                        "lang": "ru",
                        "session_id": "jokes",
                        "creativity": 1.0,
                    }
                )["result"]["response"]
                for _ in range(3)
            ]

            self.assertGreater(len(set(responses)), 1)
            self.assertTrue(any(response not in answers for response in responses))

    def test_question_answer_training_variants_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})
            answers = [
                "Муж купает ребёнка, а он ест пену.",
                "Кот сидит на клавиатуре и пишет код.",
                "Сосед обещал похудеть, но весы его узнали.",
            ]

            for answer in answers:
                service._resonance_train_qa(
                    {
                        "question": "расскажи анекдот",
                        "expected_answer": answer,
                        "lang": "ru",
                        "reward": 1.0,
                    }
                )

            restarted = self.make_service(tmp)
            responses = [
                restarted.resonance_generate(
                    {
                        "text": "расскажи анекдот",
                        "lang": "ru",
                        "session_id": "jokes",
                        "creativity": 1.0,
                    }
                )["result"]["response"]
                for _ in range(4)
            ]

            self.assertGreater(len(set(responses)), 1)
            self.assertTrue(
                any(
                    len(item.get("responses", [])) >= 3
                    for item in restarted.engine.checkpoint.response_memory.values()
                    if isinstance(item, dict)
                )
            )

    def test_question_answer_training_creates_new_plane_for_new_concept(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})

            report = service._resonance_train_qa(
                {
                    "question": "что такое квазидрево?",
                    "expected_answer": "Квазидрево растет.",
                    "lang": "ru",
                    "reward": 1.0,
                }
            )

            self.assertTrue(any(plane.startswith("semantic:learned:ru:квазидрево") for plane in report["created_planes"]))
            self.assertTrue(any(plane.startswith("semantic:learned:ru:квазидрево") for plane in service.engine.checkpoint.planes))

    def test_unrelated_prompt_does_not_reuse_previous_answer_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})
            target = "Программист пишет код на компьютере."

            service._resonance_train_qa(
                {
                    "question": "что делает программист?",
                    "expected_answer": target,
                    "lang": "ru",
                    "reward": 1.5,
                }
            )

            response = service.resonance_generate(
                {
                    "text": "Дети",
                    "lang": "ru",
                    "session_id": "default",
                    "creativity": 1.0,
                }
            )

            self.assertNotEqual(response["result"]["response"], target)

    def test_graph_limit_zero_returns_all_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})

            limited = service.graph({"limit": 5})
            unlimited = service.graph({"limit": 0})

            self.assertLessEqual(limited["stats"]["nodes"], unlimited["stats"]["nodes"])
            self.assertLessEqual(limited["stats"]["edges"], unlimited["stats"]["edges"])
            self.assertGreater(unlimited["stats"]["nodes"], 0)

    def test_question_answer_training_respects_annotation_planes(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})

            report = service._resonance_train_qa(
                {
                    "question": "что такое дерево файлов?",
                    "expected_answer": "Дерево хранит файлы.",
                    "lang": "ru",
                    "annotations": [
                        {
                            "index": 0,
                            "token": "дерево",
                            "lemma": "дерево",
                            "role": "subject",
                            "concept": "/c/ru/дерево",
                            "planes": ["dev:filesystem"],
                            "gram": {"case": "nomn", "number": "sing"},
                        }
                    ],
                    "reward": 2.0,
                }
            )

            checkpoint = service.engine.checkpoint
            tree = checkpoint.canonical_uri("/c/ru/дерево")
            role = checkpoint.canonical_uri("/role/ru/subject")
            self.assertIn("dev:filesystem", report["planes"])
            self.assertLess(effective_distance(checkpoint, tree, role, "dev:filesystem"), 9.0)

    def test_plane_distances_are_contextual(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})
            checkpoint = service.engine.checkpoint
            tree = checkpoint.canonical_uri("/c/ru/дерево")
            plant = checkpoint.canonical_uri("/c/ru/растение")
            lamp = checkpoint.canonical_uri("/c/ru/лампочка")
            file = checkpoint.canonical_uri("/c/ru/файл")

            self.assertLess(
                effective_distance(checkpoint, tree, plant, "semantic:nature"),
                effective_distance(checkpoint, tree, lamp, "semantic:nature"),
            )
            self.assertLess(
                effective_distance(checkpoint, tree, file, "dev:filesystem"),
                effective_distance(checkpoint, tree, plant, "dev:filesystem"),
            )

    def test_chat_context_reuses_previous_plane(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})

            first = service.resonance_generate({"text": "дерево файл папка", "lang": "ru", "session_id": "s"})
            second = service.resonance_generate({"text": "дерево", "lang": "ru", "session_id": "s"})

            self.assertEqual(first["result"]["semantic_vector"]["active_plane"], "dev:filesystem")
            self.assertEqual(second["result"]["semantic_vector"]["active_plane"], "dev:filesystem")

    def test_unknown_word_is_committed_by_positive_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})
            response = service.resonance_generate({"text": "глум subject plur", "lang": "ru", "session_id": "s"})
            result = response["result"]

            self.assertEqual(result["response"], "глумы")
            self.assertTrue(result["semantic_vector"]["tentative_forms"])

            service.resonance_feedback({"result_id": result["result_id"], "score": 5, "session_id": "s"})
            forms = service.engine.checkpoint.morph_forms["ru:глум"]
            self.assertTrue(any(form["surface"] == "глумы" and not form["tentative"] for form in forms))

    def test_runtime_does_not_call_understand_decode_or_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.make_service(tmp)
            service._resonance_seed({"force": True})
            with patch("semantic_ants.understanding.understand_text", side_effect=AssertionError), patch(
                "semantic_ants.decoding.decode_words",
                side_effect=AssertionError,
            ), patch(
                "semantic_ants.generation.torch_dialogue.TorchDialogueNavigator.generate",
                side_effect=AssertionError,
            ):
                response = service.resonance_generate({"text": "дерево subject plur", "lang": "ru"})

            self.assertEqual(response["result"]["response"], "деревья")


if __name__ == "__main__":
    unittest.main()
