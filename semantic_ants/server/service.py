from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from semantic_ants.datasets import download_koziev_dialogues_dataset, download_spc_dataset, download_tatoeba_translation_dataset
from semantic_ants.decoding import decode_words
from semantic_ants.engine import EngineConfig, SemanticEngine
from semantic_ants.knowledge import bootstrap_builtin_knowledge
from semantic_ants.learning import ACOTrainer, Checkpoint, FeedbackTrainer, SimpleQATrainer, Trainer
from semantic_ants.resonance import (
    apply_resonance_feedback,
    clean_resonance_checkpoint,
    generate_resonance,
    resonance_areas,
    resonance_memory,
    resonance_planes,
    resonance_session_context,
    seed_resonance,
    train_resonance_form,
    train_resonance_qa,
    train_resonance_sentence,
)
from semantic_ants.server.graph import (
    concept_detail,
    concept_list,
    graph_from_checkpoint,
    graph_snapshot,
    trace_interpretation,
)
from semantic_ants.server.jobs import Job, JobRegistry
from semantic_ants.understanding import understand_text


@dataclass(frozen=True)
class ServerConfig:
    state_dir: Path = Path(".semantic_ants")
    host: str = "127.0.0.1"
    port: int = 8765
    allow_network: bool = True
    autoload_builtin: bool = True
    static_dir: Path | None = None


class EngineService:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.engine = SemanticEngine(
            config=EngineConfig(
                state_dir=config.state_dir,
                allow_network=config.allow_network,
                autoload_builtin=config.autoload_builtin,
            )
        )
        self.jobs = JobRegistry()
        self._lock = RLock()
        self._ensure_resonance_checkpoint()

    def health(self) -> dict[str, Any]:
        return {"ok": True, "service": "semantic_ants", "time": time.time()}

    def app_config(self) -> dict[str, Any]:
        checkpoint = self.engine.checkpoint
        return {
            "state_dir": str(self.config.state_dir),
            "allow_network": self.config.allow_network,
            "autoload_builtin": self.config.autoload_builtin,
            "checkpoint_version": checkpoint.version,
            "resonance_experiment": bool(checkpoint.metadata.get("resonance_experiment")),
            "resonance_seeded": bool(checkpoint.metadata.get("resonance_seeded")),
            "examples_seen": checkpoint.examples_seen,
            "last_result_id": checkpoint.last_result_id,
        }

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            result, graph = self.engine.analyze_with_graph(
                text=str(payload.get("text", "")),
                lang=str(payload.get("lang", "auto")),
                ant_count=_optional_int(payload.get("ants")),
                max_depth=_optional_int(payload.get("depth")),
                top_concepts=_optional_int(payload.get("top_concepts")),
                mode=str(payload.get("mode", "graph")),
                candidates=int(payload.get("candidates") or 3),
                session_id=payload.get("session_id"),
                reset_session=bool(payload.get("reset_session", False)),
                strength_vector=_parse_strength_vector(payload.get("strength_vector")),
            )
            result_dict = result.to_dict()
            return {
                "result": result_dict,
                "graph": graph_snapshot(graph, self.engine.checkpoint, result),
                "trace_interpretation": trace_interpretation(result_dict),
            }

    def understand(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            result = understand_text(
                text=str(payload.get("text", "")),
                lang=str(payload.get("lang", "auto")),
                checkpoint=self.engine.checkpoint,
                session_id=payload.get("session_id"),
                turn_id=payload.get("turn_id"),
            )
            return result.to_dict()

    def decode(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            result = decode_words(
                text=str(payload.get("text", "")),
                tokens=[str(token) for token in payload.get("tokens", [])] if isinstance(payload.get("tokens"), list) else None,
                lang=str(payload.get("lang", "auto")),
                session_id=payload.get("session_id"),
                turn_id=payload.get("turn_id"),
                checkpoint=self.engine.checkpoint,
            )
            return result.to_dict()

    def chat_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        chat_payload = {
            **payload,
            "session_id": payload.get("session_id") or "default",
        }
        return self.analyze(chat_payload)

    def sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": session_id,
                    "turns": list(turns),
                    "turn_count": len(turns),
                    "updated_at": max((float(turn.get("created_at", 0.0)) for turn in turns), default=0.0),
                }
                for session_id, turns in sorted(self.engine.checkpoint.chat_sessions.items())
            ]

    def reset_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self.engine.checkpoint.reset_chat_session(session_id)
            self.engine.store.save(self.engine.checkpoint)
        return {"session_id": session_id, "reset": True}

    def feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        corrected = payload.get("corrected_concepts")
        with self._lock:
            return FeedbackTrainer(self.engine, self.engine.store).apply(
                result_id=payload.get("result_id"),
                score=int(payload.get("score", 0)),
                corrected_concepts=[str(value) for value in corrected] if isinstance(corrected, list) else None,
                corrected_response=payload.get("corrected_response"),
            )

    def interpret_vector(self, payload: dict[str, Any]) -> dict[str, Any]:
        vector = payload.get("semantic_vector", payload)
        with self._lock:
            return {"response": self.engine.interpret_vector(vector)}

    def memory_summary(self) -> dict[str, Any]:
        with self._lock:
            checkpoint = self.engine.checkpoint
            graph = graph_from_checkpoint(checkpoint)
            return {
                "version": checkpoint.version,
                "examples_seen": checkpoint.examples_seen,
                "last_result_id": checkpoint.last_result_id,
                "canonical_concepts": len(checkpoint.canonical_concepts),
                "concept_redirects": len(checkpoint.concept_redirects),
                "pheromones": len(checkpoint.pheromones),
                "concept_pheromones": len(checkpoint.concept_pheromones),
                "concept_layer_pheromones": len(checkpoint.concept_layer_pheromones),
                "edge_layer_pheromones": len(checkpoint.edge_layer_pheromones),
                "suppressed_concepts": len(checkpoint.suppressed_concepts),
                "custom_edges": len(checkpoint.custom_edges),
                "learned_bridges": len(checkpoint.learned_bridges),
                "accepted_answers": len(checkpoint.accepted_answers),
                "negative_memory": len(checkpoint.negative_memory),
                "response_memory": len(checkpoint.response_memory),
                "chat_sessions": len(checkpoint.chat_sessions),
                "results": len(checkpoint.results),
                "graph": {"nodes": len(graph.nodes), "edges": len(graph.edges())},
            }

    def results(self) -> list[dict[str, Any]]:
        with self._lock:
            values = list(self.engine.checkpoint.results.values())
        values.sort(key=lambda item: str(item.get("result_id", "")), reverse=True)
        return values

    def concepts(self, query: str | None = None, layer: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            return concept_list(self.engine.checkpoint, query=query, layer=layer, limit=limit)

    def concept_detail(self, uri: str, result_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            result = self._result_by_id(result_id)
            return concept_detail(self.engine.checkpoint, uri=uri, result=result)

    def graph(self, filters: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            checkpoint = self.engine.checkpoint
            result = self._result_by_id(filters.get("result_id"))
            graph = graph_from_checkpoint(checkpoint)
            return graph_snapshot(
                graph,
                checkpoint,
                result,
                layer=_optional_int(filters.get("layer")),
                source=filters.get("source"),
                edge_type=filters.get("edge_type"),
                relation=filters.get("relation"),
                query=filters.get("query"),
                plane_id=filters.get("plane_id"),
                area_id=filters.get("area_id"),
                min_pheromone=_optional_float(filters.get("min_pheromone")),
                only_signal=bool(filters.get("only_signal", False)),
                limit=_optional_int(filters.get("limit")),
            )

    def memory_collections(self) -> dict[str, Any]:
        with self._lock:
            checkpoint = self.engine.checkpoint
            return {
                "accepted_answers": checkpoint.accepted_answers,
                "negative_memory": checkpoint.negative_memory,
                "response_memory": checkpoint.response_memory,
                "experiences": checkpoint.experiences[-200:],
                "aliases": checkpoint.aliases,
                "canonical_concepts": checkpoint.canonical_concepts,
                "concept_redirects": checkpoint.concept_redirects,
                "surface_forms": checkpoint.surface_forms,
                "suppressed_concepts": checkpoint.suppressed_concepts,
                "concept_layer_pheromones": checkpoint.concept_layer_pheromones,
                "edge_layer_pheromones": checkpoint.edge_layer_pheromones,
            }

    def submit_train(self, payload: dict[str, Any]) -> Job:
        return self._submit("train", self._train, payload)

    def submit_learn(self, payload: dict[str, Any]) -> Job:
        return self._submit("learn", self._learn, payload)

    def submit_learn_dialogues(self, payload: dict[str, Any]) -> Job:
        return self._submit("learn-dialogues", self._learn_dialogues, payload)

    def submit_simple_train(self, payload: dict[str, Any]) -> Job:
        return self._submit("simple-train", self._simple_train, payload)

    def submit_eval(self, payload: dict[str, Any]) -> Job:
        return self._submit("eval", self._eval, payload)

    def submit_dream(self, payload: dict[str, Any]) -> Job:
        return self._submit("dream", self._dream, payload)

    def submit_bootstrap(self, payload: dict[str, Any]) -> Job:
        return self._submit("bootstrap", self._bootstrap, payload)

    def submit_reset_network(self, payload: dict[str, Any]) -> Job:
        return self._submit("reset-network", self._reset_network, payload)

    def submit_download_spc(self, payload: dict[str, Any]) -> Job:
        return self._submit("download-spc", self._download_spc, payload)

    def submit_download_koziev(self, payload: dict[str, Any]) -> Job:
        return self._submit("download-koziev", self._download_koziev, payload)

    def submit_download_tatoeba(self, payload: dict[str, Any]) -> Job:
        return self._submit("download-tatoeba", self._download_tatoeba, payload)

    def submit_export(self, payload: dict[str, Any]) -> Job:
        return self._submit("export", self._export, payload)

    def submit_resonance_reset(self, payload: dict[str, Any]) -> Job:
        return self._submit("resonance-reset", self._resonance_reset, payload)

    def submit_resonance_seed(self, payload: dict[str, Any]) -> Job:
        return self._submit("resonance-seed", self._resonance_seed, payload)

    def submit_resonance_train_form(self, payload: dict[str, Any]) -> Job:
        return self._submit("resonance-train-form", self._resonance_train_form, payload)

    def submit_resonance_train_sentence(self, payload: dict[str, Any]) -> Job:
        return self._submit("resonance-train-sentence", self._resonance_train_sentence, payload)

    def submit_resonance_train_qa(self, payload: dict[str, Any]) -> Job:
        return self._submit("resonance-train-qa", self._resonance_train_qa, payload)

    def resonance_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            result = generate_resonance(self.engine.checkpoint, payload)
            self.engine.store.save(self.engine.checkpoint)
            graph = graph_from_checkpoint(self.engine.checkpoint)
            return {
                "result": result,
                "graph": graph_snapshot(graph, self.engine.checkpoint, result),
                "trace_interpretation": trace_interpretation(result),
            }

    def resonance_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            result = apply_resonance_feedback(self.engine.checkpoint, payload)
            self.engine.store.save(self.engine.checkpoint)
            return result

    def resonance_memory(self) -> dict[str, Any]:
        with self._lock:
            return resonance_memory(self.engine.checkpoint)

    def resonance_planes(self) -> list[dict[str, Any]]:
        with self._lock:
            return resonance_planes(self.engine.checkpoint)

    def resonance_areas(self) -> list[dict[str, Any]]:
        with self._lock:
            return resonance_areas(self.engine.checkpoint)

    def resonance_session_context(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return resonance_session_context(self.engine.checkpoint, session_id)

    def job(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        return job.to_dict() if job else None

    def job_list(self) -> list[dict[str, Any]]:
        return [job.to_dict() for job in self.jobs.list()]

    def _submit(self, name: str, fn: Callable[[dict[str, Any]], Any], payload: dict[str, Any]) -> Job:
        return self.jobs.submit(name, self._locked_call, fn, payload)

    def _locked_call(self, fn: Callable[[dict[str, Any]], Any], payload: dict[str, Any]) -> Any:
        with self._lock:
            return fn(payload)

    def _train(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._payload_jsonl_path(payload, "train")
        report = Trainer(self.engine, self.engine.store).train_file(path, epochs=int(payload.get("epochs") or 1))
        return report.to_dict()

    def _learn(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._payload_jsonl_path(payload, "learn")
        report = ACOTrainer(self.engine, self.engine.store).learn_file(path, epochs=int(payload.get("epochs") or 1))
        return report.to_dict()

    def _learn_dialogues(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._payload_jsonl_path(payload, "learn_dialogues")
        report = ACOTrainer(self.engine, self.engine.store).learn_dialogue_file(
            path,
            epochs=int(payload.get("epochs") or 1),
            batch_size=int(payload.get("batch_size") or 32),
            max_examples=_optional_int(payload.get("max_examples")),
            torch_steps=int(payload.get("torch_steps") or 1),
        )
        return report.to_dict()

    def _simple_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        report = SimpleQATrainer(self.engine, self.engine.store).train_payload(payload)
        return report.to_dict()

    def _eval(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._payload_jsonl_path(payload, "eval")
        total = 0
        hits = 0
        rows = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                example = json.loads(stripped)
                total += 1
                result = self.engine.analyze(
                    str(example["text"]),
                    lang=str(example.get("lang", "auto")),
                    strength_vector=_parse_strength_vector(example.get("strength_vector")),
                )
                expected = set(map(str, example.get("target_concepts", [])))
                actual = {item["uri"] for item in result.activated_concepts}
                ok = bool(expected & actual) if expected else True
                hits += int(ok)
                rows.append({"text": example["text"], "ok": ok, "expected": sorted(expected), "actual": sorted(actual)})
        return {"total": total, "hits": hits, "accuracy": hits / total if total else 0.0, "rows": rows}

    def _dream(self, payload: dict[str, Any]) -> dict[str, Any]:
        return ACOTrainer(self.engine, self.engine.store).dream(steps=int(payload.get("steps") or 100))

    def _bootstrap(self, payload: dict[str, Any]) -> dict[str, Any]:
        report = bootstrap_builtin_knowledge(self.engine.checkpoint, force=bool(payload.get("force", False)))
        self.engine.store.save(self.engine.checkpoint)
        return report.to_dict()

    def _reset_network(self, payload: dict[str, Any]) -> dict[str, Any]:
        keep_builtin = bool(payload.get("keep_builtin", True))
        old_checkpoint = self.engine.checkpoint
        self.engine.checkpoint = Checkpoint(seed=old_checkpoint.seed)
        bootstrap_report = None
        if keep_builtin:
            bootstrap_report = bootstrap_builtin_knowledge(self.engine.checkpoint, force=True).to_dict()
        model_path = self.engine.speech.model_path(self.engine.model_dir)
        removed_model = False
        if model_path is not None and model_path.exists():
            model_path.unlink()
            removed_model = True
        self.engine.store.save(self.engine.checkpoint)
        return {
            "reset": True,
            "keep_builtin": keep_builtin,
            "removed_dialogue_model": removed_model,
            "examples_seen": self.engine.checkpoint.examples_seen,
            "custom_edges": len(self.engine.checkpoint.custom_edges),
            "accepted_answers": len(self.engine.checkpoint.accepted_answers),
            "bootstrap": bootstrap_report,
        }

    def _download_spc(self, payload: dict[str, Any]) -> dict[str, Any]:
        output = Path(str(payload.get("output") or self.config.state_dir / "datasets" / "spc_dialogues.jsonl"))
        count = download_spc_dataset(
            split=str(payload.get("split") or "train"),
            output=output,
            limit=_optional_int(payload.get("limit")),
        )
        return {"dataset": "spc", "split": payload.get("split") or "train", "output": str(output), "examples": count}

    def _download_koziev(self, payload: dict[str, Any]) -> dict[str, Any]:
        output = Path(str(payload.get("output") or self.config.state_dir / "datasets" / "koziev_dialogues.jsonl"))
        count = download_koziev_dialogues_dataset(
            output=output,
            source=payload.get("path"),
            limit=_optional_int(payload.get("limit")),
            timeout=float(payload.get("timeout") or 60.0),
        )
        return {
            "dataset": "koziev",
            "path": payload.get("path"),
            "output": str(output),
            "examples": count,
        }

    def _download_tatoeba(self, payload: dict[str, Any]) -> dict[str, Any]:
        output = Path(str(payload.get("output") or self.config.state_dir / "datasets" / "tatoeba_translation.jsonl"))
        count = download_tatoeba_translation_dataset(
            output=output,
            source_lang=str(payload.get("source_lang") or "ru"),
            target_lang=str(payload.get("target_lang") or "en"),
            limit=_optional_int(payload.get("limit")),
            bidirectional=bool(payload.get("bidirectional", True)),
            timeout=float(payload.get("timeout") or 60.0),
        )
        return {
            "dataset": "tatoeba",
            "source_lang": payload.get("source_lang") or "ru",
            "target_lang": payload.get("target_lang") or "en",
            "bidirectional": bool(payload.get("bidirectional", True)),
            "output": str(output),
            "examples": count,
        }

    def _export(self, payload: dict[str, Any]) -> dict[str, Any]:
        destination = payload.get("destination")
        if not destination:
            raise ValueError("destination is required")
        self.engine.store.export(Path(str(destination)))
        return {"destination": str(destination)}

    def _ensure_resonance_checkpoint(self) -> None:
        checkpoint = self.engine.checkpoint
        if checkpoint.version == 7 and checkpoint.metadata.get("resonance_experiment"):
            return
        self.engine.checkpoint = clean_resonance_checkpoint(seed=getattr(checkpoint, "seed", 42))
        self.engine.store.save(self.engine.checkpoint)

    def _resonance_reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        seed = getattr(self.engine.checkpoint, "seed", 42)
        self.engine.checkpoint = clean_resonance_checkpoint(seed=seed)
        self._remove_experiment_state_dirs()
        if bool(payload.get("seed", False)):
            seed_resonance(self.engine.checkpoint, force=True, session_id=str(payload.get("session_id") or "default"))
        self.engine.store.save(self.engine.checkpoint)
        return resonance_memory(self.engine.checkpoint)

    def _resonance_seed(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = seed_resonance(
            self.engine.checkpoint,
            force=bool(payload.get("force", False)),
            session_id=str(payload.get("session_id") or "default"),
        )
        self.engine.store.save(self.engine.checkpoint)
        return result

    def _resonance_train_form(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = train_resonance_form(self.engine.checkpoint, payload)
        self.engine.store.save(self.engine.checkpoint)
        return result

    def _resonance_train_sentence(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = train_resonance_sentence(self.engine.checkpoint, payload)
        self.engine.store.save(self.engine.checkpoint)
        return result

    def _resonance_train_qa(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = train_resonance_qa(self.engine.checkpoint, payload)
        self.engine.store.save(self.engine.checkpoint)
        return result

    def _remove_experiment_state_dirs(self) -> None:
        for folder in (self.engine.model_dir, self.config.state_dir / "web_inputs"):
            try:
                resolved = folder.resolve()
                state_root = self.config.state_dir.resolve()
            except OSError:
                continue
            if resolved == state_root or state_root not in resolved.parents:
                continue
            if resolved.exists():
                shutil.rmtree(resolved)

    def _payload_jsonl_path(self, payload: dict[str, Any], prefix: str) -> Path:
        path = payload.get("path")
        if path:
            return Path(str(path))
        jsonl = str(payload.get("jsonl") or "").strip()
        if not jsonl:
            raise ValueError("path or jsonl is required")
        folder = self.config.state_dir / "web_inputs"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{prefix}-{time.time_ns()}.jsonl"
        target.write_text(jsonl + "\n", encoding="utf-8")
        return target

    def _result_by_id(self, result_id: str | None) -> dict[str, Any] | None:
        if not result_id:
            result_id = self.engine.checkpoint.last_result_id
        if not result_id:
            return None
        return self.engine.checkpoint.results.get(str(result_id))


def _parse_strength_vector(value: Any) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return (max(value, 0),)
    if isinstance(value, str):
        return tuple(max(int(part.strip()), 0) for part in value.replace(";", ",").split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(max(int(part), 0) for part in value)
    raise ValueError(f"Invalid strength_vector: {value!r}")


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
