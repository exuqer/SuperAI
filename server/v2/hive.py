"""Public V2 hive facade."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .local_memory import V2LocalMemoryService, HiveLocalMemoryConfig
from .repository import V2Repository
from .vibration import HiveVibrationEngine, QueryActivation, VibrationConfig
from .export import HiveExportService
from .hive_snapshot import HiveSnapshotProjector
from .analytics import HiveAnalyticsService
from .query_scene import QuerySceneService
from .unknown_search import UnknownTokenSearchService
from .morphology import MorphologyService
from .dynamics import HiveDynamicsService
from .resonance import InputIntentClassifier, LexicalCandidateResolver, PROBE_INTENTS
from .hive_resonance import HiveResonanceEngine
from server.analytics import MultilevelTraceAnalytics
from server.hive.hive_dispatcher import MultilevelHiveService
from server.visualization import VisualizationSuite


class V2HiveService:
    def __init__(self, repository: Optional[V2Repository] = None, config: Optional[HiveLocalMemoryConfig] = None) -> None:
        self.service = V2LocalMemoryService(repository, config)
        self.query_scenes = QuerySceneService(self.service.repository)
        self.unknown_searches = UnknownTokenSearchService(self.service.repository)
        self.dynamics = HiveDynamicsService(self.service.repository)
        self.intent_classifier = InputIntentClassifier(self.service.repository)
        self.lexical = LexicalCandidateResolver(self.service.repository)
        self.resonance = self.lexical
        self.hive_resonance = HiveResonanceEngine(self.service.repository)
        self.multilevel = MultilevelHiveService(self.service.repository)

    def create(self, max_cells: int = 24, conversation_id: str = "") -> Dict[str, Any]:
        return self.service.create_hive(max_cells, conversation_id)

    def preview(self, hive_id: str, text: str) -> Dict[str, Any]:
        return self.service.preview(hive_id, text)

    def query(
        self,
        hive_id: str,
        text: str,
        resolved_mode: Optional[str] = None,
        resonance_scope: str = "LOCAL_THEN_GLOBAL",
    ) -> Dict[str, Any]:
        intent = self.intent_classifier.classify(text)
        if intent in PROBE_INTENTS:
            scope = resonance_scope if resonance_scope in {"LOCAL_ONLY", "LOCAL_THEN_GLOBAL", "GLOBAL_ONLY"} else "LOCAL_THEN_GLOBAL"
            probe = self.lexical.create(hive_id, text, scope)
            self.lexical.run(hive_id, probe["id"])
            state = self.lexical.state(hive_id)
            result = {
                "message_id": probe["message_id"], "intent": intent, "resolved_mode": intent,
                "decision": {"decision": "LEXICAL_CANDIDATE_PROBE", "external_search_required": False, "reasons": ["input classified before scene parsing"], "matches": []},
                "metrics": {"external_search": False, "local_resonance": True, "working_cells": state["stats"]["working_cells"]},
                "external_search": {"sources": [], "bees": [], "iterations": 0, "anchors": []}, "merge_results": [], "resonance_events": [],
                **state,
            }
            return self._attach_multilevel(hive_id, text, result)
        mode = self.query_scenes.resolve_mode(hive_id, text, resolved_mode)
        if mode == "LOCAL_RESONANCE":
            allow_global = resonance_scope == "LOCAL_THEN_GLOBAL"
            working_hive = self.query_scenes.local_resonance(hive_id, text, allow_global=allow_global)
            working_hive["dynamics"] = self.dynamics.get(hive_id)
            result = {
                "message_id": working_hive.get("query_session", {}).get("message_id", ""),
                "resolved_mode": mode,
                "decision": {"decision": "LOCAL_RESONANCE", "external_search_required": False, "reasons": ["single-token probe routed to active hive"], "matches": []},
                "metrics": {"external_search": False, "local_resonance": True},
                "external_search": {"sources": [], "bees": [], "iterations": 0, "anchors": []},
                "merge_results": [], "resonance_events": [], "cells": working_hive.get("cells", []),
                "hive": working_hive.get("hive", {}), "local_resonance": working_hive.get("local_resonance"),
                "resonance_probes": working_hive.get("resonance_probes", []),
                "resonance_scope": "LOCAL_THEN_GLOBAL" if allow_global else "LOCAL_ONLY",
                    **{key: working_hive.get(key) for key in ("query_frame", "query_scene", "memory_scenes", "candidates", "answer", "pipeline", "capacity", "energy", "stats", "display_status", "role_searches", "memory_sources", "inspection_projections", "sentence_plan", "full_sentence_plan", "generation_candidates", "morphology_trace", "reverse_validation", "reasoning_trace", "active_query", "query_session", "query_sessions", "active_query_session_id", "hive_structure", "dialogue_context", "context_resolution", "retrieval_scope", "semantic_total", "gravity", "decision_score")},
            }
            session = self.hive_resonance.create(
                hive_id, text, use_global_memory=resonance_scope != "LOCAL_ONLY",
            )
            result["resonance_session"] = self.hive_resonance.run(session["id"])
            return self._attach_multilevel(hive_id, text, result)
        result = self.service.query(hive_id, text)
        parsed = self.query_scenes.parse(text)
        self.query_scenes.activate(hive_id, text, mode)
        working_hive = self.query_scenes.get(hive_id)
        result.update({
            "query_frame": working_hive["query_frame"],
            "query_scene": working_hive["query_scene"],
            "memory_scenes": working_hive["memory_scenes"],
            "candidates": working_hive["candidates"],
                    "answer": working_hive["answer"],
                })
        if parsed["query_frame"].get("requested_role"):
            searches = self.unknown_searches.resolve_query_unknowns(hive_id)
            if searches:
                result.update(self.service.get_hive(hive_id))
                working_hive = self.query_scenes.get(hive_id)
                result.update({
                    "query_frame": working_hive["query_frame"],
                    "query_scene": working_hive["query_scene"],
                    "memory_scenes": working_hive["memory_scenes"],
                    "candidates": working_hive["candidates"],
                    "answer": working_hive["answer"],
                    "pipeline": working_hive.get("pipeline", {}),
                    "unknown_token_searches": searches,
                })
            else:
                working_hive = self.query_scenes.get(hive_id)
        result["cells"] = working_hive.get("cells", result.get("cells", []))
        result.update({key: working_hive.get(key) for key in ("pipeline", "capacity", "energy", "stats", "display_status", "role_searches", "memory_sources", "inspection_projections", "sentence_plan", "full_sentence_plan", "generation_candidates", "morphology_trace", "reverse_validation", "reasoning_trace", "active_query", "query_session", "query_sessions", "active_query_session_id", "hive_structure", "resonance_probes", "local_resonance", "dialogue_context", "context_resolution", "retrieval_scope", "semantic_total", "gravity", "decision_score") if key in working_hive})
        result["dynamics"] = self.dynamics.get(hive_id)
        working_hive = self.query_scenes.get(hive_id)
        result["resolved_mode"] = mode
        result["hive"] = {**result.get("hive", {}), "intent": working_hive.get("hive", {}).get("intent"), "pipeline": working_hive.get("pipeline", {}), "capacity": working_hive.get("hive", {}).get("capacity", result.get("hive", {}).get("capacity")), "energy": working_hive.get("hive", {}).get("energy", result.get("hive", {}).get("energy")), "max_cells": result.get("hive", {}).get("max_cells", 24)}
        result["hive"].pop("total_energy", None)
        session = self.hive_resonance.create(
            hive_id, text, use_global_memory=resonance_scope != "LOCAL_ONLY",
        )
        result["resonance_session"] = self.hive_resonance.run(session["id"])
        return self._attach_multilevel(hive_id, text, result)

    def _attach_multilevel(self, hive_id: str, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        multilevel = self.multilevel.process(hive_id, text, result)
        result["multilevel"] = multilevel
        result.setdefault("hive", {})["multilevel"] = multilevel
        return result

    def unknown_search_start(self, hive_id: str, surface: str, token_index: int, query_role: str = "", query_scene_id: str = "") -> Dict[str, Any]:
        return self.unknown_searches.start(hive_id, surface, token_index, query_role, query_scene_id)

    def unknown_search_step(self, hive_id: str, search_id: str) -> Dict[str, Any]:
        return self.unknown_searches.step(hive_id, search_id)

    def unknown_search_run(self, hive_id: str, search_id: str) -> Dict[str, Any]:
        return self.unknown_searches.run(hive_id, search_id)

    def unknown_search_vibrate(self, hive_id: str, search_id: str) -> Dict[str, Any]:
        return self.unknown_searches.vibrate(hive_id, search_id)

    def unknown_search_get(self, hive_id: str, search_id: str) -> Dict[str, Any]:
        return self.unknown_searches.get(hive_id, search_id)

    def unknown_search_evidence(self, hive_id: str, search_id: str) -> List[Dict[str, Any]]:
        return self.unknown_searches.evidence(hive_id, search_id)

    def unknown_search_routes(self, hive_id: str, search_id: str) -> List[Dict[str, Any]]:
        return self.unknown_searches.routes(hive_id, search_id)

    def unknown_search_confirm(self, hive_id: str, search_id: str) -> Dict[str, Any]:
        return self.unknown_searches.confirm(hive_id, search_id)

    def parse_query(self, text: str) -> Dict[str, Any]:
        return self.query_scenes.parse(text)

    def classify_resonance(self, text: str) -> Dict[str, str]:
        return self.lexical.classify(text)

    def lexical_candidates(self, hive_id: str, text: str, use_global_memory: bool = True) -> Dict[str, Any]:
        return {"candidates": self.lexical.resolve(hive_id, text, use_global=use_global_memory)}

    def resonance_create(self, hive_id: str, text: str, scope: str = "LOCAL_THEN_GLOBAL", **options: Any) -> Dict[str, Any]:
        options.setdefault("use_global_memory", scope != "LOCAL_ONLY")
        return self.hive_resonance.create(hive_id, text, **options)

    def resonance_step(self, hive_id: str, probe_id: str) -> Dict[str, Any]:
        return self.hive_resonance.step(probe_id)

    def resonance_run(self, hive_id: str, probe_id: str) -> Dict[str, Any]:
        return self.hive_resonance.run(probe_id)

    def resonance_get(self, hive_id: str, probe_id: str) -> Dict[str, Any]:
        return self.hive_resonance.get_for_hive(hive_id, probe_id)

    def resonance_import(self, hive_id: str, probe_id: str, match_id: str, include_scenes: bool = False) -> Dict[str, Any]:
        return self.lexical.import_match(hive_id, probe_id, match_id, include_scenes)

    def resonance_related_scenes(self, hive_id: str, probe_id: str, match_id: str = "") -> Dict[str, Any]:
        return self.lexical.related_scenes(hive_id, probe_id, match_id)

    def resonance_stop(self, session_id: str) -> Dict[str, Any]:
        return self.hive_resonance.stop(session_id)

    def resonance_snapshots(self, session_id: str) -> List[Dict[str, Any]]:
        return self.hive_resonance.snapshots(session_id)

    def import_resonance_concept(self, session_id: str, concept_id: str) -> Dict[str, Any]:
        return self.hive_resonance.import_concept(session_id, concept_id)

    def activate_query(self, hive_id: str, text: str, resolved_mode: str = "NEW_QUERY") -> Dict[str, Any]:
        result = self.query_scenes.activate(hive_id, text, resolved_mode)
        result["dynamics"] = self.dynamics.get(hive_id)
        return result

    def vibration_step(self, hive_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            state = self.resonance.state(hive_id)
            if not state["vibration"]["enabled"]:
                raise ValueError("Для запуска вибрации сначала перенесите хотя бы один результат резонанса в улей")
            dynamics = self.dynamics.step(hive_id, config)
            state["dynamics"] = dynamics
            payload = {"step": int(dynamics.get("step", 0)), "candidates": [], "history_event": None, "hive": state}
            payload["multilevel"] = self.multilevel.refresh_answer(hive_id, payload)
            state["multilevel"] = payload["multilevel"]
            return payload
        except KeyError:
            pass
        dynamics = self.dynamics.step(hive_id, config)
        result = self.query_scenes.step(hive_id, config)
        result["dynamics"] = dynamics
        result["hive"] = self.query_scenes.get(hive_id)
        result["hive"]["dynamics"] = result["dynamics"]
        result["multilevel"] = self.multilevel.refresh_answer(hive_id, result["hive"])
        result["hive"]["multilevel"] = result["multilevel"]
        return result

    def vibration_run(self, hive_id: str, steps: int = 3, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            state = self.resonance.state(hive_id)
            if not state["vibration"]["enabled"]:
                raise ValueError("Для запуска вибрации сначала перенесите хотя бы один результат резонанса в улей")
            for _ in range(max(1, int(steps))):
                self.dynamics.step(hive_id, config)
            state = self.resonance.state(hive_id)
            state["dynamics"] = self.dynamics.get(hive_id)
            payload = {"status": "FINISHED", "steps_completed": max(1, int(steps)), "winner": None, "answer": None, "hive": state}
            payload["multilevel"] = self.multilevel.refresh_answer(hive_id, payload)
            state["multilevel"] = payload["multilevel"]
            return payload
        except KeyError:
            pass
        step_config = {**(config or {}), "max_steps": max(1, int(steps))}
        completed = 0
        last = None
        for _ in range(max(1, int(steps))):
            self.dynamics.step(hive_id, step_config)
            last = self.query_scenes.step(hive_id, step_config)
            completed += 1
            if last["hive"]["vibration"]["status"] in {"FINISHED", "finished"}:
                break
        state = self.query_scenes.get(hive_id)
        dynamics = self.dynamics.get(hive_id)
        state["dynamics"] = dynamics
        winner = next((item for item in state.get("candidates", []) if item.get("status") == "winner"), None)
        if winner and state.get("answer", {}).get("status") == "PLANNING":
            self.query_scenes.generate_resolved_answer(hive_id)
            state = self.query_scenes.get(hive_id)
            state["dynamics"] = dynamics
        if state.get("answer", {}).get("status") == "RESOLVED":
            self.query_scenes.persist_assistant_answer(hive_id)
            state = self.query_scenes.get(hive_id)
            state["dynamics"] = dynamics
        payload = {"status": state["vibration"]["status"], "steps_completed": completed, "winner": winner, "answer": state.get("answer"), "hive": state}
        payload["multilevel"] = self.multilevel.refresh_answer(hive_id, payload)
        state["multilevel"] = payload["multilevel"]
        return payload

    def dynamics_state(self, hive_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.dynamics.get(hive_id, config)

    def dynamics_step(self, hive_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.dynamics.step(hive_id, config)

    def dynamics_history(self, hive_id: str) -> List[Dict[str, Any]]:
        return self.dynamics.history(hive_id)

    def dynamics_reset(self, hive_id: str) -> Dict[str, Any]:
        return self.dynamics.reset(hive_id)

    def dynamics_node(self, hive_id: str, cell_id: str) -> Dict[str, Any]:
        return self.dynamics.node(hive_id, cell_id)

    def dynamics_evictions(self, hive_id: str) -> List[Dict[str, Any]]:
        return self.dynamics.evictions(hive_id)

    def vibration_stop(self, hive_id: str) -> Dict[str, Any]:
        return self.query_scenes.stop(hive_id)

    def query_working_state(self, hive_id: str) -> Dict[str, Any]:
        try:
            state = self.resonance.state(hive_id)
        except KeyError:
            state = self.query_scenes.get(hive_id)
        state["dynamics"] = self.dynamics.get(hive_id)
        with self.service.repository.transaction() as conn:
            row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
            working = json.loads(row["metadata_json"] or "{}").get("query_working_memory", {}) if row else {}
        session_id = working.get("active_resonance_session_id")
        if session_id:
            try:
                state["resonance_session"] = self.hive_resonance.get_for_hive(hive_id, session_id)
            except KeyError:
                pass
        state["multilevel"] = self.multilevel.get(hive_id)
        return state

    def multilevel_state(self, hive_id: str) -> Dict[str, Any]:
        return self.multilevel.get(hive_id)

    def multilevel_traces(self, hive_id: str) -> List[Dict[str, Any]]:
        return self.multilevel.traces(hive_id)

    def multilevel_views(self, hive_id: str, view_id: str = "all") -> Dict[str, Any]:
        state = self.multilevel.store.load(hive_id)
        return VisualizationSuite().build(state, view_id)

    def multilevel_analytics(self, hive_id: str) -> Dict[str, Any]:
        return MultilevelTraceAnalytics().all(self.multilevel.store.load(hive_id))

    def compose_form(
        self,
        hive_id: str,
        concept: str,
        features: Dict[str, Any],
        root: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.multilevel.compose_form(hive_id, concept, features, root)

    def generate(self, hive_id: str, sentence_plan: Dict[str, Any]) -> Dict[str, Any]:
        return MorphologyService(self.service.repository).generate_sentence(hive_id, sentence_plan)

    def forage(self, query: str, max_cells: int = 24) -> Dict[str, Any]:
        """Compatibility helper: create a hive and process its first query."""
        hive = self.create(max_cells)
        return self.query(hive["hive"]["id"], query)

    def get_hive(self, hive_id: str) -> Dict[str, Any]:
        return self.service.get_hive(hive_id)

    def reason(self, hive_id: str, text: str = "", config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.service.repository.transaction() as conn:
            hive = conn.execute("SELECT query_json, metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            parsed = self.service.parser.parse(text, conn) if text else json.loads(hive["query_json"] or "{}")
            if not parsed.get("components"):
                working = json.loads(hive["metadata_json"] or "{}").get("query_working_memory", {})
                frame = working.get("query_frame", {})
                parsed["components"] = [{"normalized_form": token.get("normalized", ""), "word_form_cloud_id": token.get("word_form_cloud_id"), "expected_role": next((role for role, value in frame.get("roles", {}).items() if value.get("index") == token.get("index")), ""), "resolution_state": token.get("resolution_state", "") } for token in frame.get("tokens", [])]
        components = [
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in parsed.get("components", [])
        ]
        activation = QueryActivation(
            tuple(sorted({int(item.get("word_form_cloud_id")) for item in components if item.get("word_form_cloud_id") is not None})),
            tuple(item.get("normalized_form", "") for item in components),
            tuple(item.get("expected_role", "") for item in components),
        )
        result = HiveVibrationEngine(self.service.repository).reason(hive_id, activation, VibrationConfig(**(config or {})))
        return result.to_dict() | {"hive": self.get_hive(hive_id)}

    def export(self, hive_id: str, mode: str = "current", run_id: Optional[str] = None, step: Optional[int] = None, detail: str = "full") -> Dict[str, Any]:
        exporter = HiveExportService(self.service.repository)
        if mode == "current":
            return exporter.current(hive_id, detail)
        if mode == "snapshot":
            if not run_id:
                raise KeyError("run_id")
            return exporter.snapshot(run_id, step)
        if mode == "trace":
            if not run_id:
                raise KeyError("run_id")
            return exporter.trace(run_id, detail)
        if mode == "initial":
            if not run_id:
                raise KeyError("run_id")
            return exporter.snapshot(run_id, 0)
        raise ValueError("unsupported export mode")

    def snapshot(self, hive_id: str, **options: Any) -> Dict[str, Any]:
        return HiveSnapshotProjector(self.service.repository).project(hive_id, **options)

    def diff(self, run_id: str, from_step: int, to_step: int) -> Dict[str, Any]:
        return HiveExportService(self.service.repository).diff(run_id, from_step, to_step)

    def restore(self, hive_id: str, run_id: str, step: int) -> Dict[str, Any]:
        restored = HiveVibrationEngine(self.service.repository).restore(hive_id, run_id, step)
        restored["hive"] = self.get_hive(hive_id)
        return restored

    def runs(self, hive_id: str) -> List[Dict[str, Any]]:
        with self.service.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM hives WHERE id=?", (hive_id,)).fetchone():
                raise KeyError(hive_id)
            return [dict(row) for row in conn.execute("SELECT * FROM hive_reasoning_runs WHERE hive_id=? ORDER BY created_at DESC", (hive_id,))]

    def snapshots(self, hive_id: str, run_id: str) -> List[Dict[str, Any]]:
        with self.service.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM hives WHERE id=? AND id IN (SELECT hive_id FROM hive_reasoning_runs WHERE id=?)", (hive_id, run_id)).fetchone():
                raise KeyError(run_id)
            return [dict(row) for row in conn.execute("SELECT id, run_id, hive_id, step, phase, state_hash, delta_json, clusters_json, events_json, created_at FROM hive_reasoning_snapshots WHERE run_id=? ORDER BY step, id", (run_id,))]

    def analytics(
        self, hive_id: str, run_id: Optional[str] = None, compare_run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return HiveAnalyticsService(self.service.repository).get(hive_id, run_id, compare_run_id)
