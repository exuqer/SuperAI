"""Query-scene reasoning kept in a single working hive."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from server.tokenizer import tokenize_hierarchical

from .repository import V2Repository, decode, encode, utcnow
from .training import RussianMorphology
from .intent import GREETING_WORDS, IntentClassifier


QUESTION_ROLES = {
    "кто": "agent", "кого": "agent_or_object", "кому": "recipient",
    "что": "object", "где": "location", "куда": "destination",
    "откуда": "source", "когда": "time", "как": "manner",
    "почему": "cause", "зачем": "purpose", "чем": "instrument",
    "сколько": "quantity",
}
SUPPORTED_ROLES = ("agent", "action", "modal", "object", "location", "time", "instrument")
ANSWER_ROLE_ORDER = ("agent", "modal", "action", "object", "location", "time", "instrument")
MESSAGE_MODES = {"NEW_QUERY", "LOCAL_RESONANCE", "FOLLOW_UP", "CORRECTION"}
MODAL_WORDS = {
    "можно": {"semantic_function": "possibility"},
    "нельзя": {"semantic_function": "prohibition"},
    "надо": {"semantic_function": "necessity"},
    "нужно": {"semantic_function": "necessity"},
}
COUNTERPART_ACTIONS = {
    frozenset(("покупать", "продавать")): 0.34,
    frozenset(("купить", "продать")): 0.34,
}
ACTION_NEIGHBORS = {
    "покупать": {"купить": 0.82, "приобретать": 0.78, "брать": 0.55, "оплачивать": 0.58, "заказывать": 0.52},
    "ловить": {"поймать": 0.82},
}
ROLE_LABELS = {
    "agent": "КТО?", "object": "ЧТО?", "location": "ГДЕ?", "time": "КОГДА?", "instrument": "ЧЕМ?",
}
LOCATION_PREPOSITIONS = {"в", "во", "на", "из", "от", "к"}
INSTRUMENT_PREPOSITIONS = {"с", "со"}
AGENT_BLACKLIST = {"рынок", "рыба", "кухня", "сеть", "мяч", "птица"}


class TokenResolutionState(str, Enum):
    UNPARSED = "UNPARSED"
    PARSED = "PARSED"
    QUESTION_OPERATOR = "QUESTION_OPERATOR"
    EXACT_FORM_MATCH = "EXACT_FORM_MATCH"
    LEXEME_MATCH = "LEXEME_MATCH"
    PARSED_UNGROUNDED = "PARSED_UNGROUNDED"
    DEEP_SEARCH = "DEEP_SEARCH"
    BRIDGED_PROBABLE = "BRIDGED_PROBABLE"
    BRIDGED_RESOLVED = "BRIDGED_RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    REJECTED = "REJECTED"
    MISS = "MISS"


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class QuerySceneService:
    """Build, score and vibrate a query scene without persisting it as memory."""

    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()
        self.morphology = RussianMorphology()
        self.intent_classifier = IntentClassifier()

    def parse(self, text: str) -> Dict[str, Any]:
        intent = self.intent_classifier.classify(text)
        tokens = tokenize_hierarchical(text).all_tokens
        items = []
        question_word = ""
        for index, token in enumerate(tokens):
            morph = self.morphology.parse(token.normalized)
            normalized = token.normalized.casefold()
            is_question = normalized in QUESTION_ROLES and not question_word and intent["intent"] not in {"SMALL_TALK", "GREETING", "GREETING_WITH_SMALL_TALK"}
            if is_question:
                question_word = normalized
            modal = MODAL_WORDS.get(normalized)
            items.append({
                "index": index, "surface": token.text, "normalized": normalized,
                "lemma": morph.lemma, "part_of_speech": morph.pos_tag,
                "grammatical_features": morph.features,
                "component_type": "question_operator" if is_question else "query_modal" if modal else "query_token",
                "role": "modal" if modal else None,
                "semantic_function": modal["semantic_function"] if modal else None,
                "expected_role": QUESTION_ROLES.get(normalized) if is_question else None,
                "resolution_state": TokenResolutionState.QUESTION_OPERATOR.value if is_question else TokenResolutionState.EXACT_FORM_MATCH.value if modal else TokenResolutionState.PARSED_UNGROUNDED.value,
            })
        with self.repository.transaction() as conn:
            for item in items:
                if item["resolution_state"] == TokenResolutionState.QUESTION_OPERATOR.value:
                    continue
                row = conn.execute("SELECT wf.cloud_id AS word_form_cloud_id, wf.lexeme_cloud_id, l.lemma FROM word_forms wf LEFT JOIN lexemes l ON l.cloud_id=wf.lexeme_cloud_id WHERE wf.normalized_form=? LIMIT 1", (item["normalized"],)).fetchone()
                if row:
                    item.update({"lexeme": row["lemma"] or item["lemma"], "word_form_cloud_id": int(row["word_form_cloud_id"]), "lexeme_cloud_id": int(row["lexeme_cloud_id"]) if row["lexeme_cloud_id"] else None, "resolution_state": TokenResolutionState.EXACT_FORM_MATCH.value})
        requested_role = QUESTION_ROLES.get(question_word, "")
        if intent["intent"] in {"SMALL_TALK", "GREETING", "GREETING_WITH_SMALL_TALK"}:
            requested_role = ""
        roles = self._roles(items)
        for role, value in roles.items():
            if value.get("status") == "empty":
                continue
            token = next((item for item in items if item["index"] == value.get("index")), None)
            if token:
                token["component_type"] = {"action": "query_predicate", "object": "query_object", "agent": "query_subject", "modal": "query_modal"}.get(role, "query_token")
                if token.get("normalized") in MODAL_WORDS:
                    token.update({"component_type": "query_modal", "role": "modal", "semantic_function": MODAL_WORDS[token["normalized"]]["semantic_function"], "resolution_state": TokenResolutionState.EXACT_FORM_MATCH.value})
        if requested_role:
            roles[requested_role] = {
                "status": "empty", "value": None, "required": True,
                "question_word": question_word,
            }
        frame_id = f"query-frame-{uuid.uuid4().hex[:12]}"
        normalized_text = " ".join(item["lemma"] for item in items)
        predicate = roles.get("action", {})
        query_frame = {
            "id": frame_id,
            "source_text": text, "original_text": text,
            "normalized_text": normalized_text,
            "query_type": "role_question" if requested_role else "statement",
            "question_word": question_word or None,
            "requested_role": requested_role or None,
            "intent": intent["intent"],
            "intent_classification": intent,
            "predicate": self._token_value(predicate) if predicate else None,
            "roles": roles,
            "tokens": items,
        }
        slots = []
        for role in SUPPORTED_ROLES:
            value = roles.get(role)
            if not value:
                continue
            if value.get("status") == "empty":
                slots.append({
                    "id": f"slot-{role}", "type": "role_slot", "role": role,
                    "question_word": question_word, "label": ROLE_LABELS.get(role, role.upper()),
                    "status": "empty", "required": True, "candidates": [],
                })
            else:
                slots.append({"id": f"slot-{role}", "role": role, "status": "fixed", **self._token_value(value)})
        query_scene = {
            "id": f"query-scene-{uuid.uuid4().hex[:12]}", "type": "query_scene",
            "status": "INCOMPLETE" if requested_role else "RESOLVED", "source_query": text,
            "requested_role": requested_role or None, "slots": slots,
        }
        if intent["intent"] in {"SMALL_TALK", "GREETING", "GREETING_WITH_SMALL_TALK"}:
            return {"intent_classification": intent, "query_frame": query_frame, "query_scene": None}
        return {"intent_classification": intent, "query_frame": query_frame, "query_scene": query_scene}

    def activate(self, hive_id: str, text: str, mode: str = "NEW_QUERY") -> Dict[str, Any]:
        parsed = self.parse(text)
        with self.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM hives WHERE id=?", (hive_id,)).fetchone():
                raise KeyError(hive_id)
            hive = conn.execute("SELECT conversation_id FROM hives WHERE id=?", (hive_id,)).fetchone()
            message = conn.execute("SELECT id FROM hive_messages WHERE hive_id=? ORDER BY turn_index DESC LIMIT 1", (hive_id,)).fetchone()
            parsed = self._resolve_token_states(conn, parsed)
            mode = mode if mode in MESSAGE_MODES else "NEW_QUERY"
            self._clear_temporary_query_objects(conn, hive_id)
            if parsed["intent_classification"]["intent"] in {"SMALL_TALK", "GREETING", "GREETING_WITH_SMALL_TALK"}:
                return self._activate_conversational(conn, hive_id, text, mode, hive, message, parsed)
            memory_scenes = self._memory_scenes(conn)
            evaluated = [self._score_scene(parsed["query_frame"], scene) for scene in memory_scenes]
            candidates = self._candidates(parsed["query_frame"], evaluated)
            for slot in parsed["query_scene"]["slots"]:
                if slot["status"] == "empty":
                    slot["candidates"] = [candidate["id"] for candidate in candidates]
            result_type = self._result_type(evaluated, candidates)
            active_session_id = f"query-session-{uuid.uuid4().hex[:12]}"
            previous = self._load(conn, hive_id)
            sessions = list(previous.get("query_sessions", []))
            if previous.get("query_session"):
                previous["query_session"]["status"] = "ARCHIVED"
                previous["query_session"]["completed_at"] = utcnow()
                sessions.append(previous["query_session"])
            query_session = {
                "id": active_session_id,
                "hive_id": hive_id,
                "message_id": str(message["id"] if message else ""),
                "source_text": text,
                "mode": mode,
                "status": "ACTIVE",
                "started_at": utcnow(),
            }
            state = {
                "id": hive_id, "status": "ACTIVE", "energy": self._energy(evaluated),
                "conversation_id": str(hive["conversation_id"] or "") if hive else "",
                "message_id": str(message["id"] if message else ""),
                "query_frame_id": parsed["query_frame"]["id"],
                "query_scene_id": parsed["query_scene"]["id"],
                "active_query_session_id": active_session_id,
                "query_session": query_session,
                "query_sessions": sessions,
                "created_for_surface": text,
                **parsed, "memory_scenes": evaluated, "candidates": candidates,
                "result_type": result_type,
                "memory_sources": [], "inspection_projections": [], "generation_candidates": [],
                "role_searches": [{"id": f"role-search-{uuid.uuid4().hex[:10]}", "target_role": parsed["query_frame"].get("requested_role"), "role_candidates": [], "selected_role_candidate": None, "query_frame_id": parsed["query_frame"]["id"], "query_scene_id": parsed["query_scene"]["id"]}] if parsed["query_frame"].get("requested_role") else [],
                "sentence_plan": None, "morphology_trace": [],
                "vibration": {"current_step": 0, "max_steps": 5, "status": "READY", "history": []},
                "answer": self._empty_answer(parsed["query_frame"], result_type),
                "reasoning_trace": {
                    "version": 1,
                    "query_session_id": active_session_id,
                    "stages": [
                        {
                            "id": "intent-classification", "stage": "INTENT_CLASSIFICATION", "status": "RESOLVED",
                            "input": text, "output": deepcopy(parsed["intent_classification"]),
                        },
                        {
                            "id": "query-frame", "stage": "QUERY_FRAME", "status": "RESOLVED",
                            "output": {
                                "requested_role": parsed["query_frame"].get("requested_role"),
                                "roles": deepcopy(parsed["query_frame"].get("roles", {})),
                                "tokens": deepcopy(parsed["query_frame"].get("tokens", [])),
                            },
                        },
                        {
                            "id": "memory-scene-search", "stage": "MEMORY_SCENE_SEARCH",
                            "status": "MATCHES_FOUND" if any(item["result_type"] != "NO_HIT" for item in evaluated) else "NO_MATCH",
                            "output": [self._scene_trace(item) for item in evaluated],
                        },
                        {
                            "id": "candidate-ranking", "stage": "CANDIDATE_RANKING",
                            "status": "CANDIDATES_FOUND" if candidates else "NO_CANDIDATES",
                            "output": [self._candidate_trace(item) for item in candidates],
                        },
                    ],
                },
            }
            from .unknown_search import UnknownTokenSearchService
            placer = UnknownTokenSearchService(self.repository)
            placement_search = {"id": f"query-placement-{uuid.uuid4().hex[:10]}", "query_session_id": active_session_id, "query_frame_id": parsed["query_frame"]["id"], "query_scene_id": parsed["query_scene"]["id"], "message_id": str(message["id"] if message else ""), "created_for_surface": text}
            for candidate in candidates:
                candidate.update({"conversation_id": str(hive["conversation_id"] or "") if hive else "", "message_id": str(message["id"] if message else ""), "query_session_id": active_session_id, "query_frame_id": parsed["query_frame"]["id"], "query_scene_id": parsed["query_scene"]["id"], "created_for_surface": text})
                placer._place_role_candidate(conn, hive_id, candidate)
                for source_id in candidate.get("sources", []):
                    source = next((item for item in evaluated if item.get("id") == source_id), None)
                    scene_id = str(source_id).removeprefix("scene-")
                    if source and scene_id.isdigit() and source.get("result_type") in {"FULL_HIT", "ROLE_HIT"} and float(source.get("scores", {}).get("total_score", 0.0)) >= 0.55:
                        placer._place_memory_source(conn, hive_id, int(scene_id), source, placement_search)
                        if not any(item.get("source_scene_id") == int(scene_id) for item in state["memory_sources"]):
                            state["memory_sources"].append({"id": f"memory-source-{scene_id}", "component_class": "memory_source", "source_scene_id": int(scene_id), "source_text": source.get("source_text", ""), "query_frame_id": parsed["query_frame"]["id"], "message_id": str(message["id"] if message else "")})
            self._set_pipeline(state, candidate_count=len(candidates), memory_source_count=len(state["memory_sources"]))
            state["hive"] = {"status": "ACTIVE", "reasoning_step": 0, "pipeline": state["pipeline"]}
            conn.execute("UPDATE hives SET query_text=?, query_json=?, updated_at=? WHERE id=?", (text, encode({"original_text": text, "query_frame_id": parsed["query_frame"]["id"], "query_scene_id": parsed["query_scene"]["id"], "active_query_session_id": active_session_id}), utcnow(), hive_id))
            self._save(conn, hive_id, state)
            return deepcopy(state)

    def _activate_conversational(self, conn: Any, hive_id: str, text: str, mode: str, hive: Any, message: Any, parsed: Dict[str, Any]) -> Dict[str, Any]:
        self._clear_context_objects(conn, hive_id)
        active_session_id = f"query-session-{uuid.uuid4().hex[:12]}"
        previous = self._load(conn, hive_id)
        sessions = list(previous.get("query_sessions", []))
        if previous.get("query_session"):
            previous["query_session"].update({"status": "ARCHIVED", "completed_at": utcnow()})
            sessions.append(previous["query_session"])
        intent = parsed["intent_classification"]
        answer_status = "UNRESOLVED_CONVERSATIONAL" if intent["intent"] != "GREETING" else "RESOLVED_GREETING"
        surface = "Подходящий разговорный ответ в доступной памяти не найден." if answer_status == "UNRESOLVED_CONVERSATIONAL" else "Здравствуйте!"
        state = {
            "id": hive_id, "status": "ACTIVE", "conversation_id": str(hive["conversation_id"] or "") if hive else "",
            "message_id": str(message["id"] if message else ""), "created_for_surface": text,
            "active_query_session_id": active_session_id,
            "query_session": {"id": active_session_id, "hive_id": hive_id, "message_id": str(message["id"] if message else ""), "source_text": text, "mode": mode, "status": "ACTIVE", "started_at": utcnow()},
            "query_sessions": sessions, **parsed,
            "query_frame_id": parsed["query_frame"]["id"], "query_scene_id": None,
            "memory_scenes": [], "search_hits": [], "candidates": [], "role_candidates": [], "role_searches": [],
            "working_cells": [], "memory_sources": [], "inspection_projections": [], "generation_candidates": [],
            "vibration": {"current_step": 0, "max_steps": 0, "status": "IDLE", "enabled": False, "history": []},
            "dynamics": {"status": "IDLE", "temperature": {"status": "NOT_STARTED"}, "nodes": [], "history": [], "semantic_reasoning_step": 0, "physical_step": 0},
            "pipeline": {"intent_classification": {"status": intent["intent"]}, "greeting_processing": {"status": "RESOLVED" if intent.get("greeting") else "SKIPPED"}, "conversational_memory_search": {"status": "NO_MATCH"}, "scene_query": {"status": "SKIPPED"}, "role_search": {"status": "SKIPPED"}, "dynamics": {"status": "IDLE"}, "answer": {"status": answer_status}},
            "answer": {"status": answer_status, "answer_mode": "conversational", "resolved_role": None, "resolved_value": None, "confidence": 0.0, "supporting_scenes": [], "surface": surface, "surface_answer": surface},
            "display_status": "CONVERSATIONAL_NO_MATCH" if answer_status == "UNRESOLVED_CONVERSATIONAL" else "SMALL_TALK_DETECTED",
            "hive": {"status": "ACTIVE", "reasoning_step": 0, "semantic_reasoning_step": 0, "physical_step": 0},
        }
        conn.execute("UPDATE hives SET query_text=?, query_json=?, reasoning_step=0, current_temperature=0, updated_at=? WHERE id=?", (text, encode({"original_text": text, "query_frame_id": parsed["query_frame"]["id"], "query_scene_id": None, "active_query_session_id": active_session_id}), utcnow(), hive_id))
        self._save(conn, hive_id, state)
        return deepcopy(state)

    def _resolve_token_states(self, conn: Any, parsed: Dict[str, Any]) -> Dict[str, Any]:
        frame = parsed["query_frame"]
        for token in frame["tokens"]:
            if token["resolution_state"] == TokenResolutionState.QUESTION_OPERATOR.value:
                continue
            row = conn.execute("""SELECT wf.cloud_id AS word_form_cloud_id, wf.lexeme_cloud_id, l.lemma
                FROM word_forms wf LEFT JOIN lexemes l ON l.cloud_id=wf.lexeme_cloud_id
                WHERE wf.normalized_form=? LIMIT 1""", (token["normalized"],)).fetchone()
            role = next((name for name, value in frame.get("roles", {}).items() if value.get("index") == token["index"]), "")
            token["component_type"] = {
                "action": "query_predicate", "object": "query_object", "agent": "query_subject"
            }.get(role, "query_token")
            if row:
                token.update({"lexeme": row["lemma"] or token["lemma"], "word_form_cloud_id": int(row["word_form_cloud_id"]), "lexeme_cloud_id": int(row["lexeme_cloud_id"]) if row["lexeme_cloud_id"] else None, "resolution_state": TokenResolutionState.EXACT_FORM_MATCH.value})
            elif token.get("lemma"):
                token["resolution_state"] = TokenResolutionState.PARSED_UNGROUNDED.value
            else:
                token["resolution_state"] = TokenResolutionState.MISS.value
        for role, value in frame.get("roles", {}).items():
            if value.get("status") == "empty":
                continue
            matching = next((item for item in frame["tokens"] if item["index"] == value.get("index")), None)
            if matching:
                value.update({key: matching[key] for key in ("word_form_cloud_id", "lexeme_cloud_id") if key in matching})
        return parsed

    def resolve_mode(self, hive_id: str, text: str, resolved_mode: Optional[str] = None) -> str:
        if resolved_mode:
            if resolved_mode not in MESSAGE_MODES:
                raise ValueError(f"unsupported resolved_mode: {resolved_mode}")
            return resolved_mode
        return "NEW_QUERY"

    def local_resonance(self, hive_id: str, surface: str, allow_global: bool = False) -> Dict[str, Any]:
        normalized = surface.strip().casefold()
        if not normalized:
            raise ValueError("surface must not be empty")
        with self.repository.transaction() as conn:
            state = self._load(conn, hive_id)
            if not state:
                raise KeyError(hive_id)
            active = conn.execute(
                """SELECT wf.cloud_id AS word_form_cloud_id, wf.lexeme_cloud_id, l.lemma
                FROM word_forms wf LEFT JOIN lexemes l ON l.cloud_id=wf.lexeme_cloud_id
                WHERE wf.normalized_form=? AND EXISTS (
                    SELECT 1 FROM hive_cell_components hcc JOIN hive_cells hc ON hc.id=hcc.cell_id
                    WHERE hc.hive_id=? AND hcc.cloud_id=wf.cloud_id
                ) LIMIT 1""", (normalized, hive_id)
            ).fetchone()
            source = "active_hive"
            if not active and allow_global:
                active = conn.execute(
                    """SELECT wf.cloud_id AS word_form_cloud_id, wf.lexeme_cloud_id, l.lemma
                    FROM word_forms wf LEFT JOIN lexemes l ON l.cloud_id=wf.lexeme_cloud_id
                    WHERE wf.normalized_form=? LIMIT 1""", (normalized,)
                ).fetchone()
                source = "global_memory" if active else source
            probe_id = f"resonance-probe-{uuid.uuid4().hex[:12]}"
            now = utcnow()
            probe = {
                "id": probe_id, "hive_id": hive_id,
                "active_query_session_id": state.get("active_query_session_id"),
                "surface": surface, "mode": "LOCAL_ONLY" if not allow_global else "LOCAL_THEN_GLOBAL",
                "status": "LOCAL_HIT" if active and source == "active_hive" else "GLOBAL_HIT" if active else "COMPLETED_NO_MATCH",
                "matches": [], "started_at": now, "completed_at": now,
            }
            if active:
                probe["matches"] = [{
                    "type": "word_form", "cloud_id": int(active["word_form_cloud_id"]),
                    "surface": surface, "lemma": active["lemma"] or normalized,
                    "word_form_cloud_id": int(active["word_form_cloud_id"]),
                    "lexeme_cloud_id": int(active["lexeme_cloud_id"]) if active["lexeme_cloud_id"] else None,
                    "confidence": 1.0, "source": source,
                }]
            probes = [item for item in state.get("resonance_probes", []) if item.get("id") != probe_id]
            probes.append(probe)
            state["resonance_probes"] = probes
            state["local_resonance"] = {
                "latest_probe_id": probe_id, "latest_surface": surface,
                "status": probe["status"], "probe_text": surface,
                "matched_form": surface if active else None,
                "matched_lexeme": active["lemma"] if active else None,
            }
            self._save(conn, hive_id, state)
            return self._decorate_state(conn, hive_id, state)

    @staticmethod
    def _clear_temporary_query_objects(conn: Any, hive_id: str) -> None:
        rows = conn.execute("""SELECT hive_placement_id FROM hive_cells WHERE hive_id=?
            AND component_class IN ('role_candidate','memory_source','semantic_bridge')""", (hive_id,)).fetchall()
        conn.execute("""DELETE FROM hive_cells WHERE hive_id=?
            AND component_class IN ('role_candidate','memory_source','semantic_bridge')""", (hive_id,))
        for row in rows:
            conn.execute("DELETE FROM cloud_placements WHERE id=?", (row["hive_placement_id"],))
        conn.execute("DELETE FROM hive_subspaces WHERE hive_id=?", (hive_id,))
        conn.execute("DELETE FROM hive_generation_candidates WHERE hive_id=?", (hive_id,))

    @staticmethod
    def _clear_context_objects(conn: Any, hive_id: str) -> None:
        rows = conn.execute("SELECT hive_placement_id FROM hive_cells WHERE hive_id=? AND component_class='context'", (hive_id,)).fetchall()
        conn.execute("DELETE FROM hive_cells WHERE hive_id=? AND component_class='context'", (hive_id,))
        for row in rows:
            conn.execute("DELETE FROM cloud_placements WHERE id=?", (row["hive_placement_id"],))

    @staticmethod
    def _set_pipeline(state: Dict[str, Any], candidate_count: int, memory_source_count: int) -> None:
        empty_roles = [slot["role"] for slot in state["query_scene"]["slots"] if slot.get("status") == "empty"]
        state["pipeline"] = {
            "query_parse": {"status": "RESOLVED"},
            "token_resolution": {"status": "PROBABLE_MATCH" if any(token.get("resolution_state") in {"PARSED_UNGROUNDED", "BRIDGED_PROBABLE"} for token in state["query_frame"]["tokens"]) else "RESOLVED"},
            "memory_search": {"status": "ROLE_CANDIDATES_FOUND" if candidate_count else "NO_MATCH", "memory_source_count": memory_source_count, "candidate_count": candidate_count},
            "query_scene": {"status": "INCOMPLETE" if empty_roles else "RESOLVED", "empty_roles": empty_roles},
            "vibration": {"status": "READY", "current_step": 0},
            "sentence_planning": {"status": "WAITING"},
            "morphology_generation": {"status": "WAITING"},
            "answer": {"status": "PENDING"},
        }

    def get(self, hive_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            state = self._load(conn, hive_id)
            if not state:
                raise KeyError(hive_id)
            return self._decorate_state(conn, hive_id, state)

    def step(self, hive_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = {**self._defaults(), **(config or {})}
        with self.repository.transaction() as conn:
            state = self._load(conn, hive_id)
            if not state:
                raise KeyError(hive_id)
            vibration = state["vibration"]
            if vibration["status"] in {"FINISHED", "STOPPED", "finished", "stopped"}:
                return {"step": vibration["current_step"], "candidates": state["candidates"], "history_event": None, "hive": state}
            step = int(vibration["current_step"]) + 1
            active = [candidate for candidate in state["candidates"] if candidate["status"] not in {"evicted", "EVICTED", "conflict"}]
            total_source = sum(candidate["scores"]["source_confidence"] for candidate in active) or 1.0
            changes, evicted = [], []
            for candidate in active:
                scores = candidate["scores"]
                before = float(scores["total"])
                competition = max(0.0, (sum(other["scores"]["total"] for other in active if other["id"] != candidate["id"]) / max(1, len(active) - 1) - before) * 0.25)
                resonance = clamp(scores["source_confidence"] / total_source * len(active))
                contradiction = float(scores.get("contradiction", 0.0))
                after = clamp(
                    before * config["inertia"]
                    + scores["role_compatibility"] * config["role_force"]
                    + scores["action_compatibility"] * config["action_force"]
                    + scores["object_compatibility"] * config["object_force"]
                    + scores["source_confidence"] * config["source_force"]
                    + resonance * config["resonance_force"]
                    - contradiction * config["contradiction_force"]
                    - competition * config["competition_force"]
                )
                scores["resonance"] = resonance
                scores["activation"] = after
                scores["retention"] = clamp(scores.get("retention", before) * 0.65 + after * 0.35)
                scores["evidence_confidence"] = float(scores.get("evidence_confidence", scores.get("source_confidence", before)))
                scores["semantic_confidence"] = float(scores.get("semantic_confidence", scores.get("source_confidence", before)))
                scores["answer_confidence"] = min(scores["semantic_confidence"], scores["evidence_confidence"], float(scores.get("source_confidence", before)), scores["retention"])
                scores["total"] = after
                candidate["weak_steps"] = candidate.get("weak_steps", 0) + 1 if after < config["activation_threshold"] and scores["retention"] < config["retention_threshold"] else 0
                if contradiction:
                    candidate["status"] = "conflict"
                elif candidate["weak_steps"] >= config["eviction_after_steps"]:
                    candidate["status"] = "evicted"
                    evicted.append(candidate["id"])
                elif after > before + 0.015:
                    candidate["status"] = "strengthened"
                elif after < before - 0.015:
                    candidate["status"] = "weakened"
                else:
                    candidate["status"] = "stable"
                changes.append({
                    "candidate_id": candidate["id"], "before": round(before, 4), "after": round(after, 4),
                    "delta": round(after - before, 4), "reasons": self._reasons(candidate),
                })
            state["candidates"].sort(key=lambda item: (-item["scores"]["total"], item["lemma"]))
            winner = self._winner(state, config)
            if winner:
                winner["status"] = "winner"
                self._resolve(state, winner, step)
            event = {"step": step, "timestamp": 0, "candidate_changes": changes, "evicted_candidates": evicted, "winner": winner["id"] if winner else None}
            state.setdefault("reasoning_trace", {}).setdefault("stages", []).append({
                "id": f"vibration-{step}", "stage": "VIBRATION", "status": "WINNER_FOUND" if winner else "RUNNING",
                "step": step, "output": deepcopy(event),
            })
            vibration["current_step"] = step
            vibration["history"].append(event)
            if winner or step >= config["max_steps"] or not any(item["status"] != "evicted" for item in state["candidates"]):
                vibration["status"] = "FINISHED"
                state["status"] = "STABLE"
                if state.get("pipeline") and not winner:
                    state["pipeline"]["vibration"] = {"status": "FINISHED", "current_step": step}
                    state["pipeline"]["answer"] = {"status": "UNRESOLVED"}
                    state["answer"].update({"status": "UNRESOLVED", "answer_mode": "partial" if state.get("result_type") in {"PARTIAL_HIT", "CONFLICT_HIT"} else "unknown"})
            elif state.get("pipeline"):
                state["pipeline"]["vibration"] = {"status": "RUNNING", "current_step": step}
            self._save(conn, hive_id, state)
            return {"step": step, "candidates": state["candidates"], "history_event": event, "hive": self._decorate_state(conn, hive_id, state)}

    def run(self, hive_id: str, steps: int = 3, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = {**(config or {}), "max_steps": max(1, int(steps))}
        completed = 0
        while completed < int(steps):
            response = self.step(hive_id, config)
            completed += 1
            if response["hive"]["vibration"]["status"] in {"FINISHED", "finished"}:
                break
        state = self.get(hive_id)
        winner = next((item for item in state["candidates"] if item["status"] == "winner"), None)
        if winner and state.get("answer", {}).get("status") == "PLANNING":
            self.generate_resolved_answer(hive_id)
            state = self.get(hive_id)
        return {"status": state["vibration"]["status"], "steps_completed": completed, "winner": winner, "answer": state["answer"], "hive": state}

    def stop(self, hive_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            state = self._load(conn, hive_id)
            if not state:
                raise KeyError(hive_id)
            state["vibration"]["status"] = "STOPPED"
            self._save(conn, hive_id, state)
            return state

    def _roles(self, tokens: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        roles: Dict[str, Dict[str, Any]] = {}
        predicate_index = next((item["index"] for item in tokens if item["part_of_speech"] in {"VERB", "INFN"}), None)
        for item in tokens:
            if item["normalized"] in QUESTION_ROLES or item["normalized"] in GREETING_WORDS or item["normalized"] in {"не", "ли"}:
                continue
            if item["normalized"] in MODAL_WORDS:
                roles["modal"] = {"status": "fixed", "component_type": "query_modal", "role": "modal", "semantic_function": MODAL_WORDS[item["normalized"]]["semantic_function"], **item}
                continue
            if item["part_of_speech"] in {"VERB", "INFN"} and "action" not in roles:
                roles["action"] = {"status": "fixed", **item}
                continue
            if item["part_of_speech"] not in {"NOUN", "NPRO", "NUMR"}:
                continue
            previous = tokens[item["index"] - 1]["normalized"] if item["index"] else ""
            case = item["grammatical_features"].get("case")
            role = ""
            if previous in LOCATION_PREPOSITIONS:
                role = "location"
            elif previous in INSTRUMENT_PREPOSITIONS or case == "ablt":
                role = "instrument"
            elif case == "nomn":
                role = "agent"
            elif case in {"accs", "gent", "datv"} or predicate_index is not None:
                role = "object"
            if role and role not in roles:
                roles[role] = {"status": "fixed", **item, "preposition": previous if previous in LOCATION_PREPOSITIONS | INSTRUMENT_PREPOSITIONS else ""}
        return roles

    @staticmethod
    def _token_value(value: Dict[str, Any]) -> Dict[str, Any]:
        features = value.get("grammatical_features", {})
        return {
            "lemma": value.get("lemma"), "surface": value.get("surface"),
            "part_of_speech": value.get("part_of_speech"), "case": features.get("case"),
            "number": features.get("number"), "gender": features.get("gender"),
            "tense": features.get("tense"), "person": features.get("person"),
            "grammatical_features": features,
            "component_type": value.get("component_type"),
            "semantic_function": value.get("semantic_function"),
        }

    def _memory_scenes(self, conn: Any) -> List[Dict[str, Any]]:
        rows = conn.execute("SELECT s.cloud_id, s.sentence_text, s.canonical_text FROM scenes s ORDER BY s.updated_at DESC").fetchall()
        result = []
        for row in rows:
            components = []
            for component in conn.execute(
                """SELECT sc.token_index, sc.grammatical_role, sc.morphology_json, wf.normalized_form,
                   l.lemma FROM scene_components sc JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id
                   LEFT JOIN lexemes l ON l.cloud_id=sc.lexeme_cloud_id
                   WHERE sc.scene_cloud_id=? ORDER BY sc.token_index""", (row["cloud_id"],)
            ):
                morphology = json.loads(component["morphology_json"] or "{}")
                components.append({
                    "index": int(component["token_index"]), "surface": component["normalized_form"],
                    "normalized": component["normalized_form"], "lemma": component["lemma"] or morphology.get("lemma") or component["normalized_form"],
                    "part_of_speech": morphology.get("pos", "UNK"),
                    "grammatical_features": {key: value for key, value in morphology.items() if key != "pos" and key != "lemma"},
                })
            result.append({
                "id": f"scene-{row['cloud_id']}", "cloud_id": int(row["cloud_id"]), "type": "memory_scene",
                "source_text": row["sentence_text"], "roles": self._roles(components),
                "negation": any(item["normalized"] == "не" for item in components),
            })
        return result

    def _action_match(self, query: str, memory: str) -> float:
        if not query or not memory:
            return 0.0
        if query == memory:
            return 1.0
        if memory in ACTION_NEIGHBORS.get(query, {}):
            return ACTION_NEIGHBORS[query][memory]
        if query in ACTION_NEIGHBORS.get(memory, {}):
            return ACTION_NEIGHBORS[memory][query]
        return COUNTERPART_ACTIONS.get(frozenset((query, memory)), 0.0)

    def _score_scene(self, frame: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Any]:
        query_roles, memory_roles = frame["roles"], scene["roles"]
        requested = frame.get("requested_role")
        fixed_roles = {
            role: value for role, value in query_roles.items()
            if role != requested and value.get("status") == "fixed" and value.get("lemma")
        }
        role_matches: Dict[str, float] = {}
        for role, query_value in fixed_roles.items():
            memory_value = memory_roles.get(role, {})
            if role == "action":
                role_matches[role] = self._action_match(query_value.get("lemma", ""), memory_value.get("lemma", ""))
            else:
                role_matches[role] = float(bool(memory_value and query_value.get("lemma") == memory_value.get("lemma")))
        action_match = role_matches.get("action", 0.0)
        object_match = role_matches.get("object", 0.0)
        location_match = role_matches.get("location", 0.0)
        agent_match = role_matches.get("agent", 0.0)
        anchor_match = sum(role_matches.values()) / max(1, len(role_matches))
        grammar_match = 1.0 if anchor_match >= .99 else .65 if anchor_match > 0 else 0.0
        role_value = memory_roles.get(requested or "")
        requested_role_match = 0.0 if not role_value or scene["negation"] else 1.0
        structural_match = clamp(anchor_match * .65 + requested_role_match * .35)
        semantic_match = clamp(anchor_match * .75 + max(role_matches.values(), default=0.0) * .25)
        total = clamp(anchor_match * .50 + requested_role_match * .20 + semantic_match * .15 + structural_match * .10 + grammar_match * .05)
        matched_roles = [role for role, score in role_matches.items() if score >= .5]
        mismatched_roles = [role for role, score in role_matches.items() if score < .5]
        if scene["negation"] and (anchor_match or requested_role_match):
            result_type = "CONFLICT_HIT"
        elif requested_role_match and fixed_roles and all(score >= .99 for score in role_matches.values()):
            result_type = "FULL_HIT"
        elif requested_role_match and anchor_match > 0:
            result_type = "ROLE_HIT"
        elif anchor_match > 0:
            result_type = "PARTIAL_HIT"
        elif semantic_match:
            result_type = "WEAK_HIT"
        else:
            result_type = "NO_HIT"
        return {**scene, "scores": {
            "object_match": object_match, "action_match": action_match, "agent_match": agent_match,
            "location_match": location_match, "requested_role_match": requested_role_match,
            "anchor_match": anchor_match, "role_matches": role_matches,
            "grammar_match": grammar_match, "semantic_match": semantic_match, "structural_match": structural_match,
            "source_confidence": 0.9, "retention": 1.0, "total_score": total,
        }, "matched_roles": matched_roles, "mismatched_roles": mismatched_roles,
            "selection_reason": "совпали опорные роли: " + ", ".join(matched_roles) if matched_roles else "опорные роли не совпали",
            "result_type": result_type}

    def _candidates(self, frame: Dict[str, Any], scenes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        requested = frame.get("requested_role")
        found: Dict[str, Dict[str, Any]] = {}
        if not requested:
            return []
        for scene in scenes:
            if not scene["scores"].get("anchor_match"):
                continue
            value = scene["roles"].get(requested)
            if not value:
                continue
            if requested == "agent" and (value.get("part_of_speech") not in {"NOUN", "NPRO"} or value.get("lemma") in AGENT_BLACKLIST):
                continue
            lemma = value["lemma"]
            action = scene["scores"]["action_match"]
            object_match = scene["scores"]["object_match"]
            anchor_match = scene["scores"]["anchor_match"]
            contradiction = 1.0 if scene["negation"] else 0.0
            current = found.get(lemma)
            support = clamp(scene["scores"]["total_score"] + .15)
            query_relevance = clamp(anchor_match * .65 + scene["scores"]["requested_role_match"] * .35)
            semantic_support = clamp(scene["scores"]["semantic_match"])
            role_compatibility = .98
            exact_match = float(scene["scores"]["object_match"] > 0 and scene["scores"]["action_match"] >= .99)
            activation = clamp(.05 + exact_match * .30 + query_relevance * .25 + semantic_support * .15 + role_compatibility * .15 + support * .10 + scene["scores"]["structural_match"] * .05)
            retention = clamp(scene["scores"]["source_confidence"] * .25 + support * .20 + query_relevance * .15 + role_compatibility * .10 + .05)
            candidate = {
                "id": f"candidate-{lemma}-{uuid.uuid5(uuid.NAMESPACE_URL, lemma).hex[:8]}",
                "concept_id": f"concept-{lemma}", "lemma": lemma, "surface": value["surface"],
                "preposition": value.get("preposition", ""),
                "grammatical_features": deepcopy(value.get("grammatical_features", {})),
                "form_provenance": {
                    "source_type": "observed_training_form",
                    "scene_id": scene["id"],
                    "scene_text": scene.get("source_text", ""),
                    "observed_surface": value["surface"],
                    "generated": False,
                },
                "target_role": requested, "part_of_speech": value.get("part_of_speech", "NOUN"),
                "entity_type": "person" if requested == "agent" else "entity",
                "sources": [scene["id"]], "primary_source_id": scene["id"],
                "scores": {
                    "role_compatibility": role_compatibility, "query_relevance": query_relevance, "semantic_support": semantic_support, "structural_support": scene["scores"]["structural_match"], "exact_match": exact_match, "action_compatibility": action, "object_compatibility": object_match, "anchor_compatibility": anchor_match,
                    "grammar_compatibility": .99, "source_confidence": support, "resonance": 0.0,
                    "retention": retention, "activation": activation, "contradiction": contradiction,
                    "evidence_confidence": support, "semantic_confidence": support,
                    "answer_confidence": support,
                    "total": clamp(.28 * role_compatibility + .30 * anchor_match + .17 * semantic_support + .15 * support + .10 * scene["scores"]["requested_role_match"] - contradiction * .55),
                }, "competition_group_id": f"{frame.get('id')}:{requested}", "status": "conflict" if contradiction else "new", "weak_steps": 0,
                "selection_reason": scene.get("selection_reason", ""),
            }
            if current:
                current["sources"].append(scene["id"])
                if candidate["scores"]["total"] > current["scores"]["total"]:
                    candidate["sources"] = current["sources"]
                    found[lemma] = candidate
            else:
                found[lemma] = candidate
        return sorted(found.values(), key=lambda item: (-item["scores"]["total"], item["lemma"]))

    @staticmethod
    def _result_type(scenes: Iterable[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> str:
        labels = {scene["result_type"] for scene in scenes}
        if "FULL_HIT" in labels:
            return "FULL_HIT"
        if candidates and any(item["status"] != "conflict" for item in candidates):
            return "ROLE_HIT"
        if "CONFLICT_HIT" in labels:
            return "CONFLICT_HIT"
        if "PARTIAL_HIT" in labels:
            return "PARTIAL_HIT"
        if "WEAK_HIT" in labels:
            return "WEAK_HIT"
        return "NO_HIT"

    @staticmethod
    def _energy(scenes: Iterable[Dict[str, Any]]) -> float:
        values = [scene["scores"]["total_score"] for scene in scenes]
        return round(sum(values) / max(1, len(values)), 4)

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "inertia": .55, "role_force": .25, "action_force": .20, "object_force": .15,
            "source_force": .10, "resonance_force": .12, "contradiction_force": .30,
            "competition_force": .08, "activation_threshold": .18, "retention_threshold": .20,
            "eviction_after_steps": 2, "max_steps": 5, "confidence_threshold": .88,
            "winner_gap_threshold": .15,
        }

    @staticmethod
    def _reasons(candidate: Dict[str, Any]) -> List[str]:
        scores = candidate["scores"]
        reasons = [f"совместимость с ролью {candidate['target_role']}"]
        if scores["action_compatibility"]:
            reasons.append("поддержка действия")
        if scores["object_compatibility"]:
            reasons.append("поддержка объекта")
        if scores["contradiction"]:
            reasons.append("противоречие отрицанием")
        return reasons

    @staticmethod
    def _winner(state: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        active = [item for item in state["candidates"] if item["status"] not in {"evicted", "conflict"}]
        if not active:
            return None
        first, second = active[0], active[1] if len(active) > 1 else None
        gap = first["scores"]["total"] - (second["scores"]["total"] if second else 0.0)
        if first["scores"]["total"] >= config["confidence_threshold"] or gap >= config["winner_gap_threshold"]:
            return first
        return None

    def _resolve(self, state: Dict[str, Any], winner: Dict[str, Any], step: int) -> None:
        scene = state["query_scene"]
        role = winner["target_role"]
        for slot in scene["slots"]:
            if slot["role"] == role and slot["status"] == "empty":
                slot.update({"status": "RESOLVED", "lemma": winner["lemma"], "surface": winner["surface"], "value": {"lemma": winner["lemma"], "surface": winner["surface"], "concept_id": winner["concept_id"], "preposition": winner.get("preposition", "")}, "confidence": winner["scores"].get("answer_confidence", winner["scores"]["total"]), "resolved_at_step": step})
        scene["status"] = "RESOLVED"
        for role_search in state.get("role_searches", []):
            if role_search.get("target_role") == role:
                role_search["selected_role_candidate"] = deepcopy(winner)
        full = any(item["result_type"] == "FULL_HIT" for item in state["memory_scenes"])
        mode = "exact" if full else "probable"
        state["sentence_plan"] = self._sentence_plan(state, winner, step)
        semantic_confidence = min((float(item.get("candidate_bridges", [{}])[0].get("confidence", item.get("selected_candidate", {}).get("scores", {}).get("semantic_total", 1.0))) for item in state.get("unknown_token_searches", []) if item.get("candidate_bridges")), default=1.0)
        source_confidence = float(winner["scores"].get("source_confidence", winner["scores"].get("total", 0.0)))
        role_evidence = float(winner["scores"].get("evidence_confidence", winner["scores"].get("total", 0.0)))
        answer_confidence = min(float(winner["scores"].get("total", 0.0)), semantic_confidence, source_confidence, role_evidence, 1.0)
        winner["scores"]["semantic_confidence"] = semantic_confidence
        winner["scores"]["answer_confidence"] = answer_confidence
        answer = {
            "query": state["query_frame"]["source_text"], "answer_mode": mode, "resolved_role": role,
            "resolved_value": winner["surface"], "confidence": answer_confidence,
            "supporting_scenes": winner["sources"], "surface_answer": None,
            "full_surface_answer": None, "status": "PLANNING",
        }
        if state.get("pipeline"):
            state["pipeline"]["memory_search"] = {"status": "ROLE_HIT", "candidate_count": len(state.get("candidates", []))}
            state["pipeline"]["query_scene"] = {"status": "RESOLVED"}
            state["pipeline"]["vibration"] = {"status": "FINISHED", "current_step": step}
            state["pipeline"]["sentence_planning"] = {"status": "READY"}
            state["pipeline"]["morphology_generation"] = {"status": "WAITING"}
            state["pipeline"]["answer"] = {"status": "PLANNING"}
            state["hive"]["reasoning_step"] = step
        state["answer"] = answer

    def _sentence_plan(self, state: Dict[str, Any], winner: Dict[str, Any], step: int) -> Dict[str, Any]:
        role = winner["target_role"]
        slot = {
            "order": 0, "role": role, "slot_id": f"slot-{role}", "lemma": winner["lemma"],
            "surface": winner["surface"], "preposition": winner.get("preposition", ""),
            "requested_features": {"case": "loct", "number": "sing"} if role == "location" else {},
            "observed_features": deepcopy(winner.get("grammatical_features", {})),
            "source_type": "known_word_form",
            "form_provenance": deepcopy(winner.get("form_provenance", {})),
        }
        return {
            "id": f"sentence-plan-{uuid.uuid4().hex[:12]}", "status": "READY",
            "answer_style": "short", "query_frame_id": state["query_frame"]["id"],
            "query_scene_id": state["query_scene"]["id"], "created_for_surface": state["created_for_surface"],
            "created_at_step": step, "slots": [slot],
            "lexical_preference": {"mode": "preserve_user_local_lemma", "fallback": "global_semantic_lemma"},
        }

    def generate_resolved_answer(self, hive_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            state = self._load(conn, hive_id)
            if not state or (state.get("query_scene") or {}).get("status") != "RESOLVED" or not state.get("sentence_plan"):
                raise ValueError("SentencePlan requires a resolved QueryScene")
            winner = next((item for item in state.get("candidates", []) if item.get("status") == "winner"), None)
            if not winner:
                raise ValueError("resolved role has no winner")
            role = winner["target_role"]
            plan = deepcopy(state["sentence_plan"])
            selected = winner["surface"]
            trace = {
                "lemma": winner["lemma"],
                "requested_case": "loct" if role == "location" else None,
                "observed_features": deepcopy(winner.get("grammatical_features", {})),
                "selected_form": selected,
                "source_type": "known_word_form",
                "selection_mode": "reuse_observed_training_form",
                "form_provenance": deepcopy(winner.get("form_provenance", {})),
            }
            plan["status"] = "FINISHED"
            plan["slots"][0]["selected_surface"] = selected
            bridge = next((item for item in state.get("unknown_token_searches", []) if item.get("query_role") == "object"), None)
            roles = state.get("query_frame", {}).get("roles", {})
            source_scene = self._supporting_scene(state, winner)
            source_roles = (source_scene or {}).get("roles", {})
            answer_roles: Dict[str, Dict[str, Any]] = {}
            role_sources: Dict[str, str] = {}
            for name in ANSWER_ROLE_ORDER:
                query_value = roles.get(name, {})
                source_value = source_roles.get(name, {})
                if name == role:
                    value = {**source_value, **winner}
                    role_source = "resolved_candidate"
                elif query_value.get("status") == "fixed" and query_value.get("surface"):
                    value = query_value
                    role_source = "query_frame"
                elif name == "action" and source_value.get("surface"):
                    value = source_value
                    role_source = "memory_scene"
                else:
                    continue
                answer_roles[name] = value
                role_sources[name] = role_source
            short = f"{winner.get('preposition', '')} {selected}".strip().capitalize() + "."
            full_plan = deepcopy(plan)
            full_plan["id"] = f"sentence-plan-full-{uuid.uuid4().hex[:12]}"
            full_plan["answer_style"] = "full"
            full_plan["source_scene_id"] = (source_scene or {}).get("id")
            full_plan["source_scene_text"] = (source_scene or {}).get("source_text")
            full_plan["slots"] = []
            for name in ANSWER_ROLE_ORDER:
                value = answer_roles.get(name)
                if not value or not value.get("surface"):
                    continue
                requested_features = {"case": "loct", "number": "sing"} if name == role and name == "location" else {}
                full_plan["slots"].append({
                    "order": len(full_plan["slots"]), "role": name,
                    "slot_id": f"slot-{name}", "lemma": value.get("lemma"),
                    "local_lemma": value.get("lemma"),
                    "semantic_lemma": bridge.get("selected_candidate", {}).get("candidate_lexeme") if name == "object" and bridge else value.get("lemma"),
                    "surface": value.get("surface"), "preposition": value.get("preposition", ""),
                    "semantic_function": value.get("semantic_function"),
                    "requested_features": requested_features,
                    "observed_features": deepcopy(value.get("grammatical_features", {})),
                    "source_type": "query_surface" if role_sources[name] == "query_frame" else "known_word_form",
                    "form_provenance": deepcopy(value.get("form_provenance", {})),
                    "context_source": role_sources[name],
                })
            full = " ".join(
                part
                for slot in full_plan["slots"]
                for part in (slot.get("preposition", ""), slot.get("surface", ""))
                if part
            ).capitalize() + "."
            answer = state["answer"]
            confidence = float(winner["scores"].get("answer_confidence", winner["scores"]["total"]))
            validation = self.reverse_validate(state, full, answer_roles)
            answer.update({"status": "RESOLVED" if validation["status"] == "PASSED" else "BUILD_FAILED", "answer_mode": answer.get("answer_mode") if answer.get("answer_mode") not in {"pending", "unknown"} else "probable", "surface_answer": short, "full_surface_answer": full, "confidence": confidence, "short": {"surface": short, "status": "RESOLVED" if validation["status"] == "PASSED" else "BUILD_FAILED"}, "full": {"surface": full, "status": "RESOLVED" if validation["status"] == "PASSED" else "BUILD_FAILED"}})
            score_breakdown = self._score_breakdown()
            row = conn.execute("SELECT cloud_id FROM lexemes WHERE lemma=? LIMIT 1", (winner["lemma"],)).fetchone()
            candidate = {
                "id": f"generation-candidate-{uuid.uuid4().hex[:12]}", "sentence_plan_id": plan["id"],
                "sentence_slot_id": plan["slots"][0]["slot_id"], "candidate_text": selected,
                "source_lexeme_cloud_id": int(row["cloud_id"]) if row else None, "source_type": "known_word_form", "generation_required": False,
                "requested_features": plan["slots"][0].get("requested_features", {}),
                "observed_features": plan["slots"][0].get("observed_features", {}),
                "form_provenance": plan["slots"][0].get("form_provenance", {}),
                "status": "SELECTED",
                **score_breakdown,
            }
            state.update({"sentence_plan": plan, "full_sentence_plan": full_plan, "answer_context_roles": answer_roles, "morphology_trace": [trace], "generation_candidates": [candidate], "reverse_validation": validation, "selected_surface": short})
            state.setdefault("reasoning_trace", {}).setdefault("stages", []).append({
                "id": "answer-assembly", "stage": "ANSWER_ASSEMBLY",
                "status": "RESOLVED" if validation["status"] == "PASSED" else "FAILED",
                "output": {"source_scene": self._scene_trace(source_scene) if source_scene else None, "short_plan": deepcopy(plan), "full_plan": deepcopy(full_plan), "short_answer": short, "full_answer": full, "reverse_validation": deepcopy(validation)},
            })
            state["pipeline"]["sentence_planning"] = {"status": "FINISHED"}
            state["pipeline"]["morphology_generation"] = {"status": "FINISHED"}
            state["pipeline"]["answer"] = {"status": "RESOLVED" if validation["status"] == "PASSED" else "REBUILDING"}
            if validation["status"] != "PASSED":
                state["display_status"] = "ANSWER_NEEDS_REBUILD"
            now = utcnow()
            conn.execute("DELETE FROM hive_generation_candidates WHERE hive_id=?", (hive_id,))
            if row:
                breakdown = score_breakdown["score_breakdown"]
                conn.execute("""INSERT INTO hive_generation_candidates
                    (hive_id, source_lexeme_cloud_id, candidate_text, requested_features_json,
                     applied_patterns_json, character_sequence_json, score_total, score_semantic,
                     score_grammar, score_pattern, score_orthography, score_context,
                     reverse_validation_score, status, provenance_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, '[]', ?, ?, ?, ?, ?, ?, ?, ?, 'SELECTED', ?, ?, ?)""",
                    (hive_id, int(row["cloud_id"]), selected, encode(plan["slots"][0].get("requested_features", {})), encode(list(selected)), score_breakdown["score_total"], breakdown["semantic"]["value"], breakdown["grammar"]["value"], breakdown["pattern"]["value"], breakdown["orthography"]["value"], breakdown["context"]["value"], breakdown["reverse_validation"]["value"], encode({"sentence_plan_id": plan["id"], "source_type": "known_word_form", "generation_required": False, "score_breakdown": breakdown}), now, now))
            self._save(conn, hive_id, state)
            return state

    @staticmethod
    def _score_breakdown() -> Dict[str, Any]:
        values = {"semantic": 1.0, "grammar": 1.0, "pattern": 1.0, "orthography": 1.0, "context": 1.0, "reverse_validation": 1.0}
        weights = {"semantic": .25, "grammar": .25, "pattern": .15, "orthography": .15, "context": .10, "reverse_validation": .10}
        breakdown = {name: {"value": value, "weight": weights[name], "contribution": round(value * weights[name], 6)} for name, value in values.items()}
        return {"score_breakdown": breakdown, "score_total": round(sum(item["contribution"] for item in breakdown.values()), 6)}

    @staticmethod
    def _supporting_scene(state: Dict[str, Any], winner: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        source_ids = [winner.get("primary_source_id"), *winner.get("sources", [])]
        return next(
            (scene for source_id in source_ids for scene in state.get("memory_scenes", []) if scene.get("id") == source_id),
            None,
        )

    @staticmethod
    def reverse_validate(
        state: Dict[str, Any], surface: str, answer_roles: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        text = str(surface or "").casefold()
        query_roles = state.get("query_frame", {}).get("roles", {})
        roles = answer_roles or query_roles
        scene = state.get("query_scene") or {}
        checks: Dict[str, bool] = {}
        errors: List[Dict[str, Any]] = []
        expected = {
            "object": roles.get("object", {}).get("surface") or roles.get("object", {}).get("lemma"),
            "modal": roles.get("modal", {}).get("surface") or roles.get("modal", {}).get("lemma"),
            "action": roles.get("action", {}).get("surface") or roles.get("action", {}).get("lemma"),
        }
        requested_role = state.get("query_frame", {}).get("requested_role")
        resolved = next((slot for slot in scene.get("slots", []) if slot.get("status", "").upper() == "RESOLVED" and slot.get("role") == requested_role), None)
        resolved_value = resolved.get("value", {}) if resolved else {}
        location = resolved_value if requested_role == "location" else roles.get("location", {})
        checks["object_preserved"] = not expected["object"] or str(expected["object"]).casefold() in text
        checks["modal_preserved"] = not expected["modal"] or str(expected["modal"]).casefold() in text
        checks["action_preserved"] = not expected["action"] or str(expected["action"]).casefold() in text
        location_surface = location.get("surface") if isinstance(location, dict) else None
        location_lemma = location.get("lemma") if isinstance(location, dict) else None
        checks["location_preserved"] = not location_surface or str(location_surface).casefold() in text or bool(location_lemma and str(location_lemma).casefold()[:4] in text)
        resolved_surface = resolved_value.get("surface") if isinstance(resolved_value, dict) else None
        resolved_lemma = resolved_value.get("lemma") if isinstance(resolved_value, dict) else None
        checks["resolved_role_preserved"] = not resolved_surface or str(resolved_surface).casefold() in text or bool(resolved_lemma and str(resolved_lemma).casefold()[:4] in text)
        checks["grammar_valid"] = bool(text.strip()) and text.rstrip().endswith(".")
        if expected["modal"] and not checks["modal_preserved"]:
            errors.append({"type": "MISSING_MODAL", "expected": expected["modal"], "semantic_function": roles.get("modal", {}).get("semantic_function")})
        if expected["action"] and not checks["action_preserved"]:
            errors.append({"type": "MISSING_ACTION", "expected": expected["action"]})
        if expected["object"] and not checks["object_preserved"]:
            errors.append({"type": "MISSING_OBJECT", "expected": expected["object"]})
        if not checks["location_preserved"]:
            errors.append({"type": "MISSING_LOCATION", "expected": location_surface or location_lemma})
        if not checks["resolved_role_preserved"]:
            errors.append({"type": "MISSING_RESOLVED_ROLE", "role": requested_role, "expected": resolved_surface or resolved_lemma})
        semantic_checks = [checks["object_preserved"], checks["modal_preserved"], checks["action_preserved"], checks["location_preserved"], checks["resolved_role_preserved"]]
        score = sum(semantic_checks) / max(1, len(semantic_checks))
        if not checks["grammar_valid"]:
            score = min(score, 0.75)
        return {"status": "PASSED" if not errors and all(checks.values()) else "FAILED", "score": round(score, 4), "checks": checks, "errors": errors}

    @staticmethod
    def _empty_answer(frame: Dict[str, Any], result_type: str) -> Dict[str, Any]:
        role = frame.get("requested_role")
        mode = "partial" if result_type in {"PARTIAL_HIT", "CONFLICT_HIT"} else "unknown"
        return {
            "query": frame["source_text"], "answer_mode": mode, "resolved_role": role,
            "resolved_value": None, "confidence": 0.0, "supporting_scenes": [],
            "surface_answer": None, "status": "PENDING",
            "status_message": f"В памяти нет точного указания для роли «{ROLE_LABELS.get(role or '', role or '?')}»." if role else "",
        }

    @staticmethod
    def _scene_trace(scene: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "scene_id": scene.get("id"), "source_text": scene.get("source_text"),
            "result_type": scene.get("result_type"), "matched_roles": scene.get("matched_roles", []),
            "mismatched_roles": scene.get("mismatched_roles", []), "selection_reason": scene.get("selection_reason", ""),
            "scores": deepcopy(scene.get("scores", {})),
        }

    @staticmethod
    def _candidate_trace(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "candidate_id": candidate.get("id"), "lemma": candidate.get("lemma"), "surface": candidate.get("surface"),
            "target_role": candidate.get("target_role"), "status": candidate.get("status"),
            "sources": list(candidate.get("sources", [])), "selection_reason": candidate.get("selection_reason", ""),
            "scores": deepcopy(candidate.get("scores", {})),
        }

    @staticmethod
    def _working_state(metadata: str) -> Dict[str, Any]:
        return json.loads(metadata or "{}").get("query_working_memory", {})

    def _load(self, conn: Any, hive_id: str) -> Dict[str, Any]:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        if not row:
            raise KeyError(hive_id)
        return self._working_state(row["metadata_json"])

    def _save(self, conn: Any, hive_id: str, state: Dict[str, Any]) -> None:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        metadata = json.loads(row["metadata_json"] or "{}") if row else {}
        metadata["query_working_memory"] = state
        conn.execute("UPDATE hives SET metadata_json=?, updated_at=? WHERE id=?", (encode(metadata), utcnow(), hive_id))

    def _decorate_state(self, conn: Any, hive_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        state = deepcopy(state)
        hive_row = conn.execute("SELECT * FROM hives WHERE id=?", (hive_id,)).fetchone()
        cells = [dict(row) for row in conn.execute(
            """SELECT hc.*, c.canonical_name AS lemma, p.local_activation AS energy, p.x, p.y
            FROM hive_cells hc JOIN clouds c ON c.id=hc.dominant_cloud_id
            JOIN cloud_placements p ON p.id=hc.hive_placement_id WHERE hc.hive_id=? ORDER BY hc.id""", (hive_id,)
        )]
        for item in cells:
            item["metadata"] = decode(item.get("metadata_json"), {})
        active = [item for item in cells if item.get("component_class") in {"semantic_bridge", "role_candidate", "reasoning_support"}]
        memory_sources = [item for item in cells if item.get("component_class") == "memory_source"]
        raw_sum = sum(float(item.get("energy") or item.get("local_activation") or 0) for item in cells)
        reasoning_sum = sum(float(item.get("energy") or item.get("local_activation") or 0) for item in active)
        energy = {
            "raw_sum": round(raw_sum, 6), "all_cells_average": round(raw_sum / len(cells), 6) if cells else 0.0,
            "reasoning_cells_sum": round(reasoning_sum, 6), "reasoning_cells_average": round(reasoning_sum / len(active), 6) if active else 0.0,
            "memory_sources_average": round(sum(float(item.get("energy") or item.get("local_activation") or 0) for item in memory_sources) / len(memory_sources), 6) if memory_sources else 0.0,
            "active_reasoning_cells": len(active), "active_memory_sources": len(memory_sources), "calculation_version": 1,
        }
        capacity = int((hive_row["capacity"] or hive_row["max_cells"]) if hive_row else 24)
        state["hive"] = {**state.get("hive", {}), "id": hive_id, "space_id": int(hive_row["space_id"]) if hive_row and hive_row["space_id"] is not None else None, "status": hive_row["status"] if hive_row else "ACTIVE", "query_text": hive_row["query_text"] if hive_row else state.get("created_for_surface", ""), "intent": (state.get("intent_classification") or {}).get("intent", state.get("query_frame", {}).get("intent")), "capacity": {"max_working_cells": capacity, "working_cells": len(active), "memory_sources": len(memory_sources), "inspection_projections": len(state.get("inspection_projections", [])), "search_hits": len(state.get("search_hits", [])), "total_physical_nodes": len((state.get("dynamics") or {}).get("nodes", [])), "total_placements": len(cells)}, "reasoning_step": int(hive_row["reasoning_step"] or 0) if hive_row else 0, "energy": energy, "pipeline": state.get("pipeline", {})}
        state["hive"]["query"] = {"original_text": state.get("created_for_surface", state.get("query_frame", {}).get("source_text", "")), "query_frame_id": state.get("query_frame", {}).get("id"), "query_scene_id": (state.get("query_scene") or {}).get("id")}
        state["hive"].update({
            "active_query_session_id": state.get("active_query_session_id"),
            "active_query_text": state.get("query_frame", {}).get("source_text", state.get("created_for_surface", "")),
            "display_status": state.get("display_status"),
        })
        state["active_query"] = {
            "text": state.get("query_frame", {}).get("source_text", state.get("created_for_surface", "")),
            "query_frame_id": state.get("query_frame", {}).get("id"),
            "query_scene_id": (state.get("query_scene") or {}).get("id"),
            "query_session_id": state.get("active_query_session_id"),
            "status": "ANSWER_READY" if state.get("answer", {}).get("status") == "RESOLVED" else (state.get("query_scene") or {}).get("status", "ACTIVE"),
        }
        state["hive"]["local_resonance"] = state.get("local_resonance")
        state["cells"] = cells
        state["memory_sources"] = [item for item in state.get("memory_sources", []) if item.get("query_frame_id") in {None, state.get("query_frame_id")}]
        state["capacity"] = state["hive"]["capacity"]
        if state.get("display_status") in {"CONVERSATIONAL_NO_MATCH", "SMALL_TALK_DETECTED"}:
            state["display_status"] = state["display_status"]
        elif state.get("answer", {}).get("status") == "BUILD_FAILED" or state.get("reverse_validation", {}).get("status") == "FAILED":
            state["display_status"] = "ANSWER_NEEDS_REBUILD"
        elif state.get("answer", {}).get("status") == "RESOLVED":
            state["display_status"] = "ANSWER_READY"
        elif (state.get("query_scene") or {}).get("status") == "RESOLVED":
            state["display_status"] = "ROLE_RESOLVED"
        elif state.get("vibration", {}).get("status") == "RUNNING":
            state["display_status"] = "HIVE_REASONING"
        elif state.get("candidates"):
            state["display_status"] = "ROLE_CANDIDATES_FOUND"
        else:
            state["display_status"] = "MISS"
        state["hive"]["display_status"] = state["display_status"]
        state["active_query"]["status"] = state["display_status"]
        state["hive_structure"] = {
            "placements": {"working_cells": len(active), "memory_sources": len(memory_sources), "total": len(cells)},
            "working_items": [
                {"type": item.get("component_class"), "label": (f"{item.get('metadata', {}).get('bridge', {}).get('unknown_token', {}).get('surface', '')} → {item.get('metadata', {}).get('bridge', {}).get('global_candidate', {}).get('lexeme', '')}" if item.get("component_class") == "semantic_bridge" else item.get("lemma", ""))}
                for item in active
            ],
            "sources": [{"type": "memory_scene", "label": item.get("source_text", "")} for item in state.get("memory_sources", [])],
            "selected_structure_target": state.get("selected_structure_target"),
        }
        state["stats"] = {"nodes": len(active), "cells": len(active), "components": len(active), "working_cells": len(active), "memory_sources": len(memory_sources), "inspection_projections": len(state.get("inspection_projections", [])), "total_placements": len(cells), "energy": energy}
        return state
