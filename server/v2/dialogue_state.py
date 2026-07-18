"""Persistent local dialogue state and reference resolution."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import fields
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .language.models import (
    Actuality,
    ClauseMode,
    CommitmentStatus,
    DialogueActType,
    DialogueState,
    LanguageAnalysis,
)
from .repository import decode, encode, utcnow


PRONOUN_LEMMAS = {"он", "она", "оно", "они", "его", "её", "их"}
TOPIC_ROLES = ("object", "agent", "location", "destination", "source", "action")
COMPACT_ROLE_KEYS = {
    "status",
    "value",
    "required",
    "question_word",
    "lemma",
    "surface",
    "normalized",
    "preposition",
    "role",
    "semantic_function",
    "lexeme_cloud_id",
    "word_form_cloud_id",
    "part_of_speech",
    "grammatical_features",
    "source",
    "entity_id",
    "index",
}


def _stable_id(prefix: str, *parts: object) -> str:
    key = "|".join(str(part) for part in parts)
    return f"{prefix}-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:20]}"


def _state_from_dict(value: Mapping[str, Any], conversation_id: str) -> DialogueState:
    allowed = {item.name for item in fields(DialogueState)}
    payload = {key: deepcopy(item) for key, item in value.items() if key in allowed}
    payload["conversation_id"] = conversation_id
    return DialogueState(**payload)


def _compact_role(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: deepcopy(item)
        for key, item in value.items()
        if key in COMPACT_ROLE_KEYS
    }


def _compact_query_frame(
    value: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    roles = {
        str(role): _compact_role(item)
        for role, item in (value.get("roles") or {}).items()
        if isinstance(item, Mapping)
    }
    exclusions = {
        str(role): [
            _compact_role(item)
            for item in items
            if isinstance(item, Mapping)
        ]
        for role, items in (value.get("excluded_roles") or {}).items()
    }
    correction = value.get("correction") or {}
    compact_correction = {
        key: deepcopy(item)
        for key, item in correction.items()
        if key in {
            "target",
            "target_query_frame_id",
            "target_role",
            "old_value",
            "new_value",
            "preserved_roles",
            "replacement_query_frame_id",
            "status",
        }
    }
    return {
        key: deepcopy(value.get(key))
        for key in (
            "id",
            "query_type",
            "question_word",
            "requested_role",
            "semantic_requested_role",
            "missing_role",
            "requested_slot",
            "answer_slot_type",
            "resolution_status",
            "interpretation_status",
            "continuation_of",
            "inherited_roles",
        )
        if value.get(key) is not None
    } | {
        "roles": roles,
        "requested_role_hypotheses": [
            {
                key: deepcopy(item.get(key))
                for key in (
                    "role",
                    "confidence",
                    "source",
                    "source_type",
                    "evidence",
                )
                if item.get(key) is not None
            }
            for item in (value.get("requested_role_hypotheses") or [])[:3]
        ],
        "excluded_roles": exclusions,
        **({"correction": compact_correction} if compact_correction else {}),
    }


def _compact_answer(
    value: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    return {
        key: deepcopy(value.get(key))
        for key in (
            "status",
            "answer_mode",
            "resolved_role",
            "resolved_value",
            "resolved_values",
            "confidence",
            "supporting_scenes",
            "surface_answer",
            "full_surface_answer",
            "evidence_status",
            "source_scene_id",
            "query_session_id",
        )
        if value.get(key) is not None
    }


def _compact_clause(value: Mapping[str, Any]) -> Dict[str, Any]:
    predicates = value.get("predicate_hypotheses") or []
    selected_predicates = [
        {
            key: deepcopy(predicate.get(key))
            for key in (
                "token_index",
                "lemma",
                "part_of_speech",
                "selected",
                "embedded",
            )
            if predicate.get(key) is not None
        }
        for predicate in predicates
        if predicate.get("selected")
    ] or [
        {
            key: deepcopy(predicate.get(key))
            for key in (
                "token_index",
                "lemma",
                "part_of_speech",
                "selected",
                "embedded",
            )
            if predicate.get(key) is not None
        }
        for predicate in predicates[:1]
    ]
    return {
        key: deepcopy(value.get(key))
        for key in (
            "id",
            "mode",
            "actuality",
            "evidence_status",
            "polarity",
            "negation_scope",
            "modality",
            "completion_status",
            "temporal_anchor",
            "speaker",
            "quoted_speaker",
        )
        if value.get(key) is not None
    } | {
        "predicate_hypotheses": selected_predicates,
        "participants": [
            {
                key: deepcopy(participant.get(key))
                for key in (
                    "mention_id",
                    "lemma",
                    "role_hypotheses",
                )
                if participant.get(key) is not None
            }
            for participant in (value.get("participants") or [])
        ],
    }


class ReferenceResolver:
    @staticmethod
    def _compatible(
        pronoun_features: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> tuple[bool, Dict[str, float]]:
        candidate_features = candidate.get("grammatical_features") or {}
        evidence: Dict[str, float] = {}
        for feature in ("gender", "number"):
            expected = pronoun_features.get(feature)
            actual = candidate_features.get(feature)
            if expected and actual and expected != actual:
                return False, {}
            if expected and actual and expected == actual:
                evidence[f"morphology_{feature}"] = 0.82
        expected_animacy = pronoun_features.get("animacy")
        actual_animacy = candidate_features.get("animacy")
        if expected_animacy and actual_animacy and expected_animacy == actual_animacy:
            evidence["semantics_animacy"] = 0.68
        return True, evidence

    def candidates(
        self,
        analysis: LanguageAnalysis,
        state: DialogueState,
    ) -> Dict[int, List[Dict[str, Any]]]:
        result: Dict[int, List[Dict[str, Any]]] = {}
        focus = list(state.focus_stack)
        for token in analysis.tokens:
            if token.pos != "NPRO" and token.lemma.casefold() not in PRONOUN_LEMMAS:
                continue
            candidates: List[Dict[str, Any]] = []
            for distance, item in enumerate(reversed(focus)):
                compatible, evidence = self._compatible(token.features, item)
                if not compatible:
                    continue
                activation = float(item.get("activation", 0.5))
                inertia = float(item.get("inertia", 0.5))
                distance_support = max(0.18, 0.92 - 0.11 * distance)
                evidence.update({
                    "discourse": round(
                        0.45 * activation
                        + 0.25 * inertia
                        + 0.30 * distance_support,
                        6,
                    ),
                    "temporal_context": round(distance_support, 6),
                })
                candidates.append({
                    "id": item.get("id"),
                    "lemma": item.get("lemma"),
                    "surface": item.get("surface"),
                    "role": item.get("role"),
                    "grammatical_features": deepcopy(
                        item.get("grammatical_features") or {}
                    ),
                    "evidence": evidence,
                    "activation": activation,
                    "inertia": inertia,
                    "distance": distance,
                })
            candidates.sort(
                key=lambda item: (
                    -sum(float(value) for value in item["evidence"].values()),
                    item["distance"],
                    str(item.get("id") or ""),
                )
            )
            result[token.index] = candidates
        return result


class DialogueStateService:
    def __init__(self, repository: Any = None) -> None:
        self.repository = repository
        self.references = ReferenceResolver()

    def load(self, conversation_id: str, conn: Any = None) -> DialogueState:
        if not conversation_id:
            return DialogueState(conversation_id="")
        if conn is None:
            if self.repository is None:
                return DialogueState(conversation_id=conversation_id)
            with self.repository.transaction() as transaction:
                return self.load(conversation_id, transaction)
        row = conn.execute(
            "SELECT state_json FROM dialogue_states WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        if not row:
            return DialogueState(conversation_id=conversation_id)
        return _state_from_dict(
            decode(row["state_json"], {}),
            conversation_id,
        )

    def save(self, state: DialogueState, conn: Any = None) -> DialogueState:
        if not state.conversation_id:
            return state
        if conn is None:
            if self.repository is None:
                return state
            with self.repository.transaction() as transaction:
                return self.save(state, transaction)
        now = utcnow()
        conn.execute(
            """INSERT INTO dialogue_states
               (conversation_id,state_json,version,created_at,updated_at)
               VALUES(?,?,?,?,?)
               ON CONFLICT(conversation_id) DO UPDATE SET
                 state_json=excluded.state_json,
                 version=excluded.version,
                 updated_at=excluded.updated_at""",
            (
                state.conversation_id,
                encode(state.as_dict()),
                state.version,
                now,
                now,
            ),
        )
        return state

    @staticmethod
    def _focus_values(
        analysis: LanguageAnalysis,
        query_frame: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = []
        for role, value in (query_frame or {}).get("roles", {}).items():
            if not isinstance(value, dict) or value.get("status") == "empty":
                continue
            lemma = value.get("lemma") or value.get("lexeme")
            if not lemma:
                continue
            values.append({
                "id": _stable_id(
                    "focus",
                    analysis.utterance.id if analysis.utterance else "",
                    role,
                    lemma,
                ),
                "role": role,
                "lemma": lemma,
                "surface": value.get("surface") or lemma,
                "grammatical_features": deepcopy(
                    value.get("grammatical_features") or {}
                ),
                "source_utterance_id": (
                    analysis.utterance.id if analysis.utterance else None
                ),
                "activation": 1.0,
                "inertia": 0.78,
            })
        if values:
            return values
        for mention in analysis.mentions:
            values.append({
                "id": _stable_id(
                    "focus",
                    analysis.utterance.id if analysis.utterance else "",
                    getattr(mention, "start", 0),
                    getattr(mention, "lemma", ""),
                ),
                "role": "entity",
                "lemma": getattr(mention, "lemma", ""),
                "surface": getattr(mention, "surface", ""),
                "grammatical_features": deepcopy(
                    getattr(mention, "features", {}) or {}
                ),
                "source_utterance_id": (
                    analysis.utterance.id if analysis.utterance else None
                ),
                "activation": 1.0,
                "inertia": 0.72,
            })
        return values

    @staticmethod
    def _resolve_expected_clarification(
        state: DialogueState,
        analysis: LanguageAnalysis,
    ) -> None:
        pending = state.pending_clarification
        if not pending or pending.get("status") != "PENDING":
            return
        if any(
            act.act_type in {
                DialogueActType.QUESTION,
                DialogueActType.CORRECTION,
            }
            for act in analysis.dialogue_acts
        ):
            return
        mention_lemmas = {
            str(getattr(mention, "lemma", "") or "").casefold()
            for mention in analysis.mentions
        }
        selected = next(
            (
                candidate
                for candidate in pending.get("candidates") or []
                if str(candidate.get("lemma") or "").casefold()
                in mention_lemmas
            ),
            None,
        )
        if not selected:
            return
        resolved = {
            **deepcopy(pending),
            "status": "ANSWERED",
            "resolved_candidate": deepcopy(selected),
            "answered_by_utterance_id": (
                analysis.utterance.id if analysis.utterance else None
            ),
        }
        state.pending_questions = [
            item for item in state.pending_questions
            if item.get("id") != resolved["id"]
        ]
        state.pending_questions.append({
            "id": resolved["id"],
            "query_frame_id": None,
            "requested_role": "referent",
            "status": "ANSWERED",
            "payload": resolved,
        })
        for reference in state.unresolved_references:
            if reference.get("id") == resolved["id"]:
                reference.update({
                    "status": "RESOLVED",
                    "resolved_candidate": deepcopy(selected),
                })
        state.pending_clarification = None
        state.expected_response = None

    @staticmethod
    def _decay_focus(state: DialogueState) -> None:
        for item in state.focus_stack:
            item["activation"] = round(
                max(0.0, float(item.get("activation", 0.5)) * 0.82),
                6,
            )
            item["inertia"] = round(
                max(0.0, float(item.get("inertia", 0.5)) * 0.94),
                6,
            )

    @staticmethod
    def _topic_signature(values: Sequence[Mapping[str, Any]]) -> set[str]:
        return {
            str(item.get("lemma") or "").casefold()
            for item in values
            if item.get("role") in TOPIC_ROLES and item.get("lemma")
        }

    def _update_topic(
        self,
        state: DialogueState,
        values: Sequence[Dict[str, Any]],
        turn_index: int,
    ) -> None:
        signature = self._topic_signature(values)
        active_signature = set(
            (state.active_topic or {}).get("signature", [])
        )
        if not signature:
            return
        if state.active_topic and signature.isdisjoint(active_signature):
            archived = {
                **deepcopy(state.active_topic),
                "last_active_turn": turn_index - 1,
                "collapsed_state": {
                    "focus_stack": deepcopy(state.focus_stack),
                    "expected_response": deepcopy(state.expected_response),
                },
            }
            state.topic_history = [
                item for item in state.topic_history
                if item.get("id") != archived.get("id")
            ]
            state.topic_history.append(archived)
            state.topic_history = state.topic_history[-12:]
        returned = next(
            (
                item for item in reversed(state.topic_history)
                if signature & set(item.get("signature", []))
            ),
            None,
        )
        if returned and (
            not state.active_topic
            or returned.get("id") != state.active_topic.get("id")
        ):
            state.active_topic = {
                **deepcopy(returned),
                "returned_at_turn": turn_index,
            }
            collapsed = returned.get("collapsed_state") or {}
            restored_focus = collapsed.get("focus_stack") or []
            state.focus_stack = [
                *restored_focus,
                *state.focus_stack,
            ][-24:]
            return
        if not state.active_topic or signature.isdisjoint(active_signature):
            state.active_topic = {
                "id": _stable_id("topic", state.conversation_id, *sorted(signature)),
                "signature": sorted(signature),
                "main_entities": [
                    deepcopy(item) for item in values
                    if item.get("role") != "action"
                ],
                "main_events": [
                    deepcopy(item) for item in values
                    if item.get("role") == "action"
                ],
                "started_at_turn": turn_index,
                "last_active_turn": turn_index,
            }
        else:
            state.active_topic["signature"] = sorted(active_signature | signature)
            state.active_topic["last_active_turn"] = turn_index

    @staticmethod
    def _reference_clarification(
        candidates: Mapping[int, Sequence[Mapping[str, Any]]],
        analysis: LanguageAnalysis,
    ) -> Optional[Dict[str, Any]]:
        for token_index, alternatives in candidates.items():
            if len(alternatives) < 2:
                continue
            def score(item: Mapping[str, Any]) -> float:
                return sum(
                    float(value)
                    for value in (item.get("evidence") or {}).values()
                )
            first, second = alternatives[0], alternatives[1]
            if abs(score(first) - score(second)) >= 0.22:
                continue
            token = analysis.tokens[token_index]
            return {
                "id": _stable_id(
                    "clarification",
                    analysis.utterance.id if analysis.utterance else "",
                    token_index,
                ),
                "slot": "referent",
                "source_token_index": token_index,
                "surface": token.surface,
                "candidates": [deepcopy(first), deepcopy(second)],
                "status": "PENDING",
                "question": (
                    f"Неясно, к кому относится «{token.surface}»: "
                    f"к {first.get('surface') or first.get('lemma')} или "
                    f"к {second.get('surface') or second.get('lemma')}?"
                ),
            }
        return None

    def _add_commitments(
        self,
        state: DialogueState,
        analysis: LanguageAnalysis,
        *,
        supersede_previous: bool = True,
    ) -> None:
        if not analysis.utterance:
            return
        correction = any(
            act.act_type == DialogueActType.CORRECTION
            for act in analysis.dialogue_acts
        )
        superseded_commitment_id: Optional[str] = None
        if correction and supersede_previous:
            active = next(
                (
                    item for item in reversed(state.speaker_commitments)
                    if item.get("status") == CommitmentStatus.ACTIVE.value
                    and item.get("speaker_role") == analysis.utterance.speaker_role
                ),
                None,
            )
            if active:
                active["status"] = CommitmentStatus.SUPERSEDED.value
                active["updated_at"] = utcnow()
                superseded_commitment_id = active.get("id")
        for clause in analysis.clauses:
            if clause.mode not in {
                ClauseMode.ASSERTION,
                ClauseMode.DEFINITION,
                ClauseMode.REPORTED_SPEECH,
                ClauseMode.QUOTE,
            }:
                continue
            if clause.actuality != Actuality.ACTUAL:
                continue
            if not clause.predicate_hypotheses:
                continue
            now = utcnow()
            commitment = {
                "id": _stable_id(
                    "commitment",
                    state.conversation_id,
                    analysis.utterance.id,
                    clause.id,
                ),
                "conversation_id": state.conversation_id,
                "speaker_role": analysis.utterance.speaker_role,
                "source_utterance_id": analysis.utterance.id,
                "source_clause_id": clause.id,
                "interpretation_id": next(
                    (
                        item.id for item in analysis.hypotheses
                        if item.scope_id == clause.id
                        and item.selected
                        and item.hypothesis_type == "predicate"
                    ),
                    clause.id,
                ),
                "status": CommitmentStatus.ACTIVE.value,
                "supersedes_commitment_id": (
                    superseded_commitment_id
                    if correction and supersede_previous else None
                ),
                "created_at": now,
                "updated_at": now,
                "content": _compact_clause(clause.as_dict()),
            }
            state.speaker_commitments.append(commitment)
        active_commitments = [
            item for item in state.speaker_commitments
            if item.get("status") in {
                CommitmentStatus.ACTIVE.value,
                CommitmentStatus.CONFIRMED_IN_DIALOGUE.value,
                CommitmentStatus.DISPUTED.value,
            }
        ]
        historical_commitments = [
            item for item in state.speaker_commitments
            if item not in active_commitments
        ][-8:]
        state.speaker_commitments = [
            *historical_commitments,
            *active_commitments,
        ]

    @staticmethod
    def _apply_correction(
        state: DialogueState,
        analysis: LanguageAnalysis,
        query_frame: Dict[str, Any],
        previous_query_frame: Optional[Dict[str, Any]],
    ) -> None:
        if not previous_query_frame:
            return
        if not any(
            act.act_type == DialogueActType.CORRECTION
            for act in analysis.dialogue_acts
        ):
            return
        tokens = query_frame.get("tokens") or []
        surfaces = [
            str(item.get("normalized") or "").casefold()
            for item in tokens
        ]
        correction_scaffold = {
            "иметь",
            "спрашивать",
            "значить",
            "говорить",
        }
        explicit_question = bool(query_frame.get("requested_role"))
        if explicit_question:
            question_clause = next(
                (
                    clause for clause in query_frame.get("clauses") or []
                    if clause.get("mode") == "QUESTION"
                    and clause.get("predicate_hypotheses")
                ),
                None,
            )
            if question_clause:
                predicate = next(
                    (
                        item
                        for item in question_clause["predicate_hypotheses"]
                        if item.get("selected")
                    ),
                    question_clause["predicate_hypotheses"][0],
                )
                predicate_token = next(
                    (
                        item for item in tokens
                        if item.get("index") == predicate.get("token_index")
                    ),
                    None,
                )
                if predicate_token:
                    query_frame.setdefault("roles", {})["action"] = {
                        "status": "fixed",
                        **deepcopy(predicate_token),
                    }
        else:
            query_frame["requested_role"] = previous_query_frame.get(
                "requested_role"
            )
            query_frame["semantic_requested_role"] = previous_query_frame.get(
                "semantic_requested_role"
            )
            query_frame["missing_role"] = previous_query_frame.get(
                "missing_role"
            )
            query_frame["requested_slot"] = previous_query_frame.get(
                "requested_slot"
            )
            query_frame["requested_role_hypotheses"] = deepcopy(
                previous_query_frame.get("requested_role_hypotheses") or []
            )
            query_frame["query_type"] = previous_query_frame.get(
                "query_type",
                "role_question",
            )
            query_frame["question_word"] = previous_query_frame.get(
                "question_word"
            )
        current_roles = query_frame.setdefault("roles", {})
        previous_roles = previous_query_frame.get("roles") or {}
        for role, value in previous_roles.items():
            if role in current_roles:
                current = current_roles[role]
                if (
                    current.get("status") == "fixed"
                    and current.get("lemma")
                    and str(current.get("lemma")).casefold()
                    not in correction_scaffold
                    and not (
                        role == "agent"
                        and str(current.get("lemma")).casefold() == "я"
                    )
                ):
                    continue
            current_roles[role] = {
                **deepcopy(value),
                "source": "correction_inherited",
            }
        candidate = next(
            (
                item for item in reversed(tokens)
                if item.get("part_of_speech") in {"NOUN", "NPRO"}
                and str(item.get("lemma") or "").casefold() != "я"
            ),
            None,
        )
        corrected_role: Optional[str] = None
        old_value: Optional[Dict[str, Any]] = None
        new_value: Optional[Dict[str, Any]] = None
        if candidate:
            index = int(candidate.get("index", 0))
            previous_surface = surfaces[index - 1] if index > 0 else ""
            cue_is_reference = (
                "про" in surfaces
                or (
                    any(value.startswith("вид") for value in surfaces)
                    and "в" in surfaces
                )
            )
            if (
                previous_surface in {"в", "во", "на"}
                and not (
                    previous_surface == "в"
                    and index > 1
                    and surfaces[index - 1:index + 1] == ["в", "виду"]
                )
            ):
                corrected_role = "location"
            elif cue_is_reference and previous_roles.get("location"):
                corrected_role = "location"
            else:
                corrected_role = (
                    str(query_frame.get("requested_role") or "")
                    or "object"
                )
            old_value = deepcopy(previous_roles.get(corrected_role))
            new_value = {
                "status": "fixed",
                **deepcopy(candidate),
                "preposition": (
                    previous_surface
                    if previous_surface in {"в", "во", "на", "из", "от"}
                    else (
                        old_value.get("preposition", "")
                        if old_value else ""
                    )
                ),
                "source": "correction_explicit",
            }
            current_roles[corrected_role] = new_value
        requested_role = query_frame.get("requested_role")
        if requested_role:
            current_roles[requested_role] = {
                "status": "empty",
                "value": None,
                "required": True,
                "question_word": query_frame.get("question_word"),
            }
        if (
            current_roles.get("action", {}).get("lemma") in correction_scaffold
            and previous_roles.get("action")
        ):
            current_roles["action"] = {
                **deepcopy(previous_roles["action"]),
                "source": "correction_inherited",
            }
        if str(
            current_roles.get("agent", {}).get("lemma") or ""
        ).casefold() == "я":
            current_roles.pop("agent", None)
        query_frame["correction"] = {
            "target": "previous_query_frame",
            "target_query_frame_id": previous_query_frame.get("id"),
            "target_role": corrected_role,
            "old_value": _compact_role(old_value) if old_value else None,
            "new_value": _compact_role(new_value) if new_value else None,
            "preserved_roles": [
                role for role in previous_roles
                if role != corrected_role
            ],
            "replacement_query_frame_id": query_frame.get("id"),
            "status": "APPLIED",
        }
        if state.last_answer:
            state.last_answer["status"] = "SUPERSEDED_FOR_CURRENT_INTENT"

    def update(
        self,
        state: DialogueState,
        analysis: LanguageAnalysis,
        *,
        query_frame: Optional[Dict[str, Any]] = None,
        answer: Optional[Dict[str, Any]] = None,
    ) -> DialogueState:
        turn_index = analysis.utterance.turn_index if analysis.utterance else 0
        previous_query_frame = deepcopy(state.last_query_frame)
        self._resolve_expected_clarification(state, analysis)
        if query_frame is not None:
            self._apply_correction(
                state,
                analysis,
                query_frame,
                previous_query_frame,
            )
        self._decay_focus(state)
        reference_candidates = self.references.candidates(analysis, state)
        values = self._focus_values(analysis, query_frame)
        known = {
            (item.get("role"), str(item.get("lemma") or "").casefold())
            for item in values
        }
        state.focus_stack = [
            item for item in state.focus_stack
            if (
                item.get("role"),
                str(item.get("lemma") or "").casefold(),
            ) not in known
            and float(item.get("activation", 0.0)) >= 0.08
        ]
        state.focus_stack.extend(values)
        state.focus_stack = state.focus_stack[-24:]
        state.entity_candidates = [
            deepcopy(item) for item in state.focus_stack
            if item.get("role") != "action"
        ]
        self._update_topic(state, values, turn_index)
        clarification = self._reference_clarification(
            reference_candidates,
            analysis,
        )
        if clarification:
            state.pending_clarification = clarification
            state.unresolved_references.append({
                "utterance_id": (
                    analysis.utterance.id if analysis.utterance else None
                ),
                **deepcopy(clarification),
            })
            state.expected_response = {
                "type": "CLARIFICATION",
                "slot": "referent",
                "clarification_id": clarification["id"],
            }
        requested_role = (query_frame or {}).get("requested_role")
        is_user_turn = bool(
            analysis.utterance
            and analysis.utterance.speaker_role == "user"
        )
        if query_frame is not None and is_user_turn and answer is None:
            current_frame_id = query_frame.get("id")
            for pending in state.pending_questions:
                if (
                    pending.get("status") == "PENDING"
                    and pending.get("query_frame_id") != current_frame_id
                ):
                    pending["status"] = "SUPERSEDED"
                    pending["superseded_at_turn"] = turn_index
            state.expected_response = None
        if requested_role:
            pending = {
                "id": _stable_id(
                    "pending-question",
                    state.conversation_id,
                    (query_frame or {}).get("id"),
                ),
                "query_frame_id": (query_frame or {}).get("id"),
                "requested_role": requested_role,
                "status": "PENDING",
                "created_at_turn": turn_index,
            }
            state.pending_questions = [
                item for item in state.pending_questions
                if item.get("query_frame_id") != pending["query_frame_id"]
            ]
            state.pending_questions.append(pending)
            if not clarification:
                state.expected_response = {
                    "type": "ROLE_VALUE",
                    "role": requested_role,
                    "query_frame_id": pending["query_frame_id"],
                }
            state.last_query_frame = _compact_query_frame(query_frame)
        elif query_frame is not None and is_user_turn:
            state.last_query_frame = _compact_query_frame(query_frame)
        if answer:
            state.last_answer = _compact_answer(answer)
            supporting = answer.get("resolved_role")
            for pending in reversed(state.pending_questions):
                if (
                    pending.get("status") == "PENDING"
                    and pending.get("requested_role") == supporting
                ):
                    pending["status"] = "ANSWERED"
                    pending["answered_at_turn"] = turn_index
                    if (
                        state.expected_response
                        and state.expected_response.get("query_frame_id")
                        == pending.get("query_frame_id")
                    ):
                        state.expected_response = None
                    break
        for act in analysis.dialogue_acts:
            if act.act_type == DialogueActType.CONFIRMATION:
                confirmation = {
                    "id": _stable_id(
                        "confirmation",
                        state.conversation_id,
                        analysis.utterance.id if analysis.utterance else "",
                    ),
                    "source_utterance_id": (
                        analysis.utterance.id if analysis.utterance else None
                    ),
                    "target": deepcopy(state.last_answer or state.last_query_frame),
                }
                state.shared_confirmations.append(confirmation)
            if act.act_type == DialogueActType.CORRECTION and query_frame is not None:
                query_frame.setdefault("correction", {}).update({
                    "replacement_query_frame_id": query_frame.get("id"),
                    "status": "APPLIED",
                })
        query_correction = bool(
            (query_frame or {}).get("correction", {}).get(
                "target_query_frame_id"
            )
        )
        self._add_commitments(
            state,
            analysis,
            supersede_previous=not query_correction,
        )
        active_questions = [
            item for item in state.pending_questions
            if item.get("status") == "PENDING"
        ]
        historical_questions = [
            item for item in state.pending_questions
            if item.get("status") != "PENDING"
        ][-12:]
        state.pending_questions = [
            *historical_questions,
            *active_questions,
        ]
        state.shared_confirmations = state.shared_confirmations[-16:]
        state.unresolved_references = state.unresolved_references[-16:]
        return state

    def persist_interpretation(
        self,
        conn: Any,
        analysis: LanguageAnalysis,
        *,
        message_id: Optional[str] = None,
    ) -> None:
        if not analysis.utterance:
            return
        envelope = analysis.utterance
        conn.execute(
            """INSERT INTO utterances
               (id,conversation_id,turn_index,speaker_role,raw_text,
                normalized_text,received_at,language,source_type,parser_version,
                interpretation_status,message_id)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 interpretation_status=excluded.interpretation_status,
                 parser_version=excluded.parser_version,
                 message_id=COALESCE(excluded.message_id,utterances.message_id)""",
            (
                envelope.id,
                envelope.conversation_id,
                envelope.turn_index,
                envelope.speaker_role,
                envelope.raw_text,
                envelope.normalized_text,
                envelope.received_at,
                envelope.language,
                envelope.source_type,
                envelope.parser_version,
                envelope.interpretation_status.value,
                message_id,
            ),
        )
        for act in analysis.dialogue_acts:
            conn.execute(
                """INSERT OR REPLACE INTO dialogue_acts
                   (id,utterance_id,act_type,token_start,token_end,target_act_id,
                    addressee,confidence,evidence_json,alternatives_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    act.id,
                    act.utterance_id,
                    act.act_type.value,
                    act.token_start,
                    act.token_end,
                    act.target_act_id,
                    act.addressee,
                    act.confidence,
                    encode(act.evidence),
                    encode(act.alternatives),
                ),
            )
        for clause in analysis.clauses:
            item = clause.as_dict()
            conn.execute(
                """INSERT OR REPLACE INTO clauses
                   (id,utterance_id,sentence_index,parent_clause_id,token_start,
                    token_end,clause_type,relation_to_parent,
                    predicate_hypotheses_json,mode,actuality,evidence_status,
                    polarity,negation_scope_json,modality,completion_status,
                    temporal_anchor_json,speaker,quoted_speaker,surface,
                    evidence_json,alternatives_json,participants_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    clause.id,
                    clause.utterance_id,
                    clause.sentence_index,
                    None,
                    clause.token_start,
                    clause.token_end,
                    clause.clause_type,
                    item["relation_to_parent"],
                    encode(item["predicate_hypotheses"]),
                    item["mode"],
                    item["actuality"],
                    item["evidence_status"],
                    item["polarity"],
                    encode(item["negation_scope"]),
                    item["modality"],
                    item["completion_status"],
                    encode(item["temporal_anchor"]),
                    clause.speaker,
                    clause.quoted_speaker,
                    clause.surface,
                    encode(clause.evidence),
                    encode(clause.alternative_boundaries),
                    encode(clause.participants),
                ),
            )
        for clause in analysis.clauses:
            if clause.parent_clause_id:
                conn.execute(
                    "UPDATE clauses SET parent_clause_id=? WHERE id=?",
                    (clause.parent_clause_id, clause.id),
                )
        for relation in analysis.clause_relations:
            conn.execute(
                """INSERT OR REPLACE INTO clause_relations
                   (id,source_clause_id,target_clause_id,relation_type,
                    confidence,evidence_json)
                   VALUES(?,?,?,?,?,?)""",
                (
                    relation.id,
                    relation.source_clause_id,
                    relation.target_clause_id,
                    relation.relation_type.value,
                    relation.confidence,
                    encode(relation.evidence),
                ),
            )
        for hypothesis in analysis.hypotheses:
            item = hypothesis.as_dict()
            conn.execute(
                """INSERT OR REPLACE INTO interpretation_hypotheses
                   (id,scope_type,scope_id,hypothesis_type,value_json,status,
                    support_by_group_json,support,penalties_json,constraints_json,
                    unresolved_slots_json,stability_cycles,leader_margin,selected,
                    parser_version)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    hypothesis.id,
                    hypothesis.scope_type,
                    hypothesis.scope_id,
                    hypothesis.hypothesis_type,
                    encode(item["value"]),
                    item["status"],
                    encode(item["support_by_group"]),
                    item["support"],
                    encode(item["penalties"]),
                    encode(item["constraints"]),
                    encode(item["unresolved_slots"]),
                    item["stability_cycles"],
                    item["leader_margin"],
                    int(item["selected"]),
                    item["parser_version"],
                ),
            )
        for packet in analysis.evidence_packets:
            item = packet.as_dict()
            conn.execute(
                """INSERT OR REPLACE INTO interpretation_evidence
                   (id,origin,target_hypothesis_id,value_json,support,penalty,
                    evidence_type,independent_group,scope_type,scope_id,
                    source_token_start,source_token_end,source_object_id,
                    parser_version)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    packet.id,
                    packet.origin,
                    packet.target_hypothesis_id,
                    encode(item["value"]),
                    packet.support,
                    packet.penalty,
                    packet.evidence_type,
                    packet.independent_group,
                    packet.scope_type,
                    packet.scope_id,
                    packet.source_token_start,
                    packet.source_token_end,
                    packet.source_object_id,
                    packet.parser_version,
                ),
            )

    def persist_state_objects(self, conn: Any, state: DialogueState) -> None:
        conversation_id = state.conversation_id
        conn.execute(
            "DELETE FROM dialogue_focus_items WHERE conversation_id=?",
            (conversation_id,),
        )
        for rank, item in enumerate(reversed(state.focus_stack)):
            conn.execute(
                """INSERT OR REPLACE INTO dialogue_focus_items
                   (id,conversation_id,focus_rank,role,value_json,activation,
                    inertia,status,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    item.get("id") or _stable_id(
                        "focus", conversation_id, rank, item.get("lemma")
                    ),
                    conversation_id,
                    rank,
                    item.get("role"),
                    encode(item),
                    float(item.get("activation", 0.0)),
                    float(item.get("inertia", 0.0)),
                    "ACTIVE",
                    utcnow(),
                ),
            )
        for topic in [
            *state.topic_history,
            *([state.active_topic] if state.active_topic else []),
        ]:
            conn.execute(
                """INSERT INTO dialogue_topics
                   (id,conversation_id,status,topic_json,last_active_turn,updated_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     status=excluded.status,topic_json=excluded.topic_json,
                     last_active_turn=excluded.last_active_turn,
                     updated_at=excluded.updated_at""",
                (
                    topic["id"],
                    conversation_id,
                    "ACTIVE"
                    if state.active_topic
                    and topic["id"] == state.active_topic.get("id")
                    else "COLLAPSED",
                    encode(topic),
                    int(topic.get("last_active_turn", 0)),
                    utcnow(),
                ),
            )
        for question in state.pending_questions:
            conn.execute(
                """INSERT INTO dialogue_pending_questions
                   (id,conversation_id,query_frame_id,requested_role,status,
                    payload_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     status=excluded.status,payload_json=excluded.payload_json,
                     updated_at=excluded.updated_at""",
                (
                    question["id"],
                    conversation_id,
                    question.get("query_frame_id"),
                    question.get("requested_role"),
                    question.get("status", "PENDING"),
                    encode(question),
                    utcnow(),
                    utcnow(),
                ),
            )
        if state.pending_clarification:
            clarification = state.pending_clarification
            conn.execute(
                """INSERT INTO dialogue_pending_questions
                   (id,conversation_id,query_frame_id,requested_role,status,
                    payload_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     status=excluded.status,payload_json=excluded.payload_json,
                     updated_at=excluded.updated_at""",
                (
                    clarification["id"],
                    conversation_id,
                    None,
                    clarification.get("slot"),
                    clarification.get("status", "PENDING"),
                    encode(clarification),
                    utcnow(),
                    utcnow(),
                ),
            )
        for commitment in state.speaker_commitments:
            conn.execute(
                """INSERT INTO speaker_commitments
                   (id,conversation_id,speaker_role,source_utterance_id,
                    source_clause_id,interpretation_id,status,
                    supersedes_commitment_id,content_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     status=excluded.status,
                     supersedes_commitment_id=excluded.supersedes_commitment_id,
                     content_json=excluded.content_json,
                     updated_at=excluded.updated_at""",
                (
                    commitment["id"],
                    conversation_id,
                    commitment["speaker_role"],
                    commitment["source_utterance_id"],
                    commitment["source_clause_id"],
                    commitment["interpretation_id"],
                    commitment["status"],
                    commitment.get("supersedes_commitment_id"),
                    encode(commitment.get("content") or {}),
                    commitment.get("created_at") or utcnow(),
                    commitment.get("updated_at") or utcnow(),
                ),
            )

    def process(
        self,
        analysis: LanguageAnalysis,
        *,
        query_frame: Optional[Dict[str, Any]] = None,
        answer: Optional[Dict[str, Any]] = None,
        conn: Any = None,
        message_id: Optional[str] = None,
    ) -> DialogueState:
        conversation_id = (
            analysis.utterance.conversation_id if analysis.utterance else ""
        )
        state = self.load(conversation_id, conn)
        state = self.update(
            state,
            analysis,
            query_frame=query_frame,
            answer=answer,
        )
        correction = (query_frame or {}).get("correction") or {}
        if conn is not None:
            if correction.get("target_query_frame_id"):
                conn.execute(
                    """UPDATE query_frames SET status='SUPERSEDED'
                       WHERE id=?""",
                    (correction["target_query_frame_id"],),
                )
                conn.execute(
                    """UPDATE derived_answers
                       SET status='SUPERSEDED_FOR_CURRENT_INTENT'
                       WHERE id=(
                         SELECT id FROM derived_answers
                         WHERE conversation_id=?
                           AND status='DERIVED_ANSWER'
                         ORDER BY created_at DESC,id DESC LIMIT 1
                       )""",
                    (conversation_id,),
                )
            self.persist_interpretation(
                conn,
                analysis,
                message_id=message_id,
            )
            self.persist_state_objects(conn, state)
            self.save(state, conn)
        elif self.repository is not None:
            with self.repository.transaction() as transaction:
                if correction.get("target_query_frame_id"):
                    transaction.execute(
                        """UPDATE query_frames SET status='SUPERSEDED'
                           WHERE id=?""",
                        (correction["target_query_frame_id"],),
                    )
                    transaction.execute(
                        """UPDATE derived_answers
                           SET status='SUPERSEDED_FOR_CURRENT_INTENT'
                           WHERE id=(
                             SELECT id FROM derived_answers
                             WHERE conversation_id=?
                               AND status='DERIVED_ANSWER'
                             ORDER BY created_at DESC,id DESC LIMIT 1
                           )""",
                        (conversation_id,),
                    )
                self.persist_interpretation(
                    transaction,
                    analysis,
                    message_id=message_id,
                )
                self.persist_state_objects(transaction, state)
                self.save(state, transaction)
        return state
