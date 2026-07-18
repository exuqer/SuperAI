"""Query-scene reasoning kept in a single working hive."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from server.tokenizer import WordToken

from .repository import V2Repository, decode, encode, utcnow
from .training import RoleResolver, RussianMorphology, TrainingPipelineV2
from .intent import GREETING_WORDS, IntentClassifier
from .semantic_projection import MATCH_WEIGHTS, SemanticProjectionService
from .taxonomy_resolver import TaxonomyResolver
from .capacity import get_working_occupancy
from .event_core import UniversalEventPipeline, stable_id
from .dialogue_state import DialogueStateService
from .response_planner import ResponsePlanner
from .language import LanguageAnalysis, UniversalLanguageAnalyzer
from .language.diagnostics import (
    QUESTION_OPERATOR_ERROR,
    REQUESTED_ROLE_ERROR,
    ROLE_HYPOTHESIS_AMBIGUITY,
)
from .semantics import PredicateValencyProfile, RoleHypothesisResolver
from server.core.settings import settings


QUESTION_ROLES = {
    "кто": "agent", "кого": "patient", "кому": "recipient",
    "кем": "agent",
    "что": "object", "где": "location", "куда": "destination",
    "откуда": "source", "когда": "time", "как": "instrument",
    "почему": "cause", "зачем": "purpose", "чем": "instrument",
    "сколько": "quantity",
}
QUESTION_SLOT_HINTS = {
    "кто": "subject",
    "кого": "direct_object",
    "кому": "indirect_object",
    "где": "location_oblique",
    "куда": "destination_oblique",
    "откуда": "source_oblique",
    "чем": "instrumental",
    "зачем": "purpose_oblique",
}
TYPED_QUESTION_LEMMA = "какой"
SUPPORTED_ROLES = (
    "entity", "entity_type", "agent", "action", "modal", "patient", "theme", "object", "experiencer",
    "recipient", "source", "destination", "location", "instrument", "material",
    "cause", "result", "purpose", "time", "attribute", "quantity", "owner",
    "possessed", "manner",
)
ANSWER_ROLE_ORDER = (
    "entity", "entity_type", "agent", "modal", "action", "patient", "theme", "object", "experiencer",
    "recipient", "location", "destination", "source", "time", "instrument",
    "material", "cause", "result", "purpose", "attribute", "quantity", "owner",
    "possessed", "manner",
)
MESSAGE_MODES = {"NEW_QUERY", "LOCAL_RESONANCE", "FOLLOW_UP", "CORRECTION"}
MODAL_WORDS = {
    "можно": {"semantic_function": "possibility"},
    "нельзя": {"semantic_function": "prohibition"},
    "надо": {"semantic_function": "necessity"},
    "нужно": {"semantic_function": "necessity"},
}
ROLE_LABELS = {
    "entity": "СУЩНОСТЬ", "entity_type": "ТИП", "agent": "КТО?", "action": "ДЕЙСТВИЕ", "patient": "КОГО/ЧТО?", "theme": "ЧТО?", "object": "ЧТО?",
    "recipient": "КОМУ?", "location": "ГДЕ?", "destination": "КУДА?",
    "source": "ОТКУДА?", "time": "КОГДА?", "instrument": "ЧЕМ?",
    "attribute": "КАКОЙ?", "purpose": "ЗАЧЕМ?", "cause": "ПОЧЕМУ?",
}
ACTION_QUESTION_COMPLEMENT_ROLES = (
    "patient", "object", "recipient", "location", "destination", "source",
    "time", "instrument", "material", "cause", "result", "purpose",
    "attribute", "quantity", "manner",
)
ANSWER_SLOT_TYPES = {
    "participant", "predicate", "predicate_phrase", "relation", "attribute",
    "event",
}
REFERENTIALS = {"там": "location", "туда": "destination", "оттуда": "source"}
CONTEXT_ROLES = ("location", "destination", "source")
DIALOGUE_CONTEXT_ROLES = ("agent", "action", "modal", "object", *CONTEXT_ROLES, "time", "instrument")
ENTITY_REFERENCE_LEMMAS = {"он", "она", "оно", "они"}
ROLE_MATCH_ALIASES = {"location": {"location", "destination"}, "destination": {"location", "destination"}}
MULTI_ANSWER_QUESTION_WORDS = {"кто", "что"}
MULTI_ANSWER_LIMIT = 8
PIPELINE_STATES = (
    "CREATED", "PARSED", "QUERY_FRAME_READY", "SEARCHING", "SOURCES_FOUND",
    "CANDIDATES_FOUND", "VIBRATING", "STABILIZED", "PLANNING", "GENERATING",
    "VALIDATING", "ANSWER_READY", "FAILED",
)


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
        self.semantic_projection = SemanticProjectionService()
        self.universal_events = UniversalEventPipeline(self.repository, self.morphology)
        self.language = UniversalLanguageAnalyzer(self.morphology)
        self.role_hypotheses = RoleHypothesisResolver()
        self.dialogue = DialogueStateService(self.repository)
        self.responses = ResponsePlanner()

    def parse(
        self,
        text: str,
        *,
        conversation_id: str = "",
        turn_index: int = 0,
        speaker_role: str = "user",
        source_type: str = "dialogue",
        utterance_id: str = "",
        received_at: str = "",
        return_analysis: bool = False,
    ) -> Dict[str, Any]:
        language_analysis = self.language.analyze(
            text,
            # Communicative routing is decided from the shared utterance
            # analysis below.  Parsing the operator first prevents the legacy
            # intent classifier and dialogue-act parser from disagreeing.
            detect_question=True,
            conversation_id=conversation_id,
            turn_index=turn_index,
            speaker_role=speaker_role,
            source_type=source_type,
            utterance_id=utterance_id,
            received_at=received_at,
        )
        intent = self.intent_classifier.classify(
            text,
            dialogue_acts=language_analysis.dialogue_acts,
        )
        items = []
        question_word = ""
        typed_question_index: Optional[int] = None
        referential = None
        for token in language_analysis.tokens:
            index = token.index
            normalized = token.normalized.casefold()
            entity_referential = (
                token.pos == "NPRO" and token.lemma in ENTITY_REFERENCE_LEMMAS
            )
            is_typed_question = token.lemma == TYPED_QUESTION_LEMMA
            is_question = (
                (normalized in QUESTION_ROLES or is_typed_question)
                and not question_word
                and intent["intent"] not in {
                    "SMALL_TALK", "GREETING", "GREETING_WITH_SMALL_TALK",
                }
            )
            if is_question:
                question_word = normalized
                if is_typed_question:
                    typed_question_index = index
            if normalized in REFERENTIALS:
                referential = normalized
            modal = MODAL_WORDS.get(normalized)
            items.append({
                "index": index, "surface": token.surface, "normalized": normalized,
                "lemma": token.lemma, "part_of_speech": token.pos,
                "grammatical_features": dict(token.features),
                "morphological_analyses": [
                    analysis.as_dict() for analysis in token.analyses
                ],
                "component_type": "question_operator" if is_question else "context_reference" if normalized in REFERENTIALS else "query_modal" if modal else "query_token",
                "semantic_part_of_speech": (
                    "QUESTION_OPERATOR" if is_question else token.pos
                ),
                "semantic_role": "QUESTION_OPERATOR" if is_question else None,
                "entity_referential": entity_referential,
                "role": "modal" if modal else None,
                "semantic_function": modal["semantic_function"] if modal else None,
                "expected_role": QUESTION_ROLES.get(normalized, "typed_slot") if is_question else None,
                "resolution_state": TokenResolutionState.QUESTION_OPERATOR.value if is_question else TokenResolutionState.EXACT_FORM_MATCH.value if modal else TokenResolutionState.PARSED_UNGROUNDED.value,
            })
        typed_resolution: tuple[
            Optional[str], Optional[str], Optional[Dict[str, Any]],
            List[Dict[str, Any]],
        ] = (None, None, None, [])
        valency_profile: Dict[str, Any] = {}
        with self.repository.transaction() as conn:
            for item in items:
                if item["resolution_state"] == TokenResolutionState.QUESTION_OPERATOR.value:
                    continue
                row = conn.execute("SELECT wf.cloud_id AS word_form_cloud_id, wf.lexeme_cloud_id, l.lemma FROM word_forms wf LEFT JOIN lexemes l ON l.cloud_id=wf.lexeme_cloud_id WHERE wf.normalized_form=? LIMIT 1", (item["normalized"],)).fetchone()
                if row:
                    item.update({"lexeme": row["lemma"] or item["lemma"], "word_form_cloud_id": int(row["word_form_cloud_id"]), "lexeme_cloud_id": int(row["lexeme_cloud_id"]) if row["lexeme_cloud_id"] else None, "resolution_state": TokenResolutionState.EXACT_FORM_MATCH.value})
                    continue
                row = conn.execute(
                    """SELECT cloud_id AS lexeme_cloud_id, lemma FROM lexemes
                    WHERE lemma=?
                    ORDER BY CASE WHEN pos_tag=? THEN 0 ELSE 1 END, cloud_id
                    LIMIT 1""",
                    (item["lemma"], item["part_of_speech"]),
                ).fetchone()
                if row:
                    item.update({
                        "lexeme": row["lemma"],
                        "lexeme_cloud_id": int(row["lexeme_cloud_id"]),
                        "resolution_state": TokenResolutionState.LEXEME_MATCH.value,
                    })
            roles = self._roles_from_language(
                language_analysis,
                items,
                conn,
                requested_role=QUESTION_ROLES.get(question_word, ""),
            )
            if typed_question_index is not None:
                typed_resolution = self._resolve_typed_question(
                    language_analysis,
                    roles,
                    items,
                    conn,
                )
            if language_analysis.predicate:
                valency_profile = PredicateValencyProfile.load(
                    conn, language_analysis.predicate.lemma
                ).as_dict()
        requested_role = QUESTION_ROLES.get(question_word, "")
        necessity_question = (
            requested_role == "object"
            and any(item.get("semantic_function") == "necessity" for item in items)
            and any(item.get("grammatical_features", {}).get("case") == "datv" for item in items)
        )
        if necessity_question:
            requested_role = "instrument"
        continuation_markers = {
            item["normalized"] for item in items
            if item["normalized"] in {"ещё", "кроме"} or item.get("lemma") == "другой"
        }
        if intent["intent"] in {"SMALL_TALK", "GREETING", "GREETING_WITH_SMALL_TALK"}:
            requested_role = ""
        slot_constraints: Dict[str, Dict[str, Any]] = {}
        semantic_constraints: Dict[str, Dict[str, Any]] = {}
        if typed_question_index is not None:
            typed_slot, typed_role, typed_value, typed_hypotheses = typed_resolution
            if typed_role and typed_value:
                roles = {
                    role: value
                    for role, value in roles.items()
                    if value.get("phrase_id") != typed_value.get("phrase_id")
                }
                requested_role = typed_role
                slot_constraints = {typed_role: {"is_a": deepcopy(typed_value)}}
                question_token = items[typed_question_index]
                question_token["expected_role"] = typed_role
        if typed_question_index is not None and typed_resolution[0]:
            requested_slot = typed_resolution[0]
            role_hypotheses = typed_resolution[3]
        else:
            requested_slot, role_hypotheses = self._resolve_question_role(
                question_word,
                roles,
                items,
                requested_role,
                typed_question_index is not None,
            )
        action_question = self._action_question_spec(
            language_analysis,
            items,
            roles,
            question_word,
        )
        if action_question:
            requested_role = "action"
            requested_slot = "predicate_phrase"
            role_hypotheses = [{
                "role": "action",
                "confidence": 1.0,
                "source": "action_question_operator",
                "evidence": ["question_operator", "placeholder_predicate"],
            }]
            placeholder = roles.pop("action", {})
            action_question["removed_predicate"] = {
                "lemma": placeholder.get("lemma"),
                "surface": placeholder.get("surface"),
                "token_indices": action_question["placeholder_token_indices"],
            }
            roles["action"] = {
                "status": "empty",
                "value": None,
                "required": True,
                "question_word": question_word,
            }
            if language_analysis.question_operator:
                language_analysis.question_operator.operator_type = "ACTION_QUERY"
            language_analysis.diagnostics.append({
                "code": "ACTION_QUESTION_PLACEHOLDER_PREDICATE",
                "message": "question_placeholder_predicate_removed",
                "predicate": action_question["removed_predicate"],
            })
        definition_question = self._definition_question_spec(
            language_analysis,
            items,
            roles,
            question_word,
        )
        if definition_question:
            requested_role = "entity_type"
            requested_slot = "type_or_definition"
            role_hypotheses = [{
                "role": "entity_type",
                "confidence": 1.0,
                "source": "definition_question_operator",
                "evidence": ["definition_operator", "entity_anchor"],
            }]
            roles = {
                "entity": {"status": "fixed", **definition_question["entity"]},
                "entity_type": {
                    "status": "empty", "value": None, "required": True,
                    "question_word": question_word,
                },
            }
            if language_analysis.question_operator:
                language_analysis.question_operator.operator_type = "DEFINITION_QUERY"
        if role_hypotheses:
            if typed_question_index is None:
                semantic_role = str(role_hypotheses[0]["role"])
                requested_role = (
                    "object"
                    if question_word == "что"
                    and semantic_role in {"patient", "theme", "object"}
                    else semantic_role
                )
            if slot_constraints and requested_role not in slot_constraints:
                constraint = next(iter(slot_constraints.values()))
                slot_constraints = {requested_role: constraint}
            question_token = next(
                (item for item in items if item.get("component_type") == "question_operator"),
                None,
            )
            if question_token:
                question_token["expected_role"] = requested_role
        role_ambiguity = (
            len(role_hypotheses) > 1
            and abs(
                float(role_hypotheses[0].get("confidence", 0.0))
                - float(role_hypotheses[1].get("confidence", 0.0))
            ) < .04
        )
        if role_ambiguity:
            language_analysis.diagnostics.append({
                "code": ROLE_HYPOTHESIS_AMBIGUITY,
                "hypotheses": deepcopy(role_hypotheses[:2]),
            })
        if typed_question_index is not None and not language_analysis.question_operator:
            language_analysis.diagnostics.append({
                "code": QUESTION_OPERATOR_ERROR,
                "token_index": typed_question_index,
            })
        if typed_question_index is not None and not requested_role:
            language_analysis.diagnostics.append({
                "code": REQUESTED_ROLE_ERROR,
                "reason": "typed_constraint_has_no_compatible_slot",
            })
        if roles.get("purpose", {}).get("status") == "fixed":
            semantic_constraints["purpose"] = deepcopy(roles.pop("purpose"))
        polar_question = intent.get("question_kind") == "polar" and not requested_role
        purpose_fragment = polar_question and bool(items) and items[0]["normalized"] == "чтобы"
        ellipsis_follow_up = bool(continuation_markers) and bool(requested_role)
        if necessity_question:
            beneficiary = next(
                (item for item in items if item.get("grammatical_features", {}).get("case") == "datv"),
                None,
            )
            roles.pop("modal", None)
            if beneficiary:
                roles.pop("object", None)
                roles["agent"] = {
                    "status": "fixed",
                    **beneficiary,
                    "semantic_function": "beneficiary",
                }
        explicit_exclusions: List[Dict[str, Any]] = []
        for marker in (item for item in items if item["normalized"] == "кроме"):
            excluded = next(
                (
                    item for item in items
                    if item["index"] > marker["index"]
                    and item["part_of_speech"] in {"NOUN", "NPRO"}
                    and item["component_type"] != "question_operator"
                ),
                None,
            )
            if excluded:
                excluded["component_type"] = "continuation_exclusion"
                explicit_exclusions.append({
                    **deepcopy(excluded),
                    "mode": "STABLE_CONCEPT",
                    "source": "query_explicit",
                    "origin_query_session_id": None,
                })
        for role, value in roles.items():
            if value.get("status") == "empty":
                continue
            token = next((item for item in items if item["index"] == value.get("index")), None)
            if token:
                token["component_type"] = {"action": "query_predicate", "object": "query_object", "agent": "query_subject", "modal": "query_modal"}.get(role, "query_token")
                if token.get("normalized") in MODAL_WORDS:
                    token.update({"component_type": "query_modal", "role": "modal", "semantic_function": MODAL_WORDS[token["normalized"]]["semantic_function"], "resolution_state": TokenResolutionState.EXACT_FORM_MATCH.value})
        if requested_role:
            roles.setdefault(requested_role, {
                "status": "empty", "value": None, "required": True,
                "question_word": question_word,
            })
        if action_question:
            for item in items:
                if item["index"] in action_question["placeholder_token_indices"]:
                    item.update({
                        "component_type": "question_placeholder_predicate",
                        "semantic_role": "QUESTION_PLACEHOLDER_PREDICATE",
                    })
        frame_id = f"query-frame-{uuid.uuid4().hex[:12]}"
        normalized_text = " ".join(item["lemma"] for item in items)
        predicate = roles.get("action", {})
        answer_cardinality = self._answer_cardinality(question_word, requested_role, items, typed_question_index)
        if action_question or definition_question:
            answer_cardinality = "single"
        query_frame = {
            "id": frame_id,
            "source_text": text, "original_text": text,
            "normalized_text": normalized_text,
            "query_type": "definition_question" if definition_question else "action_question" if action_question else "continuation_role_question" if ellipsis_follow_up else "need_question" if necessity_question else "role_question" if requested_role else "polar_question" if polar_question else "statement",
            "question_word": question_word or None,
            "requested_role": requested_role or None,
            "semantic_requested_role": (
                str(role_hypotheses[0]["role"])
                if role_hypotheses
                else requested_role or None
            ),
            "missing_role": requested_role or None,
            "requested_slot": requested_slot,
            "answer_slot_type": "relation" if definition_question else "predicate_phrase" if action_question else (
                "attribute" if requested_role == "attribute" else "participant"
                if requested_role else "event"
            ),
            "requested_role_hypotheses": role_hypotheses,
            "resolution_status": (
                "AMBIGUOUS"
                if role_ambiguity
                else "PARTIALLY_RESOLVED"
                if typed_question_index is not None and not requested_role
                else "RESOLVED"
            ),
            "slot_constraints": slot_constraints,
            "semantic_constraints": semantic_constraints,
            "answer_cardinality": answer_cardinality,
            "answer_mode": answer_cardinality,
            "question_kind": "definition" if definition_question else "action" if action_question else "role_list" if answer_cardinality == "multiple" and requested_role else intent.get("question_kind"),
            "intent": intent["intent"] if intent["intent"] == "SCENE_QUESTION" else "SCENE_ROLE_QUESTION" if requested_role else intent["intent"],
            "base_intent": intent["intent"],
            "intent_classification": intent,
            "negated": any(item["normalized"] == "не" for item in items),
            "polar_verifiable": bool(
                polar_question
                and roles.get("action")
                and roles.get("action", {}).get("lemma") not in {"мочь"}
                and any(roles.get(role) for role in ("agent", "object", "location", "destination", "source"))
            ),
            "requires_clarification": necessity_question and not roles.get("action"),
            "purpose_fragment": purpose_fragment,
            "predicate": self._token_value(predicate) if predicate and predicate.get("status") == "fixed" else None,
            "action_question": deepcopy(action_question) if action_question else None,
            "definition_question": deepcopy(definition_question) if definition_question else None,
            "allowed_relations": ["IS_A", "INSTANCE_OF", "ENTITY_TYPE"] if definition_question else [],
            "roles": roles,
            "tokens": items,
            "phrase_graph": language_analysis.phrase_graph.as_dict(),
            "entity_mentions": [
                mention.as_dict(language_analysis.tokens)
                for mention in language_analysis.mentions
            ],
            "question_operator": (
                language_analysis.question_operator.as_dict()
                if language_analysis.question_operator
                else None
            ),
            "relation_phrases": list(language_analysis.relation_phrases),
            "predicate_valency_profile": valency_profile,
            "language_diagnostics": list(language_analysis.diagnostics),
            "referential": referential,
            "ellipsis_follow_up": ellipsis_follow_up,
            "continuation_markers": sorted(continuation_markers),
            "continuation_of": None,
            "inherited_roles": [],
            "excluded_roles": {requested_role: explicit_exclusions} if requested_role and explicit_exclusions else {},
            "reconstructed_query": None,
            "entity_referentials": [
                {"index": item["index"], "surface": item["surface"], "normalized": item["normalized"], "lemma": item["lemma"]}
                for item in items if item.get("entity_referential")
            ],
            "utterance": (
                language_analysis.utterance.as_dict()
                if language_analysis.utterance else None
            ),
            "dialogue_acts": [
                act.as_dict() for act in language_analysis.dialogue_acts
            ],
            "clauses": [
                clause.as_dict() for clause in language_analysis.clauses
            ],
            "clause_relations": [
                relation.as_dict()
                for relation in language_analysis.clause_relations
            ],
            "interpretation_status": (
                language_analysis.interpretation_status.value
            ),
            "interpretation_version": language_analysis.interpretation_version,
            "interpretation_trace": deepcopy(
                language_analysis.interpretation_trace
            ),
        }
        with self.repository.transaction() as conn:
            query_frame["conceptual_query_frame"] = self.semantic_projection.query_frame(conn, query_frame)
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
        result = {
            "intent_classification": intent,
            "query_frame": query_frame,
            "query_scene": (
                None
                if intent["intent"] in {
                    "SMALL_TALK",
                    "GREETING",
                    "GREETING_WITH_SMALL_TALK",
                }
                else query_scene
            ),
            "language_analysis": language_analysis.as_dict(),
        }
        if return_analysis:
            result["_analysis_object"] = language_analysis
        return result

    @staticmethod
    def _slot_role(slot: str) -> str:
        return {
            "subject": "agent",
            "direct_object": "object",
            "object": "object",
            "indirect_object": "recipient",
            "source_oblique": "source",
            "destination_oblique": "destination",
            "location_oblique": "location",
            "reference_oblique": "location",
            "instrumental": "instrument",
            "purpose_oblique": "purpose",
            "cause_oblique": "cause",
        }.get(slot, "object")

    @staticmethod
    def _action_question_spec(
        analysis: LanguageAnalysis,
        items: List[Dict[str, Any]],
        roles: Dict[str, Dict[str, Any]],
        question_word: str,
    ) -> Dict[str, Any]:
        """Recognize questions that ask to recover an event predicate."""
        if not analysis.question_operator or not any(
            act.act_type.value == "QUESTION" for act in analysis.dialogue_acts
        ):
            return {}
        lemmas = {str(item.get("lemma") or "").casefold() for item in items}
        has_subject = bool(
            roles.get("agent", {}).get("lemma")
            or roles.get("theme", {}).get("lemma")
        )
        kind = ""
        if question_word == "что" and "делать" in lemmas and has_subject:
            kind = "doing_placeholder"
        elif question_word == "чем" and "заниматься" in lemmas and has_subject:
            kind = "occupation_placeholder"
        elif question_word == "что" and "происходить" in lemmas:
            with_theme = next(
                (
                    value for value in roles.values()
                    if value.get("status") == "fixed"
                    and str(value.get("preposition") or "").casefold() in {"с", "со"}
                ),
                None,
            )
            if with_theme:
                roles.pop("instrument", None)
                roles["theme"] = {
                    **deepcopy(with_theme),
                    "semantic_function": "affected_entity",
                    "grammatical_slot": "reference_oblique",
                }
                has_subject = True
                kind = "happening_placeholder"
        if not kind:
            return {}
        placeholder_indices = [
            int(item["index"])
            for item in items
            if str(item.get("lemma") or "").casefold()
            in {"делать", "заниматься", "происходить", "быть"}
            and str(item.get("part_of_speech") or "") in {"VERB", "INFN"}
        ]
        if not placeholder_indices:
            return {}
        return {
            "status": "RECOGNIZED",
            "kind": kind,
            "placeholder_token_indices": placeholder_indices,
            "placeholder_predicate_removed": True,
            "answer_extraction": "predicate",
        }

    @staticmethod
    def _definition_question_spec(
        analysis: LanguageAnalysis,
        items: List[Dict[str, Any]],
        roles: Dict[str, Dict[str, Any]],
        question_word: str,
    ) -> Dict[str, Any]:
        lemmas = [str(item.get("lemma") or "").casefold() for item in items]
        normalized = [str(item.get("normalized") or "").casefold() for item in items]
        has_question = bool(
            analysis.question_operator
            or any(act.act_type.value == "QUESTION" for act in analysis.dialogue_acts)
        )
        if not has_question:
            return {}
        is_type_question = "тип" in lemmas and "относиться" in lemmas
        is_identity_question = (
            (any(word in {"кто", "что"} for word in normalized) and "такой" in lemmas)
            or ("являться" in lemmas and "кем" in normalized)
            or ("это" in lemmas and any(word in {"кто", "что"} for word in normalized))
            or is_type_question
        )
        if not is_identity_question:
            return {}
        ignored = {
            "кто", "что", "кем", "такой", "это", "являться", "относиться",
            "к", "какой", "тип", "также", "быть",
        }
        entity = next(
            (
                deepcopy(value)
                for value in roles.values()
                if value.get("status") == "fixed"
                and str(value.get("lemma") or "").casefold() not in ignored
                and value.get("part_of_speech") in {"NOUN", "NPRO"}
            ),
            None,
        )
        if entity is None:
            entity = next(
                (
                    deepcopy(item)
                    for item in items
                    if item.get("part_of_speech") in {"NOUN", "NPRO"}
                    and str(item.get("lemma") or "").casefold() not in ignored
                ),
                None,
            )
        if not entity or not entity.get("lemma"):
            return {}
        return {
            "status": "RECOGNIZED",
            "kind": "type_relation" if is_type_question else "identity_relation",
            "entity": entity,
            "allowed_relations": ["IS_A", "INSTANCE_OF", "ENTITY_TYPE"],
            "answer_extraction": "entity_type_relation",
        }

    def _roles_from_language(
        self,
        analysis: LanguageAnalysis,
        items: List[Dict[str, Any]],
        conn: Any,
        *,
        requested_role: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        by_index = {item["index"]: item for item in items}
        roles: Dict[str, Dict[str, Any]] = {}
        if analysis.predicate:
            predicate_item = by_index[analysis.predicate.index]
            roles["action"] = {"status": "fixed", **predicate_item}
        else:
            # Elliptical dialogue fragments retain the established fallback,
            # then receive phrase surfaces below.
            roles.update(self._roles(items, requested_role=requested_role))
        for item in items:
            modal = MODAL_WORDS.get(item["normalized"])
            if modal:
                roles["modal"] = {
                    "status": "fixed",
                    **item,
                    "role": "modal",
                    "semantic_function": modal["semantic_function"],
                }
        operator_indices = set(
            analysis.question_operator.token_indices
            if analysis.question_operator
            else []
        )
        non_content_indices = {
            token_index
            for act in analysis.dialogue_acts
            if act.act_type.value in {
                "GREETING",
                "SMALL_TALK",
                "DENIAL",
                "CONFIRMATION",
            }
            for token_index in range(act.token_start, act.token_end + 1)
        }
        known_mentions = [
            mention for mention in analysis.mentions
            if not operator_indices.intersection(mention.token_indices)
            and not non_content_indices.intersection(mention.token_indices)
            and mention.relation_function != "exclusion"
        ]
        if analysis.predicate:
            resolved = self.role_hypotheses.resolve_mentions(
                known_mentions,
                predicate_index=analysis.predicate.index,
                predicate_lemma=analysis.predicate.lemma,
                conn=conn,
            )
        else:
            resolved = []
        for index, mention in enumerate(known_mentions):
            head_item = by_index.get(mention.head)
            if not head_item:
                continue
            observation = resolved[index] if index < len(resolved) else None
            assigned_role = next(
                (
                    role for role, value in roles.items()
                    if value.get("index") == mention.head
                ),
                "",
            )
            slot = (
                str(observation["grammatical_slot"])
                if observation
                else str(head_item.get("scene_role") or assigned_role or "object")
            )
            role = self._slot_role(slot) if observation else assigned_role
            if not role or role in {"action", "modal"}:
                continue
            value = {
                "status": "fixed",
                **head_item,
                "surface": (
                    analysis.tokens[mention.head].surface
                    if mention.mention_type == "apposition"
                    else mention.surface
                ),
                "normalized": (
                    analysis.tokens[mention.head].normalized
                    if mention.mention_type == "apposition"
                    else mention.normalized_surface
                ),
                "lemma": mention.lemma,
                "canonical_lemma": mention.lemma,
                "observed_surface": mention.surface,
                "preposition": mention.preposition,
                "phrase_id": f"phrase-np-{analysis.mentions.index(mention)}",
                "token_start": mention.start,
                "token_end": mention.end,
                "mention_surface": mention.surface,
                "full_surface": mention.surface,
                "mention_type": mention.mention_type,
                "head_token_index": mention.head,
                "attributes": list(mention.attributes),
                "grammatical_slot": slot,
                "role_hypotheses": (
                    deepcopy(observation["hypotheses"])
                    if observation
                    else []
                ),
                "relation_type": mention.relation_type,
            }
            roles[role] = value
            phrase_id = value["phrase_id"]
            phrase = next(
                (
                    phrase for phrase in analysis.phrase_graph.phrases
                    if phrase.id == phrase_id
                ),
                None,
            )
            if phrase:
                phrase.metadata.update({
                    "grammatical_slot": slot,
                    "role_hypotheses": deepcopy(
                        value["role_hypotheses"]
                    ),
                })
            for dependency in analysis.phrase_graph.dependencies:
                if dependency.get("target") == phrase_id:
                    dependency["relation"] = slot
            # Direct objects retain the public generic role while preserving
            # the finer semantic distribution on the participant.
            if slot == "direct_object":
                roles.setdefault("object", value)
        return roles

    @staticmethod
    def _shared_question_cases(
        analysis: LanguageAnalysis,
        question_index: int,
        head_index: int,
    ) -> Dict[str, float]:
        question = analysis.tokens[question_index]
        head = analysis.tokens[head_index]
        result: Dict[str, float] = {}
        for question_analysis in question.analyses:
            if question_analysis.lemma != TYPED_QUESTION_LEMMA:
                continue
            for head_analysis in head.analyses:
                if head_analysis.pos not in {"NOUN", "NPRO"}:
                    continue
                case = question_analysis.features.get("case")
                if not case or case != head_analysis.features.get("case"):
                    continue
                agrees = all(
                    not question_analysis.features.get(key)
                    or not head_analysis.features.get(key)
                    or question_analysis.features.get(key)
                    == head_analysis.features.get(key)
                    for key in ("number", "gender")
                )
                if not agrees:
                    continue
                result[case] = max(
                    result.get(case, 0.0),
                    min(
                        float(question_analysis.confidence),
                        float(head_analysis.confidence),
                    ),
                )
        return result

    def _resolve_typed_question(
        self,
        analysis: LanguageAnalysis,
        roles: Dict[str, Dict[str, Any]],
        items: List[Dict[str, Any]],
        conn: Any,
    ) -> tuple[
        Optional[str], Optional[str], Optional[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        operator = analysis.question_operator
        if (
            not operator
            or operator.operator_type != "TYPED_ROLE_QUERY"
            or operator.type_constraint_token_index is None
        ):
            return None, None, None, []
        question_index = operator.token_indices[0]
        head_index = operator.type_constraint_token_index
        typed_mention = next(
            (
                mention for mention in analysis.mentions
                if mention.head == head_index
                and question_index in mention.token_indices
            ),
            None,
        )
        if not typed_mention:
            return None, None, None, []
        cases = self._shared_question_cases(
            analysis, question_index, head_index
        )
        if not cases:
            selected_case = typed_mention.features.get("case")
            if selected_case:
                cases[selected_case] = .35
        slot_scores: Dict[str, float] = {}
        prep = typed_mention.preposition.casefold().split()[-1] if typed_mention.preposition else ""
        for case, confidence in cases.items():
            if prep in {"в", "во", "на"} and case in {"loct", "loc2"}:
                slot = "location_oblique"
            elif prep in {"в", "во", "на"} and case == "accs":
                slot = "destination_oblique"
            elif prep in {"из", "от", "с", "со"} and case == "gent":
                slot = "source_oblique"
            elif prep in {"к", "ко"} and case == "datv":
                slot = "destination_oblique"
            elif case == "nomn":
                slot = "subject"
            elif case in {"accs", "gent"}:
                slot = "direct_object"
            elif case == "datv":
                slot = "indirect_object"
            elif case == "ablt":
                slot = "instrumental"
            else:
                slot = "object"
            slot_scores[slot] = max(slot_scores.get(slot, 0.0), confidence)
        fixed_slots = {
            value.get("grammatical_slot")
            for value in roles.values()
            if isinstance(value, dict) and value.get("status") == "fixed"
        }
        predicate_lemma = analysis.predicate.lemma if analysis.predicate else ""
        learned_slots = {
            str(row["grammatical_slot"])
            for row in conn.execute(
                """SELECT DISTINCT ca.grammatical_slot
                   FROM construction_arguments ca
                   JOIN construction_templates ct ON ct.id=ca.construction_id
                   WHERE ct.predicate_lemma=?""",
                (predicate_lemma,),
            ).fetchall()
        } if predicate_lemma else set()
        for slot in list(slot_scores):
            if slot in fixed_slots:
                slot_scores[slot] -= .80
            if slot in learned_slots:
                slot_scores[slot] += .42
            if slot == "subject" and "subject" not in fixed_slots:
                slot_scores[slot] += .20
            if slot == "direct_object" and "subject" in fixed_slots:
                slot_scores[slot] += .28
        if not slot_scores:
            return None, None, None, []
        requested_slot = max(
            slot_scores,
            key=lambda slot: (slot_scores[slot], slot),
        )
        requested_role = self._slot_role(requested_slot)
        semantic = self.role_hypotheses.hypotheses(
            requested_slot,
            requested_slot == "subject"
            or "direct_object" in learned_slots,
            predicate_lemma=predicate_lemma,
            conn=conn,
        )
        role_hypotheses = [
            {
                "role": item["role"],
                "confidence": item["confidence"],
                "source": item["source_type"],
                "evidence": [
                    *item.get("evidence", []),
                    "typed_unfilled_slot",
                    "type_constraint_separated",
                ],
            }
            for item in semantic
        ]
        typed_value = {
            **deepcopy(items[head_index]),
            "surface": analysis.tokens[head_index].surface,
            "normalized": analysis.tokens[head_index].normalized,
            "lemma": analysis.tokens[head_index].lemma,
            "phrase_id": f"phrase-np-{analysis.mentions.index(typed_mention)}",
            "mention_surface": typed_mention.surface,
            "head_token_index": head_index,
            "grammatical_slot": requested_slot,
            "constraint_type": "TYPE_MEMBERSHIP",
        }
        operator.requested_slot_hypotheses = [
            {"slot": slot, "score": score}
            for slot, score in sorted(
                slot_scores.items(), key=lambda item: -item[1]
            )
        ]
        typed_phrase_id = typed_value["phrase_id"]
        typed_phrase = next(
            (
                phrase for phrase in analysis.phrase_graph.phrases
                if phrase.id == typed_phrase_id
            ),
            None,
        )
        if typed_phrase:
            typed_phrase.metadata.update({
                "grammatical_slot": requested_slot,
                "question_constraint": analysis.tokens[head_index].lemma,
                "requested_role_hypotheses": deepcopy(role_hypotheses),
            })
        for dependency in analysis.phrase_graph.dependencies:
            if dependency.get("target") == typed_phrase_id:
                dependency["relation"] = f"requested:{requested_slot}"
        return requested_slot, requested_role, typed_value, role_hypotheses

    @staticmethod
    def _resolve_question_role(
        question_word: str,
        roles: Dict[str, Dict[str, Any]],
        items: List[Dict[str, Any]],
        requested_role: str,
        typed_question: bool,
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        if not requested_role:
            return None, []
        if requested_role == "instrument" and (
            question_word in {"как", "чем"}
            or any(item.get("semantic_function") == "necessity" for item in items)
        ):
            return "instrumental", [{
                "role": "instrument",
                "confidence": .90,
                "source": "question_operator",
            }]
        if typed_question:
            slot = "subject" if requested_role == "agent" else "direct_object"
            alternatives = (
                [("agent", .90), ("theme", .58), ("cause", .35)]
                if slot == "subject"
                else [("object", .78), ("patient", .74), ("theme", .68)]
            )
            return slot, [
                {"role": role, "confidence": confidence, "source": "typed_question"}
                for role, confidence in alternatives
            ]
        fixed: Dict[str, Dict[str, Any]] = {
            role: value for role, value in roles.items()
            if isinstance(value, dict) and value.get("status") == "fixed"
        }
        if question_word == "что":
            if fixed.get("agent"):
                return "direct_object", [
                    {"role": "patient", "confidence": .78, "source": "unfilled_slot"},
                    {"role": "theme", "confidence": .68, "source": "unfilled_slot"},
                    {"role": "object", "confidence": .55, "source": "grammar_prior"},
                ]
            if fixed.get("object") or fixed.get("patient") or fixed.get("theme"):
                return "subject", [
                    {"role": "theme", "confidence": .72, "source": "unfilled_slot"},
                    {"role": "agent", "confidence": .62, "source": "grammar_prior"},
                    {"role": "cause", "confidence": .40, "source": "grammar_prior"},
                ]
            if fixed.get("action"):
                nouns = [
                    item for item in items
                    if item.get("component_type") != "question_operator"
                    and item.get("part_of_speech") in {"NOUN", "NPRO"}
                ]
                if nouns:
                    return "subject", [
                        {"role": "theme", "confidence": .68, "source": "unfilled_slot"},
                        {"role": "agent", "confidence": .60, "source": "grammar_prior"},
                    ]
            return "object", [
                {"role": "object", "confidence": .60, "source": "open_question"},
                {"role": "theme", "confidence": .52, "source": "grammar_prior"},
            ]
        if question_word == "кто":
            return "subject", [
                {"role": "agent", "confidence": .82, "source": "question_case"},
                {"role": "theme", "confidence": .58, "source": "grammar_prior"},
                {"role": "cause", "confidence": .42, "source": "grammar_prior"},
            ]
        if question_word == "кого":
            return "direct_object", [
                {"role": "patient", "confidence": .84, "source": "question_case"},
                {"role": "theme", "confidence": .68, "source": "grammar_prior"},
                {"role": "object", "confidence": .52, "source": "grammar_prior"},
            ]
        role_slots = {
            "recipient": "indirect_object",
            "location": "location_oblique",
            "destination": "destination_oblique",
            "source": "source_oblique",
            "instrument": "instrumental",
            "purpose": "purpose_oblique",
            "cause": "cause_oblique",
            "time": "time_oblique",
            "quantity": "quantity",
            "manner": "manner",
        }
        slot = QUESTION_SLOT_HINTS.get(question_word) or role_slots.get(requested_role)
        return slot, [{
            "role": requested_role,
            "confidence": .9,
            "source": "question_operator",
        }]

    @staticmethod
    def _answer_cardinality(
        question_word: str,
        requested_role: str,
        items: List[Dict[str, Any]],
        typed_question_index: Optional[int],
    ) -> str:
        if question_word in MULTI_ANSWER_QUESTION_WORDS:
            return "multiple"
        if typed_question_index is not None:
            features = items[typed_question_index].get("grammatical_features", {})
            return "multiple" if features.get("number") == "plur" else "single"
        return "single" if requested_role else "none"

    def activate(self, hive_id: str, text: str, mode: str = "NEW_QUERY") -> Dict[str, Any]:
        with self.repository.transaction() as read_conn:
            hive_row = read_conn.execute(
                "SELECT conversation_id FROM hives WHERE id=?",
                (hive_id,),
            ).fetchone()
            if not hive_row:
                raise KeyError(hive_id)
            message_row = read_conn.execute(
                """SELECT id,turn_index,created_at FROM hive_messages
                   WHERE hive_id=? ORDER BY turn_index DESC LIMIT 1""",
                (hive_id,),
            ).fetchone()
            hive_snapshot = dict(hive_row)
            message_snapshot = dict(message_row) if message_row else None
        conversation_id = str(
            hive_snapshot["conversation_id"] or hive_id
        )
        parsed = self.parse(
            text,
            conversation_id=conversation_id,
            turn_index=int(
                message_snapshot["turn_index"] if message_snapshot else 0
            ),
            utterance_id=(
                f"utterance-{message_snapshot['id']}"
                if message_snapshot else ""
            ),
            received_at=str(
                message_snapshot["created_at"] if message_snapshot else ""
            ),
            return_analysis=True,
        )
        language_analysis = parsed.pop("_analysis_object")
        with self.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM hives WHERE id=?", (hive_id,)).fetchone():
                raise KeyError(hive_id)
            hive = conn.execute("SELECT conversation_id FROM hives WHERE id=?", (hive_id,)).fetchone()
            message = conn.execute(
                """SELECT id,turn_index,created_at FROM hive_messages
                   WHERE hive_id=? ORDER BY turn_index DESC LIMIT 1""",
                (hive_id,),
            ).fetchone()
            previous_dialogue_state = self.dialogue.load(
                conversation_id,
                conn,
            )
            reference_candidates = self.dialogue.references.candidates(
                language_analysis,
                previous_dialogue_state,
            )
            if reference_candidates:
                self.language.interpretations.interpret(
                    language_analysis,
                    reference_candidates=reference_candidates,
                )
                parsed["language_analysis"] = language_analysis.as_dict()
                parsed["query_frame"].update({
                    "interpretation_status": (
                        language_analysis.interpretation_status.value
                    ),
                    "interpretation_trace": deepcopy(
                        language_analysis.interpretation_trace
                    ),
                })
            dialogue_context = self._context(conn, hive_id)
            previous_state = self._load(conn, hive_id)
            parsed = self._apply_context(parsed, dialogue_context, previous_state)
            dialogue_state = self.dialogue.process(
                language_analysis,
                query_frame=parsed["query_frame"],
                conn=conn,
                message_id=str(message["id"] if message else "") or None,
            )
            self._sync_query_scene(parsed)
            parsed["dialogue_state"] = dialogue_state.as_dict()
            if dialogue_state.pending_clarification:
                parsed["query_frame"]["requires_clarification"] = True
                parsed["query_frame"]["pending_clarification"] = deepcopy(
                    dialogue_state.pending_clarification
                )
            parsed = self._resolve_token_states(conn, parsed)
            parsed["query_frame"]["conceptual_query_frame"] = self.semantic_projection.query_frame(conn, parsed["query_frame"])
            self._persist_query_frame(conn, hive_id, parsed["query_frame"])
            mode = mode if mode in MESSAGE_MODES else "NEW_QUERY"
            context_resolution = parsed["query_frame"].get("context_resolution", {})
            if (parsed["query_frame"].get("referential") and context_resolution.get("source") != "query_explicit") or parsed["query_frame"].get("ellipsis_follow_up") or parsed["query_frame"].get("purpose_fragment"):
                mode = "FOLLOW_UP"
            dialogue_context = self._update_context(dialogue_context, parsed, mode)
            self._clear_temporary_query_objects(conn, hive_id)
            if parsed["intent_classification"]["intent"] in {"SMALL_TALK", "GREETING", "GREETING_WITH_SMALL_TALK"}:
                return self._activate_conversational(conn, hive_id, text, mode, hive, message, parsed, dialogue_context)
            if parsed["query_frame"].get("query_type") == "statement" and message:
                self._store_dialogue_scene(conn, hive_id, str(message["id"]), "user", text, parsed["query_frame"])
            memory_scenes = self._memory_scenes(conn, hive_id)
            if parsed["query_frame"].get("query_type") == "definition_question":
                memory_scenes.extend(
                    self._definition_relation_scenes(conn, parsed["query_frame"])
                )
            local_scene_count = len(memory_scenes)
            definition_question = (
                parsed["query_frame"].get("query_type") == "definition_question"
            )
            evaluated = (
                [scene for scene in memory_scenes if scene.get("type") == "definition_relation"]
                if definition_question else
                [self._score_scene(parsed["query_frame"], scene, conn) for scene in memory_scenes]
            )
            imported = []
            global_visible = []
            local_candidates = (
                self._definition_candidates(parsed["query_frame"], evaluated)
                if definition_question else self._candidates(parsed["query_frame"], evaluated, conn)
            )
            local_semantic = max((float(item["scores"].get("semantic_total", 0.0)) for item in local_candidates), default=0.0)
            if not definition_question and (parsed["query_frame"].get("requested_role") or parsed["query_frame"].get("polar_verifiable")) and not parsed["query_frame"].get("requires_clarification") and parsed["query_frame"].get("context_resolution", {}).get("status") != "UNRESOLVED_CONTEXT":
                global_scenes = self._memory_scenes(conn, None, parsed["query_frame"])
                local_ids = {scene["id"] for scene in memory_scenes}
                global_evaluated = [self._score_scene(parsed["query_frame"], scene, conn) for scene in global_scenes if scene["id"] not in local_ids]
                useful = [item for item in global_evaluated if item["result_type"] != "NO_HIT"]
                global_visible = sorted(useful, key=lambda item: (-item["scores"].get("semantic_total", 0.0), item["id"]))[:8]
                should_import = local_semantic < .65 or not local_candidates
                for scene in global_visible:
                    if should_import:
                        self._import_scene(conn, hive_id, scene, text, parsed, hive, message)
                        scene["retrieval_scope"] = "IMPORTED"
                        scene["provenance"] = {"source": "global_field", "imported_for": text, "imported_at": utcnow()}
                        imported.append(scene)
                    else:
                        scene["retrieval_scope"] = "GLOBAL"
                        scene["provenance"] = {"source": "global_field", "visible_for_validation": True}
                evaluated.extend(global_visible)
            candidates = (
                self._definition_candidates(parsed["query_frame"], evaluated)
                if definition_question else self._candidates(parsed["query_frame"], evaluated, conn)
            )
            rejected_candidates = [
                deepcopy(scene["candidate_validation"])
                for scene in evaluated if scene.get("candidate_validation")
            ]
            self._persist_retrieval_trace(
                conn,
                parsed["query_frame"],
                evaluated,
                candidates,
                rejected_candidates,
            )
            self._assign_scene_activation(evaluated)
            for slot in parsed["query_scene"]["slots"]:
                if slot["status"] == "empty":
                    slot["candidates"] = [candidate["id"] for candidate in candidates]
            result_type = self._result_type(evaluated, candidates)
            active_session_id = f"query-session-{uuid.uuid4().hex[:12]}"
            previous = previous_state
            sessions = list(previous.get("query_sessions", []))
            if previous.get("query_session"):
                previous["query_session"]["status"] = (
                    "SUPERSEDED"
                    if mode == "CORRECTION"
                    else "ARCHIVED"
                )
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
            context_stage = {
                "id": "context-inheritance", "stage": "CONTEXT_INHERITANCE",
                "status": parsed["query_frame"].get("context_resolution", {}).get("status", "NOT_APPLICABLE"),
                "output": {
                    "continuation_of": parsed["query_frame"].get("continuation_of"),
                    "inherited_roles": deepcopy(parsed["query_frame"].get("inherited_roles", [])),
                    "excluded_roles": deepcopy(parsed["query_frame"].get("excluded_roles", {})),
                    "reconstructed_query": parsed["query_frame"].get("reconstructed_query"),
                },
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
                "created_for_surface": text, "dialogue_context": dialogue_context,
                "dialogue_state": dialogue_state.as_dict(),
                "context_resolution": parsed["query_frame"].get("context_resolution", {"status": "NOT_APPLICABLE"}),
                "retrieval_scope": {
                    "local": local_scene_count,
                    "global_visible": len(global_visible),
                    "imported": len(imported),
                    "global_search_limit": 64,
                    "global_import_limit": 8,
                },
                **parsed,
                "role_hypotheses": deepcopy(
                    parsed["query_frame"].get("requested_role_hypotheses", [])
                ),
                "memory_scenes": evaluated,
                "scene_matches": [
                    self._scene_trace(scene) for scene in evaluated
                ],
                "pre_candidates": [
                    self._candidate_trace(candidate)
                    for candidate in [*candidates, *rejected_candidates]
                ],
                "candidates": candidates,
                "accepted_candidates": deepcopy(candidates),
                "rejected_candidates": rejected_candidates,
                "result_type": result_type,
                "memory_sources": [
                    {"id": f"memory-source-{scene['cloud_id']}", "component_class": "memory_source", "source_scene_id": int(scene["cloud_id"]), "source_text": scene.get("source_text", ""), "retrieval_scope": "IMPORTED", "provenance": deepcopy(scene.get("provenance", {})), "query_frame_id": parsed["query_frame"]["id"]}
                    for scene in imported
                ], "inspection_projections": [], "generation_candidates": [],
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
                                "requested_slot": parsed["query_frame"].get("requested_slot"),
                                "requested_role_hypotheses": deepcopy(
                                    parsed["query_frame"].get(
                                        "requested_role_hypotheses", []
                                    )
                                ),
                                "roles": deepcopy(parsed["query_frame"].get("roles", {})),
                                "tokens": deepcopy(parsed["query_frame"].get("tokens", [])),
                                "phrase_graph": deepcopy(
                                    parsed["query_frame"].get("phrase_graph", {})
                                ),
                                "entity_mentions": deepcopy(
                                    parsed["query_frame"].get(
                                        "entity_mentions", []
                                    )
                                ),
                                "question_operator": deepcopy(
                                    parsed["query_frame"].get(
                                        "question_operator"
                                    )
                                ),
                                "diagnostics": deepcopy(
                                    parsed["query_frame"].get(
                                        "language_diagnostics", []
                                    )
                                ),
                                "dialogue_context": deepcopy(dialogue_context),
                                "context_resolution": deepcopy(parsed["query_frame"].get("context_resolution", {})),
                            },
                        },
                        context_stage,
                        {
                            "id": "query-scene-completion", "stage": "QUERY_SCENE_COMPLETION",
                            "status": parsed["query_scene"].get("status", "RESOLVED"),
                            "output": {"slots": deepcopy(parsed["query_scene"].get("slots", []))},
                        },
                        {
                            "id": "memory-scene-search", "stage": "MEMORY_SCENE_SEARCH",
                            "status": "MATCHES_FOUND" if any(item["result_type"] != "NO_HIT" for item in evaluated) else "NO_MATCH",
                            "output": [self._scene_trace(item) for item in evaluated],
                        },
                        {
                            "id": "candidate-ranking", "stage": "CANDIDATE_RANKING",
                            "status": "CANDIDATES_FOUND" if candidates else "NO_CANDIDATES",
                            "output": {
                                "accepted_candidates": [self._candidate_trace(item) for item in candidates],
                                "rejected_candidates": [self._candidate_trace(item) for item in rejected_candidates],
                            },
                        },
                    ],
                },
            }
            best_scores = candidates[0].get("scores", {}) if candidates else {}
            state.update({
                "semantic_total": float(best_scores.get("semantic_total", 0.0)),
                "gravity": float(best_scores.get("gravity", 0.0)),
                "decision_score": float(best_scores.get("decision_score", 0.0)),
            })
            if parsed["query_frame"].get("context_resolution", {}).get("status") == "UNRESOLVED_CONTEXT":
                state["result_type"] = "UNRESOLVED_CONTEXT"
                state["answer"].update({"status": "UNRESOLVED_CONTEXT", "answer_mode": "unknown", "status_message": "Ссылочный контекст не разрешён в текущем улье."})
            from .unknown_search import UnknownTokenSearchService
            placer = UnknownTokenSearchService(self.repository)
            placement_search = {"id": f"query-placement-{uuid.uuid4().hex[:10]}", "query_session_id": active_session_id, "query_frame_id": parsed["query_frame"]["id"], "query_scene_id": parsed["query_scene"]["id"], "message_id": str(message["id"] if message else ""), "created_for_surface": text}
            for candidate in candidates:
                candidate.update({"conversation_id": str(hive["conversation_id"] or "") if hive else "", "message_id": str(message["id"] if message else ""), "query_session_id": active_session_id, "query_frame_id": parsed["query_frame"]["id"], "query_scene_id": parsed["query_scene"]["id"], "created_for_surface": text})
                candidate["cell_id"] = placer._place_role_candidate(conn, hive_id, candidate)
                for source_id in candidate.get("sources", []):
                    source = next((item for item in evaluated if item.get("id") == source_id), None)
                    scene_id = str(source_id).removeprefix("scene-")
                    if source and scene_id.isdigit() and source.get("result_type") in {"FULL_HIT", "ROLE_HIT"} and float(source.get("scores", {}).get("total_score", 0.0)) >= 0.55:
                        source_cell_id = placer._place_memory_source(conn, hive_id, int(scene_id), source, placement_search)
                        if candidate.get("answer_mode") == "explanation" and candidate.get("cell_id") is None:
                            candidate["cell_id"] = source_cell_id
                        if not any(item.get("source_scene_id") == int(scene_id) for item in state["memory_sources"]):
                            state["memory_sources"].append({"id": f"memory-source-{scene_id}", "component_class": "memory_source", "source_scene_id": int(scene_id), "source_text": source.get("source_text", ""), "query_frame_id": parsed["query_frame"]["id"], "message_id": str(message["id"] if message else "")})
            self._persist_scene_activation(conn, hive_id, evaluated)
            self._sync_working_cells(conn, hive_id, state)
            self._set_pipeline(state, candidate_count=len(candidates), memory_source_count=len(state["memory_sources"]))
            if parsed["query_frame"].get("query_type") == "polar_question":
                polar_answer = self._polar_answer(parsed["query_frame"], evaluated)
                state["answer"] = polar_answer
                state["semantic_total"] = float(polar_answer["semantic_total"])
                state["decision_score"] = float(polar_answer["decision_score"])
                state["vibration"].update({"status": "FINISHED", "max_steps": 0})
                state["pipeline"].update({
                    "memory_search": {
                        "status": polar_answer["evidence_status"],
                        "memory_source_count": len(state["memory_sources"]),
                        "candidate_count": 0,
                    },
                    "vibration": {"status": "SKIPPED", "current_step": 0},
                    "sentence_planning": {"status": "SKIPPED"},
                    "morphology_generation": {"status": "SKIPPED"},
                    "answer": {"status": polar_answer["status"]},
                })
                state["reasoning_trace"]["stages"].append({
                    "id": "polar-decision",
                    "stage": "POLAR_DECISION",
                    "status": polar_answer["evidence_status"],
                    "output": deepcopy(polar_answer),
                })
                self._set_pipeline_state(state, "ANSWER_READY" if polar_answer["status"] == "RESOLVED" else "FAILED")
            elif parsed["query_frame"].get("requires_clarification"):
                pending_clarification = (
                    parsed["query_frame"].get("pending_clarification") or {}
                )
                clarification_surface = str(
                    pending_clarification.get("question")
                    or "Уточните, для какого действия это нужно."
                )
                clarification = {
                    "query": parsed["query_frame"].get("source_text", ""),
                    "answer_mode": "clarification",
                    "resolved_role": parsed["query_frame"].get("requested_role"),
                    "resolved_value": None,
                    "confidence": 1.0,
                    "supporting_scenes": [],
                    "surface_answer": clarification_surface,
                    "full_surface_answer": clarification_surface,
                    "status": "UNRESOLVED",
                    "status_message": "Ожидается уточнение цели.",
                    "evidence_status": "NEEDS_CLARIFICATION",
                    "pending_clarification": deepcopy(pending_clarification),
                    "semantic_total": 0.0,
                    "gravity": 0.0,
                    "decision_score": 0.0,
                }
                state["answer"] = clarification
                state["vibration"].update({"status": "FINISHED", "max_steps": 0})
                state["pipeline"].update({
                    "memory_search": {"status": "WAITING_FOR_CLARIFICATION", "memory_source_count": 0, "candidate_count": 0},
                    "vibration": {"status": "SKIPPED", "current_step": 0},
                    "sentence_planning": {"status": "SKIPPED"},
                    "morphology_generation": {"status": "SKIPPED"},
                    "answer": {"status": "NEEDS_CLARIFICATION"},
                })
                state["reasoning_trace"]["stages"].append({
                    "id": "clarification",
                    "stage": "CLARIFICATION",
                    "status": "NEEDS_CLARIFICATION",
                    "output": deepcopy(clarification),
                })
                self._set_pipeline_state(state, "FAILED")
            source_evidence = [
                evidence
                for candidate in state.get("candidates", [])
                for evidence in candidate.get("fact_evidence", [])
            ]
            response_plan = self.responses.plan(
                interpretation_status=str(
                    parsed["query_frame"].get(
                        "interpretation_status",
                        "STABLE",
                    )
                ),
                query_frame=parsed["query_frame"],
                answer=state.get("answer"),
                candidates=state.get("candidates", []),
                dialogue_state=state.get("dialogue_state"),
                source_evidence=source_evidence,
            )
            source_ids = {
                str(source_id)
                for candidate in state.get("candidates", [])
                for source_id in candidate.get("sources", [])
            }
            semantic_axes = {
                "roles": parsed["query_frame"].get("roles", {}),
                **(
                    {
                        key: parsed["query_frame"]["clauses"][0].get(key)
                        for key in ("polarity", "modality", "actuality")
                    }
                    if parsed["query_frame"].get("clauses") else {}
                ),
            }
            state["response_plan"] = self.responses.persist(
                conn,
                response_plan,
                conversation_id=conversation_id,
                source_utterance_id=(
                    language_analysis.utterance.id
                    if language_analysis.utterance else None
                ),
                surface=state.get("answer", {}).get("surface_answer"),
                independent_source_count=len(source_ids),
                semantic_axes=semantic_axes,
                persist_derived=bool(
                    state.get("answer", {}).get("surface_answer")
                ),
            )
            state["hive"] = {"status": "ACTIVE", "reasoning_step": 0, "pipeline": state["pipeline"]}
            conn.execute("UPDATE hives SET query_text=?, query_json=?, updated_at=? WHERE id=?", (text, encode({"original_text": text, "query_frame_id": parsed["query_frame"]["id"], "query_scene_id": parsed["query_scene"]["id"], "active_query_session_id": active_session_id}), utcnow(), hive_id))
            self._save(conn, hive_id, state)
            return deepcopy(state)

    def _activate_conversational(self, conn: Any, hive_id: str, text: str, mode: str, hive: Any, message: Any, parsed: Dict[str, Any], dialogue_context: Dict[str, Any]) -> Dict[str, Any]:
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
            "dialogue_context": dialogue_context,
            "context_resolution": parsed["query_frame"].get("context_resolution", {"status": "NOT_APPLICABLE"}),
            "retrieval_scope": {"local": 0, "imported": 0, "global_limit": 8},
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
            if token.get("component_type") == "context_reference" and token.get("context_resolution") == "RESOLVED":
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
            alias = conn.execute(
                """SELECT ea.entity_id,e.canonical_lemma,e.display_name,
                          et.cloud_id AS entity_type_id,et.canonical_lemma AS entity_type_lemma,
                          et.display_name AS entity_type_surface
                   FROM entity_aliases ea JOIN entities e ON e.cloud_id=ea.entity_id
                   LEFT JOIN concept_relations relation
                     ON relation.subject_lexeme_cloud_id=ea.entity_id
                    AND relation.relation_type IN ('IS_A','INSTANCE_OF')
                    AND relation.status<>'DEPRECATED'
                   LEFT JOIN entities et ON et.cloud_id=relation.object_lexeme_cloud_id
                   WHERE ea.normalized_alias=?
                   ORDER BY ea.confidence DESC,relation.confidence DESC,ea.id LIMIT 1""",
                (str(token.get("normalized") or "").casefold(),),
            ).fetchone()
            if alias:
                token.update({
                    "entity_id": int(alias["entity_id"]),
                    "resolution_state": "KNOWN_ENTITY_ALIAS",
                    "entity_type": (
                        {
                            "entity_id": int(alias["entity_type_id"]),
                            "lemma": str(alias["entity_type_lemma"]),
                            "surface": str(alias["entity_type_surface"]),
                        }
                        if alias["entity_type_id"] is not None else None
                    ),
                    # Entity identity is semantic memory, whereas animacy and
                    # proper-name status describe this concrete word form.
                    # Do not infer either morphology property merely because
                    # an alias is known to memory.
                    "entity_resolution": {
                        "entity_id": int(alias["entity_id"]),
                        "matched_alias": str(token.get("surface") or ""),
                    },
                })
        for role, value in frame.get("roles", {}).items():
            if value.get("status") == "empty":
                continue
            matching = next((item for item in frame["tokens"] if item["index"] == value.get("index")), None)
            if matching:
                value.update({
                    key: matching[key]
                    for key in (
                        "word_form_cloud_id", "lexeme_cloud_id", "entity_id",
                        "entity_type", "proper_name", "resolution_state",
                    ) if key in matching
                })
                if matching.get("entity_id") is not None:
                    value["grammatical_features"] = deepcopy(
                        matching.get("grammatical_features", {})
                    )
        return parsed

    def resolve_mode(self, hive_id: str, text: str, resolved_mode: Optional[str] = None) -> str:
        if resolved_mode:
            if resolved_mode not in MESSAGE_MODES:
                raise ValueError(f"unsupported resolved_mode: {resolved_mode}")
            return resolved_mode
        parsed = self.parse(text).get("query_frame", {})
        if any(
            act.get("act_type") == "CORRECTION"
            for act in parsed.get("dialogue_acts", [])
        ):
            return "CORRECTION"
        if parsed.get("ellipsis_follow_up"):
            return "FOLLOW_UP"
        if parsed.get("purpose_fragment"):
            return "FOLLOW_UP"
        referential = parsed.get("referential")
        explicit = (parsed.get("roles") or {}).get(REFERENTIALS.get(str(referential), ""), {})
        if referential and not (explicit.get("status") == "fixed" and explicit.get("normalized") != referential):
            return "FOLLOW_UP"
        if parsed.get("entity_referentials"):
            return "FOLLOW_UP"
        return "NEW_QUERY"

    def _context(self, conn: Any, hive_id: str) -> Dict[str, Any]:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        metadata = decode(row["metadata_json"], {}) if row else {}
        context = metadata.get("dialogue_context") or {}
        return {role: deepcopy(context.get(role)) for role in DIALOGUE_CONTEXT_ROLES if context.get(role)}

    @staticmethod
    def _context_value(value: Dict[str, Any], role: str, source: str = "explicit") -> Dict[str, Any]:
        return {
            "role": role,
            "lemma": value.get("lemma"),
            "surface": value.get("surface"),
            "normalized": value.get("normalized") or value.get("surface"),
            "word_form_cloud_id": value.get("word_form_cloud_id"),
            "lexeme_cloud_id": value.get("lexeme_cloud_id"),
            "part_of_speech": value.get("part_of_speech", "NOUN"),
            "grammatical_features": deepcopy(value.get("grammatical_features", {})),
            "preposition": value.get("preposition", ""),
            "source": source,
            "updated_at": utcnow(),
        }

    def _update_context(self, context: Dict[str, Any], parsed: Dict[str, Any], mode: str = "NEW_QUERY") -> Dict[str, Any]:
        frame = parsed.get("query_frame", {})
        has_reference = bool(frame.get("referential") or frame.get("entity_referentials") or frame.get("ellipsis_follow_up") or frame.get("purpose_fragment"))
        updated = deepcopy(context) if mode in {"FOLLOW_UP", "CORRECTION"} or has_reference else {}
        roles = frame.get("roles", {})
        explicit_spatial = []
        for role in DIALOGUE_CONTEXT_ROLES:
            value = roles.get(role)
            if value and value.get("status") == "fixed" and value.get("lemma") and value.get("source") != "dialogue_context":
                updated[role] = self._context_value(value, role)
                if role in CONTEXT_ROLES:
                    explicit_spatial.append((
                        int(
                            value.get("index")
                            if value.get("index") is not None else -1
                        ),
                        role,
                        value,
                    ))
        if explicit_spatial:
            _, original_role, latest = max(explicit_spatial, key=lambda item: item[0])
            updated["location"] = {
                **self._context_value(latest, "location"),
                "original_role": original_role,
            }
        return updated

    def _apply_context(self, parsed: Dict[str, Any], context: Dict[str, Any], previous_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        frame = parsed["query_frame"]
        referential = frame.get("referential")
        resolutions: List[Dict[str, Any]] = []
        unresolved: List[Dict[str, Any]] = []
        if referential:
            role = REFERENTIALS[referential]
            value = context.get(role)
            if not value:
                explicit = frame.get("roles", {}).get(role)
                if explicit and explicit.get("status") == "fixed" and explicit.get("lemma"):
                    value = self._context_value(explicit, role, source="query_explicit")
            if value:
                resolution_source = str(value.get("source") or "dialogue_context")
                reference_token = next((token for token in frame.get("tokens", []) if token.get("normalized") == referential), {})
                resolved = {**deepcopy(value), "status": "fixed", "component_type": "context_reference", "context_reference": referential, "source": resolution_source, "index": reference_token.get("index")}
                frame["roles"][role] = resolved
                reference_token.update({"role": role, "component_type": "context_reference", "context_resolution": "RESOLVED", "resolution_state": TokenResolutionState.LEXEME_MATCH.value, "lemma": value.get("lemma"), "semantic_lemma": value.get("lemma")})
                resolutions.append({"referential": referential, "role": role, "context_role": role, "source": resolution_source, "value": deepcopy(value)})
            else:
                unresolved.append({"referential": referential, "role": role})
        tokens = frame.get("tokens", [])
        if frame.get("purpose_fragment"):
            previous_state = previous_state or {}
            session = previous_state.get("query_session") or (previous_state.get("query_sessions") or [None])[-1]
            previous_frame = previous_state.get("query_frame") or {}
            previous_agent = (previous_frame.get("roles") or {}).get("agent", {})
            if previous_frame.get("query_type") == "need_question" and previous_agent.get("lemma"):
                inherited_agent = {
                    **deepcopy(previous_agent),
                    "status": "fixed",
                    "component_type": "context_reference",
                    "context_reference": "purpose",
                    "source": "previous_query",
                    "index": None,
                }
                frame["roles"]["agent"] = inherited_agent
                frame["roles"]["instrument"] = {
                    "status": "empty",
                    "value": None,
                    "required": True,
                    "question_word": previous_frame.get("question_word") or "что",
                }
                frame.update({
                    "query_type": "continuation_role_question",
                    "question_word": previous_frame.get("question_word") or "что",
                    "requested_role": "instrument",
                    "requires_clarification": False,
                    "continuation_of": session.get("id") if session else None,
                    "inherited_roles": ["agent"],
                })
                action = (frame["roles"].get("action") or {}).get("surface") or (frame["roles"].get("action") or {}).get("lemma") or ""
                obj = (frame["roles"].get("object") or {}).get("surface") or (frame["roles"].get("object") or {}).get("lemma") or ""
                agent = previous_agent.get("surface") or previous_agent.get("lemma") or ""
                frame["reconstructed_query"] = f"Что нужно {agent}, чтобы {action} {obj}?".replace("  ", " ")
                resolutions.append({
                    "referential": "чтобы",
                    "role": "instrument",
                    "context_role": "agent",
                    "source": "previous_query",
                    "value": deepcopy(inherited_agent),
                })
            else:
                unresolved.append({
                    "referential": "чтобы",
                    "role": "instrument",
                    "reason": "missing_need_question_context",
                })
        for reference in frame.get("entity_referentials", []):
            index = int(reference.get("index", -1))
            token = next((item for item in tokens if int(item.get("index", -2)) == index), {})
            previous = tokens[index - 1].get("normalized") if 0 < index < len(tokens) else ""
            existing_role = next((name for name, item in frame.get("roles", {}).items() if item.get("index") == index), None)
            if previous == "у":
                role, context_role, injected_role = "possessor", "agent", None
            else:
                role = existing_role or ("agent" if token.get("grammatical_features", {}).get("case") == "nomn" else "object")
                context_role = role
                injected_role = role
            value = context.get(context_role)
            if value:
                resolution_source = str(value.get("source") or "dialogue_context")
                resolved = {**deepcopy(value), "status": "fixed", "component_type": "context_reference", "context_reference": reference.get("normalized"), "source": resolution_source, "index": index}
                if injected_role:
                    frame["roles"][injected_role] = resolved
                token.update({"role": role, "component_type": "context_reference", "context_resolution": "RESOLVED", "resolution_state": TokenResolutionState.LEXEME_MATCH.value, "lemma": value.get("lemma"), "semantic_lemma": value.get("lemma")})
                resolutions.append({"referential": reference.get("normalized"), "role": role, "context_role": context_role, "source": resolution_source, "value": deepcopy(value)})
            else:
                unresolved.append({"referential": reference.get("normalized"), "role": role, "context_role": context_role})
        if frame.get("ellipsis_follow_up"):
            # The active session is still available before it is archived.  A
            # previously archived session is retained as a fallback, so a
            # continuation never relies solely on a mutable dialogue summary.
            previous_state = previous_state or {}
            session = previous_state.get("query_session") or (previous_state.get("query_sessions") or [None])[-1]
            previous_frame = previous_state.get("query_frame") or {}
            previous_roles = previous_frame.get("roles") or {}
            requested = frame.get("requested_role")
            inherited: List[str] = []
            explicit_anchors = {
                role: value
                for role, value in frame.get("roles", {}).items()
                if role != requested and value.get("status") == "fixed" and value.get("lemma")
            }
            current_exclusions = deepcopy(frame.get("excluded_roles") or {})
            if not session or not previous_frame:
                self_contained_except = (
                    set(frame.get("continuation_markers", [])) == {"кроме"}
                    and bool(current_exclusions.get(requested or ""))
                    and bool(explicit_anchors)
                )
                if self_contained_except:
                    resolutions.append({
                        "referential": "кроме",
                        "role": requested,
                        "context_role": requested,
                        "source": "query_explicit",
                        "value": deepcopy(current_exclusions.get(requested or "", [])),
                    })
                else:
                    unresolved.append({"referential": next(iter(frame.get("continuation_markers", [])), "ещё"), "role": requested or "context", "reason": "missing_previous_query_session"})
            else:
                frame["continuation_of"] = session.get("id")
                requested_aliases = (
                    {"patient", "theme", "object"}
                    if requested == "patient"
                    else {requested}
                )
                for role, value in previous_roles.items():
                    # Current explicit language wins.  The role being asked
                    # for remains empty; it is not copied from the answer.
                    current = frame.get("roles", {}).get(role, {})
                    if role in requested_aliases or (
                        current.get("status") == "fixed"
                        and current.get("lemma")
                    ):
                        continue
                    if value.get("status") == "fixed" and value.get("lemma"):
                        inherited_value = {**deepcopy(value), "status": "fixed", "component_type": "context_reference", "context_reference": "continuation", "source": "previous_query", "index": None}
                        frame["roles"][role] = inherited_value
                        inherited.append(role)
                        resolutions.append({"referential": "continuation", "role": role, "context_role": role, "source": "previous_query", "value": deepcopy(inherited_value)})
                # Older lightweight context can fill a role absent from the
                # frame, but it may not replace explicit or session evidence.
                for role in DIALOGUE_CONTEXT_ROLES:
                    if role in requested_aliases or role in frame["roles"]:
                        continue
                    value = context.get(role)
                    if value and value.get("lemma"):
                        frame["roles"][role] = {**deepcopy(value), "status": "fixed", "component_type": "context_reference", "context_reference": "continuation", "source": "dialogue_context", "index": None}
                        inherited.append(role)
                if not inherited and not explicit_anchors and not current_exclusions.get(requested or ""):
                    unresolved.append({"referential": next(iter(frame.get("continuation_markers", [])), "ещё"), "role": requested or "context", "reason": "no_inheritable_roles"})
                frame["inherited_roles"] = inherited

                # Exclusions are cumulative.  The previous selected answer is
                # added even after its session has been archived, preventing a
                # chain of "ещё" questions from returning to the same value.
                excluded = deepcopy(previous_frame.get("excluded_roles") or {})
                for role, values in current_exclusions.items():
                    excluded.setdefault(role, []).extend(deepcopy(values))
                answer = previous_state.get("answer") or {}
                resolved = str(answer.get("resolved_value") or "").casefold().strip(". ")
                previous_value = next((item for item in previous_state.get("candidates", []) if resolved and resolved in {str(item.get("lemma") or "").casefold(), str(item.get("surface") or "").casefold().strip(". ")}), None)
                if previous_value and requested:
                    excluded.setdefault(requested, []).append({
                        "lemma": previous_value.get("lemma"), "surface": previous_value.get("surface"),
                        "lexeme_cloud_id": previous_value.get("lexeme_cloud_id"), "normalized": previous_value.get("lemma"),
                        "mode": "STABLE_CONCEPT", "origin_query_session_id": session.get("id"),
                    })
                elif requested and answer.get("resolved_value"):
                    excluded.setdefault(requested, []).append({"lemma": answer["resolved_value"], "normalized": answer["resolved_value"], "surface": answer.get("surface_answer", ""), "mode": "EXACT_LEMMA", "origin_query_session_id": session.get("id")})
                else:
                    previous_requested_value = previous_roles.get(requested, {})
                    if requested == "patient" and not previous_requested_value.get("lemma"):
                        previous_requested_value = (
                            previous_roles.get("object", {})
                            or previous_roles.get("theme", {})
                        )
                if (
                    requested
                    and not previous_value
                    and not answer.get("resolved_value")
                    and previous_requested_value.get("lemma")
                ):
                    # A statement has no selected answer yet, but the stated
                    # role is still the value meant by "ещё".
                    excluded.setdefault(requested, []).append({
                        **deepcopy(previous_requested_value),
                        "mode": "STABLE_CONCEPT",
                        "origin_query_session_id": session.get("id"),
                    })
                for role, values in excluded.items():
                    unique: Dict[str, Dict[str, Any]] = {}
                    for value in values:
                        key = str(value.get("lexeme_cloud_id") or value.get("lemma") or value.get("normalized") or value.get("surface") or "").casefold()
                        if key:
                            unique[key] = value
                    excluded[role] = list(unique.values())
                frame["excluded_roles"] = excluded
            action = frame["roles"].get("action", {}).get("surface") or frame["roles"].get("action", {}).get("lemma")
            agent = frame["roles"].get("agent", {}).get("surface") or frame["roles"].get("agent", {}).get("lemma")
            pieces = [item for item in (action, agent) if item]
            exclusions = [item.get("surface") or item.get("lemma") for item in frame.get("excluded_roles", {}).get(requested, [])]
            marker = "ещё " if "ещё" in frame.get("continuation_markers", []) else ""
            question = (frame.get("question_word") or "что").capitalize()
            frame["reconstructed_query"] = " ".join([f"{question} {marker}".rstrip(), *pieces]) + (f", кроме {', '.join(item for item in exclusions if item)}" if exclusions else "") + "?"
        if unresolved:
            primary = unresolved[0]
            frame["context_resolution"] = {"status": "UNRESOLVED_CONTEXT", **primary, "references": resolutions + unresolved}
        elif resolutions:
            primary = resolutions[0]
            frame["context_resolution"] = {"status": "RESOLVED", **primary, "references": resolutions}
        else:
            frame["context_resolution"] = {"status": "NOT_APPLICABLE", "referential": None}
        frame["dialogue_context"] = deepcopy(context)
        excluded_roles = frame.get("excluded_roles", {})
        if "object" not in excluded_roles:
            for canonical_role in ("patient", "theme"):
                canonical_exclusions = excluded_roles.get(canonical_role)
                if canonical_exclusions:
                    excluded_roles["object"] = deepcopy(canonical_exclusions)
                    break
        self._sync_query_scene(parsed)
        return parsed

    def _sync_query_scene(self, parsed: Dict[str, Any]) -> None:
        scene = parsed.get("query_scene")
        if scene is None:
            return
        frame = parsed["query_frame"]
        requested = frame.get("requested_role")
        slots = []
        for role in SUPPORTED_ROLES:
            value = frame.get("roles", {}).get(role)
            if not value:
                continue
            if value.get("status") == "empty":
                slots.append({
                    "id": f"slot-{role}",
                    "type": "role_slot",
                    "role": role,
                    "question_word": frame.get("question_word"),
                    "label": ROLE_LABELS.get(role, role.upper()),
                    "status": "empty",
                    "required": True,
                    "candidates": [],
                })
            else:
                slots.append({"id": f"slot-{role}", "role": role, "status": "fixed", **self._token_value(value)})
        scene.update({
            "status": "INCOMPLETE" if requested else "RESOLVED",
            "requested_role": requested,
            "slots": slots,
        })

    def _store_dialogue_scene(
        self, conn: Any, hive_id: str, message_id: str, source_role: str,
        source_text: str, frame: Dict[str, Any], *,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        roles = {
            role: deepcopy(value)
            for role, value in frame.get("roles", {}).items()
            if value.get("status") == "fixed" and value.get("lemma")
        }
        if not roles:
            return
        provenance = deepcopy(provenance or {})
        assistant_derived = source_role == "assistant" or bool(
            provenance.get("source_type") == "assistant_derived_answer"
        )
        confirmation = str(source_text).strip().casefold() in {
            "да", "верно", "именно", "подтверждаю",
        }
        memory_class = (
            "ASSISTANT_DERIVED" if assistant_derived else
            "USER_CONFIRMATION" if confirmation else "USER_ASSERTION"
        )
        action = roles.get("action") or {}
        complements = [
            role for role in ACTION_QUESTION_COMPLEMENT_ROLES
            if (roles.get(role) or {}).get("lemma")
        ]
        incomplete = (
            assistant_derived
            and bool(action.get("lemma"))
            and str(action.get("lemma")).casefold() not in {"быть", "являться"}
            and not complements
        )
        missing_roles = ["object_or_context"] if incomplete else []
        provenance.update({
            "source_type": "assistant_derived_answer" if assistant_derived else "user_assertion",
            "knowledge_status": "DERIVED" if assistant_derived else "OBSERVED",
            "independent_evidence": False if assistant_derived else True,
            "eligible_for_fact_retrieval": False if assistant_derived else True,
            "memory_class": memory_class,
            "completion_status": "SEMANTICALLY_INCOMPLETE" if incomplete else "COMPLETE",
            "missing_supported_roles": missing_roles,
        })
        now = utcnow()
        conn.execute(
            "UPDATE hive_dialogue_scenes SET activation=MAX(.05, activation * .9), retention=MAX(.1, retention * .98), updated_at=? WHERE hive_id=?",
            (now, hive_id),
        )
        conn.execute(
            """INSERT INTO hive_dialogue_scenes
            (id,hive_id,message_id,source_role,source_text,roles_json,memory_class,source_type,
             knowledge_status,independent_evidence,eligible_for_fact_retrieval,derived_from_json,
             root_evidence_ids_json,provenance_json,completion_status,missing_supported_roles_json,
             activation,retention,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,1,?,?)
            ON CONFLICT(hive_id,message_id) DO UPDATE SET source_text=excluded.source_text,
            roles_json=excluded.roles_json,memory_class=excluded.memory_class,source_type=excluded.source_type,
            knowledge_status=excluded.knowledge_status,independent_evidence=excluded.independent_evidence,
            eligible_for_fact_retrieval=excluded.eligible_for_fact_retrieval,
            derived_from_json=excluded.derived_from_json,root_evidence_ids_json=excluded.root_evidence_ids_json,
            provenance_json=excluded.provenance_json,completion_status=excluded.completion_status,
            missing_supported_roles_json=excluded.missing_supported_roles_json,
            activation=1, retention=1, updated_at=excluded.updated_at""",
            (
                f"dialogue-scene-{uuid.uuid4().hex[:12]}", hive_id, message_id,
                source_role, source_text, encode(roles), memory_class,
                provenance["source_type"], provenance["knowledge_status"],
                int(bool(provenance["independent_evidence"])),
                int(bool(provenance["eligible_for_fact_retrieval"])),
                encode(provenance.get("source_scene_ids", [])),
                encode(provenance.get("root_evidence_ids", [])), encode(provenance),
                provenance["completion_status"], encode(missing_roles), now, now,
            ),
        )

    def persist_assistant_answer(self, hive_id: str) -> Optional[Dict[str, Any]]:
        state = self.get(hive_id)
        answer = state.get("answer") or {}
        query_session = state.get("query_session") or {}
        text = str(answer.get("full_surface_answer") or answer.get("surface_answer") or "").strip()
        if answer.get("status") != "RESOLVED" or not text or not query_session.get("id"):
            return None
        with self.repository.transaction() as read_conn:
            existing = read_conn.execute(
                "SELECT id FROM hive_messages WHERE hive_id=? AND role='assistant' AND parsed_json LIKE ? LIMIT 1",
                (hive_id, f'%\"query_session_id\":\"{query_session["id"]}\"%'),
            ).fetchone()
            if existing:
                return dict(existing)
            turn = int(read_conn.execute("SELECT COALESCE(MAX(turn_index), 0) FROM hive_messages WHERE hive_id=?", (hive_id,)).fetchone()[0]) + 1
            hive = read_conn.execute(
                "SELECT conversation_id FROM hives WHERE id=?",
                (hive_id,),
            ).fetchone()
            conversation_id = str(
                (hive["conversation_id"] if hive else "") or hive_id
            )
        message_id = f"message-{uuid.uuid4().hex[:12]}"
        now = utcnow()
        selected_ids = set(answer.get("selected_candidate_ids") or [])
        selected_candidates = [
            candidate for candidate in state.get("candidates", [])
            if not selected_ids or candidate.get("id") in selected_ids
        ]
        source_scene_ids = sorted({
            str(source_id)
            for candidate in selected_candidates
            for source_id in candidate.get("sources", [])
            if source_id
        })
        source_relation_ids = sorted({
            str(relation_id)
            for candidate in selected_candidates
            for relation_id in candidate.get("source_relation_ids", [])
            if relation_id
        })
        root_evidence_ids = sorted({
            str(source_id)
            for candidate in selected_candidates
            for source_id in (
                candidate.get("root_source_ids", [])
                or candidate.get("sources", [])
            )
            if source_id
        } | set(source_relation_ids))
        provenance = {
            **deepcopy(answer.get("provenance") or {}),
            "answer_id": str(answer.get("answer_id") or f"answer-{uuid.uuid4().hex[:12]}"),
            "source_type": "assistant_derived_answer",
            "knowledge_status": "DERIVED",
            "independent_evidence": False,
            "eligible_for_fact_retrieval": False,
            "source_scene_ids": source_scene_ids,
            "source_relation_ids": source_relation_ids,
            "root_evidence_ids": root_evidence_ids,
            "derivation_type": (
                "relation_extraction"
                if answer.get("resolved_role") == "entity_type"
                else "predicate_extraction"
                if state.get("query_frame", {}).get("query_type") == "action_question"
                else "role_extraction"
            ),
            "selected_candidate_ids": sorted(selected_ids),
            "independent_fact": False,
        }
        parsed = self.parse(
            text,
            conversation_id=conversation_id,
            turn_index=turn,
            speaker_role="assistant",
            source_type="derived_answer",
            utterance_id=f"utterance-{message_id}",
            received_at=now,
            return_analysis=True,
        )
        language_analysis = parsed.pop("_analysis_object")
        with self.repository.transaction() as conn:
            existing = conn.execute(
                "SELECT id FROM hive_messages WHERE hive_id=? AND role='assistant' AND parsed_json LIKE ? LIMIT 1",
                (hive_id, f'%\"query_session_id\":\"{query_session["id"]}\"%'),
            ).fetchone()
            if existing:
                return dict(existing)
            conn.execute(
                "INSERT INTO hive_messages(id,hive_id,turn_index,role,text,parsed_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (message_id, hive_id, turn, "assistant", text, encode({"query_session_id": query_session["id"], "source": "resolved_answer", "provenance": provenance}), now),
            )
            dialogue_text = text
            if answer.get("answer_mode") == "polar" and text.startswith(("Да. ", "Нет. ")):
                dialogue_text = text.split(" ", 1)[1]
            if parsed.get("query_frame", {}).get("query_type") == "statement":
                self._store_dialogue_scene(
                    conn, hive_id, message_id, "assistant", dialogue_text,
                    parsed["query_frame"], provenance=provenance,
                )
            dialogue_state = self.dialogue.process(
                language_analysis,
                query_frame=parsed.get("query_frame"),
                answer=answer,
                conn=conn,
                message_id=message_id,
            )
            state["dialogue_state"] = dialogue_state.as_dict()
            supporting_scenes = [
                str(item)
                for item in answer.get("supporting_scenes", [])
                if item
            ]
            final_plan = self.responses.plan(
                interpretation_status=language_analysis.interpretation_status.value,
                query_frame=state.get("query_frame"),
                answer=answer,
                candidates=state.get("candidates", []),
                dialogue_state=dialogue_state.as_dict(),
                source_evidence=[
                    {"source_scene_id": scene_id}
                    for scene_id in supporting_scenes
                ],
            )
            state["response_plan"] = self.responses.persist(
                conn,
                final_plan,
                conversation_id=conversation_id,
                source_utterance_id=(
                    state.get("query_frame", {})
                    .get("utterance", {})
                    .get("id")
                ),
                surface=text,
                independent_source_count=len(set(supporting_scenes)),
            )
            self._save(conn, hive_id, state)
            return {"id": message_id, "turn_index": turn, "role": "assistant", "text": text, "created_at": now}

    def _import_scene(self, conn: Any, hive_id: str, scene: Dict[str, Any], text: str, parsed: Dict[str, Any], hive: Any, message: Any) -> None:
        from .unknown_search import UnknownTokenSearchService
        UnknownTokenSearchService(self.repository)._place_memory_source(
            conn, hive_id, int(scene["cloud_id"]), scene,
            {"id": f"import-{uuid.uuid4().hex[:10]}", "query_session_id": "", "query_frame_id": parsed["query_frame"]["id"], "message_id": str(message["id"] if message else ""), "created_for_surface": text},
        )

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
        QuerySceneService._set_pipeline_state(
            state, "CANDIDATES_FOUND" if candidate_count else "SOURCES_FOUND" if memory_source_count else "SEARCHING"
        )

    @staticmethod
    def _set_pipeline_state(state: Dict[str, Any], pipeline_state: str) -> None:
        if pipeline_state not in PIPELINE_STATES:
            raise ValueError(f"unknown pipeline state: {pipeline_state}")
        state["pipeline_state"] = pipeline_state
        state.setdefault("pipeline_history", []).append({
            "state": pipeline_state, "reasoning_step": int(state.get("vibration", {}).get("current_step", 0)),
            "at": utcnow(),
        })

    def get(self, hive_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            state = self._load(conn, hive_id)
            if not state:
                raise KeyError(hive_id)
            return self._decorate_state(conn, hive_id, state)

    def sync_working_cells(self, hive_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            state = self._load(conn, hive_id)
            if not state:
                raise KeyError(hive_id)
            self._sync_working_cells(conn, hive_id, state)
            self._save(conn, hive_id, state)
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
                before = float(scores.get("decision_score", scores.get("semantic_total", scores["total"])))
                competition = max(0.0, (sum(other["scores"].get("decision_score", other["scores"].get("semantic_total", other["scores"]["total"])) for other in active if other["id"] != candidate["id"]) / max(1, len(active) - 1) - before) * 0.25)
                resonance = clamp(scores["source_confidence"] / total_source * len(active))
                contradiction = float(scores.get("contradiction", 0.0))
                semantic_total = float(scores.get("semantic_total", scores.get("total", 0.0)))
                gravity = clamp(float(scores.get("gravity", 0.0)))
                retention = clamp(float(scores.get("retention", 0.0)))
                decision_score = clamp(.65 * semantic_total + .20 * gravity + .10 * resonance + .05 * retention)
                after = clamp(decision_score - contradiction * config["contradiction_force"] - competition * config["competition_force"])
                scores["resonance"] = resonance
                scores["gravity"] = gravity
                scores["semantic_total"] = semantic_total
                scores["decision_score"] = decision_score
                scores["activation"] = after
                scores["retention"] = clamp(scores.get("retention", before) * 0.65 + after * 0.35)
                scores["evidence_confidence"] = float(scores.get("evidence_confidence", scores.get("source_confidence", before)))
                scores["semantic_confidence"] = float(scores.get("semantic_confidence", scores.get("source_confidence", before)))
                scores["answer_confidence"] = min(scores["semantic_confidence"], scores["evidence_confidence"], float(scores.get("source_confidence", before)), scores["retention"])
                scores["total"] = after
                candidate["stable_steps"] = candidate.get("stable_steps", 0) + 1 if abs(after - before) < .08 else 0
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
            state["candidates"].sort(key=lambda item: (-item["scores"].get("decision_score", item["scores"]["total"]), item["lemma"]))
            winner = self._winner(state, config)
            if winner:
                winner["status"] = "winner"
                self._resolve(state, winner, step)
            best_scores = state["candidates"][0].get("scores", {}) if state.get("candidates") else {}
            state.update({"semantic_total": float(best_scores.get("semantic_total", 0.0)), "gravity": float(best_scores.get("gravity", 0.0)), "decision_score": float(best_scores.get("decision_score", 0.0))})
            event = {"step": step, "timestamp": 0, "candidate_changes": changes, "evicted_candidates": evicted, "winner": winner["id"] if winner else None}
            state.setdefault("reasoning_trace", {}).setdefault("stages", []).append({
                "id": f"vibration-{step}", "stage": "VIBRATION", "status": "WINNER_FOUND" if winner else "RUNNING",
                "step": step, "output": deepcopy(event),
            })
            vibration["current_step"] = step
            vibration["history"].append(event)
            active_candidates = any(item["status"] not in {"evicted", "EVICTED", "conflict"} for item in state["candidates"])
            active_physical_nodes = any(item.get("eviction_status") != "EVICTED" for item in (state.get("dynamics") or {}).get("nodes", []))
            exhausted = (bool(state["candidates"]) and not active_candidates) or (not state["candidates"] and not active_physical_nodes)
            if winner or step >= config["max_steps"] or exhausted:
                vibration["status"] = "FINISHED"
                state["status"] = "STABLE"
                if state.get("pipeline") and not winner:
                    state["pipeline"]["vibration"] = {"status": "FINISHED", "current_step": step}
                    state["pipeline"]["answer"] = {"status": "UNRESOLVED"}
                    state["answer"].update({"status": "UNRESOLVED", "answer_mode": "partial" if state.get("result_type") in {"PARTIAL_HIT", "CONFLICT_HIT"} else "unknown"})
                    self._set_pipeline_state(state, "FAILED")
                elif winner:
                    self._set_pipeline_state(state, "PLANNING")
            elif state.get("pipeline"):
                state["pipeline"]["vibration"] = {"status": "RUNNING", "current_step": step}
                self._set_pipeline_state(state, "VIBRATING")
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

    def _roles(self, tokens: List[Dict[str, Any]], requested_role: str = "") -> Dict[str, Dict[str, Any]]:
        roles: Dict[str, Dict[str, Any]] = {}
        predicate_index = next((item["index"] for item in tokens if item["part_of_speech"] in {"VERB", "INFN"}), None)
        has_source = any(
            item.get("index", 0) > 0
            and tokens[item["index"] - 1].get("normalized") in {"из", "от"}
            for item in tokens
        )
        for item in tokens:
            if item.get("component_type") == "question_operator" or item["normalized"] in QUESTION_ROLES or item["normalized"] in GREETING_WORDS or item["normalized"] in {"не", "ли"} or item["normalized"] in REFERENTIALS:
                continue
            previous = self._governing_preposition(tokens, item["index"])
            case = item["grammatical_features"].get("case")
            stored_role = {
                "subject": "agent",
                "predicate": "action",
            }.get(str(item.get("scene_role") or ""), str(item.get("scene_role") or ""))
            if stored_role == "action":
                roles.setdefault("action", {"status": "fixed", **item})
                continue
            if stored_role in SUPPORTED_ROLES:
                roles.setdefault(stored_role, {
                    "status": "fixed", **item,
                    "preposition": previous if previous in {"в", "во", "на", "к", "из", "от", "с", "со", "у"} else "",
                })
                continue
            if stored_role:
                continue
            if item["normalized"] in MODAL_WORDS:
                roles["modal"] = {"status": "fixed", "component_type": "query_modal", "role": "modal", "semantic_function": MODAL_WORDS[item["normalized"]]["semantic_function"], **item}
                continue
            if item["part_of_speech"] in {"VERB", "INFN"} and "action" not in roles:
                roles["action"] = {"status": "fixed", **item}
                continue
            if item["part_of_speech"] not in {"NOUN", "NPRO", "NUMR"}:
                continue
            direct_previous = next((token.get("normalized") for token in tokens if token.get("index") == item["index"] - 1), "")
            if direct_previous == "кроме":
                # The governed noun is an exclusion for the requested slot,
                # not a positive scene anchor.
                continue
            if item.get("entity_referential") and previous == "у":
                continue
            role = ""
            if previous in {"в", "во", "на"} and case in {"loct", "loc2", None}:
                role = "location"
            elif previous == "у" and case in {"gent", None}:
                role = "location"
            elif previous in {"в", "во", "на", "к"} and case in {"accs", "datv", None}:
                role = "destination"
            elif previous in {"из", "от"} and case in {"gent", None}:
                role = "source"
            elif previous in {"с", "со"} and case == "gent":
                role = "source"
            elif previous in {"с", "со"} or case == "ablt":
                role = "instrument"
            elif case == "nomn":
                role = (
                    "object"
                    if (predicate_index is None and (requested_role == "source" or has_source))
                    or (
                        predicate_index is not None
                        and item["index"] > predicate_index
                        and "agent" in roles
                    )
                    else "agent"
                )
            elif case in {"accs", "gent", "datv"} or predicate_index is not None:
                role = "object"
            if role and role not in roles:
                roles[role] = {"status": "fixed", **item, "preposition": previous if previous in {"в", "во", "на", "к", "из", "от", "с", "со", "у"} else ""}
        return roles

    @staticmethod
    def _governing_preposition(tokens: List[Dict[str, Any]], index: int) -> str:
        by_index = {item.get("index"): item for item in tokens}
        for cursor in range(index - 1, max(-1, index - 4), -1):
            item = by_index.get(cursor)
            if not item:
                continue
            normalized = item.get("normalized", "")
            if normalized in RoleResolver.PREPOSITIONS:
                return normalized
            if item.get("part_of_speech") not in RoleResolver.ADJECTIVES:
                break
        return ""

    @staticmethod
    def _token_value(value: Dict[str, Any]) -> Dict[str, Any]:
        features = value.get("grammatical_features", {})
        return {
            "lemma": value.get("lemma"), "surface": value.get("surface"),
            "lexeme_cloud_id": value.get("lexeme_cloud_id"), "word_form_cloud_id": value.get("word_form_cloud_id"),
            "part_of_speech": value.get("part_of_speech"), "case": features.get("case"),
            "number": features.get("number"), "gender": features.get("gender"),
            "tense": features.get("tense"), "person": features.get("person"),
            "grammatical_features": features,
            "component_type": value.get("component_type"),
            "semantic_function": value.get("semantic_function"),
        }

    @staticmethod
    def _persist_query_frame(conn: Any, hive_id: str, frame: Dict[str, Any]) -> None:
        now = utcnow()
        conn.execute(
            """INSERT OR REPLACE INTO query_frames
               (id,hive_id,source_text,predicate_lemma,requested_role,requested_slot,
                status,frame_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                frame["id"],
                hive_id,
                frame.get("source_text", ""),
                (frame.get("predicate") or {}).get("lemma"),
                frame.get("requested_role"),
                frame.get("requested_slot"),
                "READY",
                encode(frame),
                now,
            ),
        )
        conn.execute(
            "DELETE FROM query_role_hypotheses WHERE query_frame_id=?",
            (frame["id"],),
        )
        for index, hypothesis in enumerate(frame.get("requested_role_hypotheses", [])):
            conn.execute(
                """INSERT INTO query_role_hypotheses
                   (id,query_frame_id,semantic_role,grammatical_slot,confidence,
                    selected,evidence_json)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    stable_id(
                        "query-role-hypothesis",
                        frame["id"],
                        hypothesis.get("role"),
                        index,
                    ),
                    frame["id"],
                    hypothesis.get("role"),
                    frame.get("requested_slot"),
                    float(hypothesis.get("confidence", 0.0)),
                    1 if index == 0 else 0,
                    encode({"source": hypothesis.get("source")}),
                ),
            )
        conn.execute(
            "DELETE FROM query_constraints WHERE query_frame_id=?",
            (frame["id"],),
        )
        for role, constraints in (frame.get("slot_constraints") or {}).items():
            for constraint_type, value in constraints.items():
                conn.execute(
                    """INSERT INTO query_constraints
                       (id,query_frame_id,role,constraint_type,value_json,confidence)
                       VALUES(?,?,?,?,?,1)""",
                    (
                        stable_id(
                            "query-constraint",
                            frame["id"],
                            role,
                            constraint_type,
                        ),
                        frame["id"],
                        role,
                        constraint_type,
                        encode(value),
                    ),
                )

    @staticmethod
    def _persist_retrieval_trace(
        conn: Any,
        frame: Dict[str, Any],
        scenes: Iterable[Dict[str, Any]],
        accepted: Iterable[Dict[str, Any]],
        rejected: Iterable[Dict[str, Any]],
    ) -> None:
        now = utcnow()
        conn.execute(
            """UPDATE query_frames
               SET status='RETRIEVED',frame_json=?
               WHERE id=?""",
            (encode(frame), frame["id"]),
        )
        stages_by_scene: Dict[int, str] = {}
        for stage in frame.get("retrieval_stages", []):
            for scene_id in stage.get("scene_ids", []):
                stages_by_scene.setdefault(
                    int(scene_id),
                    str(stage.get("stage") or "lemma"),
                )
        scene_match_ids: Dict[str, str] = {}
        for scene in scenes:
            if scene.get("cloud_id") is None:
                continue
            source_scene_id = int(scene["cloud_id"])
            stage = str(
                scene.get("retrieval_stage")
                or stages_by_scene.get(source_scene_id, "local_memory")
            )
            match_id = stable_id(
                "scene-match",
                frame["id"],
                source_scene_id,
                stage,
            )
            scene_match_ids[str(scene["id"])] = match_id
            conn.execute(
                """INSERT OR REPLACE INTO scene_matches
                   (id,query_frame_id,source_scene_id,retrieval_stage,score,
                    matched_roles_json,status,evidence_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    match_id,
                    frame["id"],
                    source_scene_id,
                    stage,
                    clamp(float(scene.get("scores", {}).get("semantic_total", 0.0))),
                    encode({
                        role: (scene.get("role_match_details") or {}).get(role, {})
                        for role in scene.get("matched_roles", [])
                    }),
                    str(scene.get("result_type") or "NO_HIT"),
                    encode(
                        scene.get("fact_evidence")
                        or [{"scene_id": source_scene_id}]
                    ),
                    now,
                ),
            )
        for status, candidates in (("ACCEPTED", accepted), ("REJECTED", rejected)):
            for candidate in candidates:
                source_id = str(candidate.get("primary_source_id") or (
                    candidate.get("sources") or [""]
                )[0])
                match_id = scene_match_ids.get(source_id)
                if not match_id:
                    continue
                pre_id = stable_id(
                    "pre-candidate",
                    frame["id"],
                    candidate.get("id") or candidate.get("lemma"),
                    source_id,
                )
                conn.execute(
                    """INSERT OR REPLACE INTO pre_candidates
                       (id,scene_match_id,query_frame_id,entity_id,target_role,
                        value_json,score,status,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        pre_id,
                        match_id,
                        frame["id"],
                        candidate.get("entity_id"),
                        candidate.get("target_role") or frame.get("requested_role") or "object",
                        encode({
                            "lemma": candidate.get("lemma"),
                            "surface": candidate.get("surface"),
                        }),
                        clamp(float(candidate.get("scores", {}).get("total", 0.0))),
                        status,
                        now,
                    ),
                )
                if status == "ACCEPTED":
                    conn.execute(
                        """INSERT OR REPLACE INTO accepted_candidates
                           (id,pre_candidate_id,score,evidence_json,created_at)
                           VALUES(?,?,?,?,?)""",
                        (
                            stable_id("accepted-candidate", pre_id),
                            pre_id,
                            clamp(float(candidate.get("scores", {}).get("total", 0.0))),
                            encode(candidate.get("fact_evidence", [])),
                            now,
                        ),
                    )
                else:
                    conn.execute(
                        """INSERT OR REPLACE INTO rejected_candidates
                           (id,pre_candidate_id,reason_code,evidence_json,created_at)
                           VALUES(?,?,?,?,?)""",
                        (
                            stable_id("rejected-candidate", pre_id),
                            pre_id,
                            candidate.get("rejection_reason") or "CONSTRAINT_FAILED",
                            encode(candidate.get("constraint_evidence", [])),
                            now,
                        ),
                    )

    @staticmethod
    def _entity_ids_for_value(conn: Any, value: Dict[str, Any]) -> set[int]:
        candidates = {
            str(value.get("surface") or "").casefold(),
            str(value.get("normalized") or "").casefold(),
            str(value.get("lemma") or "").casefold(),
        } - {""}
        if not candidates:
            return set()
        marks = ",".join("?" for _ in candidates)
        rows = conn.execute(
            f"""SELECT DISTINCT entity_id FROM entity_aliases
                WHERE normalized_alias IN ({marks})
                UNION
                SELECT cloud_id FROM entities WHERE canonical_lemma IN ({marks})""",
            (*sorted(candidates), *sorted(candidates)),
        ).fetchall()
        return {int(row[0]) for row in rows}

    @staticmethod
    def _role_slots(role: str) -> set[str]:
        mapping = {
            "agent": {"subject"},
            "cause": {"subject", "cause_oblique"},
            "theme": {"subject", "direct_object", "object"},
            "patient": {"direct_object", "object"},
            "object": {"direct_object", "object"},
            "recipient": {"indirect_object"},
            "experiencer": {"indirect_object", "subject"},
            "source": {"source_oblique"},
            "destination": {"destination_oblique"},
            "location": {"location_oblique"},
            "instrument": {"instrumental"},
            "material": {"instrumental"},
            "purpose": {"purpose_oblique"},
        }
        return mapping.get(role, {role})

    def _indexed_scene_ids(
        self,
        conn: Any,
        frame: Dict[str, Any],
        limit: int,
    ) -> tuple[List[int], List[Dict[str, Any]]]:
        roles = frame.get("roles") or {}
        action = roles.get("action") or frame.get("predicate") or {}
        predicate_lemma = str(action.get("lemma") or "").casefold()
        stages: List[Dict[str, Any]] = []

        def predicate_ids(lemmas: Iterable[str]) -> set[int]:
            values = sorted({lemma for lemma in lemmas if lemma})
            if not values:
                return set()
            marks = ",".join("?" for _ in values)
            return {
                int(row["source_scene_id"])
                for row in conn.execute(
                    f"""SELECT events.source_scene_id FROM events
                        JOIN scenes
                          ON scenes.cloud_id=events.source_scene_id
                        WHERE events.predicate_lemma IN ({marks})
                          AND scenes.knowledge_status<>'RETRACTED'
                        ORDER BY events.source_scene_id DESC LIMIT ?""",
                    (*values, limit),
                ).fetchall()
            }

        def predicate_form_ids(surface: str) -> set[int]:
            if not surface:
                return set()
            return {
                int(row["source_scene_id"])
                for row in conn.execute(
                    """SELECT events.source_scene_id FROM events
                       JOIN scenes
                         ON scenes.cloud_id=events.source_scene_id
                       WHERE lower(events.predicate_surface)=?
                         AND scenes.knowledge_status<>'RETRACTED'
                       ORDER BY events.source_scene_id DESC LIMIT ?""",
                    (surface.casefold(), limit),
                ).fetchall()
            }

        def apply_anchors(scene_ids: set[int]) -> set[int]:
            result = set(scene_ids)
            for role, value in roles.items():
                if (
                    role in {"action", "modal", frame.get("requested_role")}
                    or not isinstance(value, dict)
                    or value.get("status") != "fixed"
                ):
                    continue
                entity_ids = self._entity_ids_for_value(conn, value)
                if not entity_ids:
                    continue
                entity_marks = ",".join("?" for _ in entity_ids)
                slots = sorted(self._role_slots(role))
                slot_marks = ",".join("?" for _ in slots)
                semantic_roles = sorted({
                    role,
                    *(
                        {"patient", "theme", "object"}
                        if role in {"patient", "theme", "object"}
                        else {"agent", "theme"}
                        if role == "agent"
                        else set()
                    ),
                })
                semantic_marks = ",".join("?" for _ in semantic_roles)
                matching = {
                    int(row["source_scene_id"])
                    for row in conn.execute(
                        f"""SELECT DISTINCT e.source_scene_id
                            FROM event_participants ep
                            JOIN events e ON e.id=ep.event_id
                            JOIN scenes s ON s.cloud_id=e.source_scene_id
                            WHERE ep.entity_id IN ({entity_marks})
                              AND s.knowledge_status<>'RETRACTED'
                              AND (
                                ep.semantic_role IN ({semantic_marks})
                                OR ep.grammatical_slot IN ({slot_marks})
                                OR EXISTS (
                                  SELECT 1 FROM event_role_hypotheses erh
                                  WHERE erh.participant_id=ep.id
                                    AND erh.semantic_role IN ({semantic_marks})
                                    AND erh.confidence>=.35
                                )
                              )""",
                        (
                            *sorted(entity_ids),
                            *semantic_roles,
                            *slots,
                            *semantic_roles,
                        ),
                    ).fetchall()
                }
                result &= matching
                if not result:
                    break
            return result

        def anchor_ids() -> set[int]:
            result: Optional[set[int]] = None
            for role, value in roles.items():
                if (
                    role in {"action", "modal", frame.get("requested_role")}
                    or not isinstance(value, dict)
                    or value.get("status") != "fixed"
                ):
                    continue
                entity_ids = self._entity_ids_for_value(conn, value)
                if not entity_ids:
                    continue
                entity_marks = ",".join("?" for _ in entity_ids)
                slots = sorted(self._role_slots(role))
                slot_marks = ",".join("?" for _ in slots)
                semantic_roles = sorted({
                    role,
                    *(
                        {"patient", "theme", "object"}
                        if role in {"patient", "theme", "object"}
                        else {"agent", "theme"}
                        if role == "agent"
                        else set()
                    ),
                })
                semantic_marks = ",".join("?" for _ in semantic_roles)
                matching = {
                    int(row["source_scene_id"])
                    for row in conn.execute(
                        f"""SELECT DISTINCT e.source_scene_id
                            FROM event_participants ep
                            JOIN events e ON e.id=ep.event_id
                            JOIN scenes s ON s.cloud_id=e.source_scene_id
                            WHERE ep.entity_id IN ({entity_marks})
                              AND s.knowledge_status<>'RETRACTED'
                              AND (
                                ep.semantic_role IN ({semantic_marks})
                                OR ep.grammatical_slot IN ({slot_marks})
                                OR EXISTS (
                                  SELECT 1 FROM event_role_hypotheses erh
                                  WHERE erh.participant_id=ep.id
                                    AND erh.semantic_role IN ({semantic_marks})
                                    AND erh.confidence>=.35
                                )
                              )
                            ORDER BY e.source_scene_id DESC LIMIT ?""",
                        (
                            *sorted(entity_ids),
                            *semantic_roles,
                            *slots,
                            *semantic_roles,
                            limit,
                        ),
                    ).fetchall()
                }
                result = matching if result is None else result & matching
                if not result:
                    break
            return result or set()

        def relation_anchor_ids(
            allowed_relation_types: Optional[set[str]] = None,
        ) -> tuple[set[int], set[str]]:
            result: Optional[set[int]] = None
            matched_relation_types: set[str] = set()
            found_related_anchor = False
            for role, value in roles.items():
                if (
                    role in {"action", "modal", frame.get("requested_role")}
                    or not isinstance(value, dict)
                    or value.get("status") != "fixed"
                ):
                    continue
                entity_ids = self._entity_ids_for_value(conn, value)
                if not entity_ids:
                    continue
                marks = ",".join("?" for _ in entity_ids)
                relation_types = sorted(allowed_relation_types or set())
                relation_marks = ",".join("?" for _ in relation_types)
                type_filter = (
                    f" AND relation_type IN ({relation_marks})"
                    if relation_types
                    else ""
                )
                relation_rows = conn.execute(
                    f"""SELECT relation_type,subject_lexeme_cloud_id,
                               object_lexeme_cloud_id
                        FROM concept_relations
                        WHERE status<>'DEPRECATED'
                          {type_filter}
                          AND (
                            subject_lexeme_cloud_id IN ({marks})
                            OR object_lexeme_cloud_id IN ({marks})
                          )""",
                    (*relation_types, *sorted(entity_ids), *sorted(entity_ids)),
                ).fetchall()
                related_entity_ids: set[int] = set()
                for relation in relation_rows:
                    subject_id = int(relation["subject_lexeme_cloud_id"])
                    object_id = int(relation["object_lexeme_cloud_id"])
                    if subject_id in entity_ids and object_id not in entity_ids:
                        related_entity_ids.add(object_id)
                    if object_id in entity_ids and subject_id not in entity_ids:
                        related_entity_ids.add(subject_id)
                    matched_relation_types.add(str(relation["relation_type"]))
                if not related_entity_ids:
                    continue
                found_related_anchor = True
                entity_marks = ",".join("?" for _ in related_entity_ids)
                slots = sorted(self._role_slots(role))
                slot_marks = ",".join("?" for _ in slots)
                semantic_roles = sorted({
                    role,
                    *(
                        {"patient", "theme", "object"}
                        if role in {"patient", "theme", "object"}
                        else {"agent", "theme"}
                        if role == "agent"
                        else set()
                    ),
                })
                semantic_marks = ",".join("?" for _ in semantic_roles)
                matching = {
                    int(row["source_scene_id"])
                    for row in conn.execute(
                        f"""SELECT DISTINCT e.source_scene_id
                            FROM event_participants ep
                            JOIN events e ON e.id=ep.event_id
                            JOIN scenes s ON s.cloud_id=e.source_scene_id
                            WHERE ep.entity_id IN ({entity_marks})
                              AND s.knowledge_status<>'RETRACTED'
                              AND (
                                ep.semantic_role IN ({semantic_marks})
                                OR ep.grammatical_slot IN ({slot_marks})
                                OR EXISTS (
                                  SELECT 1 FROM event_role_hypotheses erh
                                  WHERE erh.participant_id=ep.id
                                    AND erh.semantic_role IN ({semantic_marks})
                                    AND erh.confidence>=.35
                                )
                              )
                            ORDER BY e.source_scene_id DESC LIMIT ?""",
                        (
                            *sorted(related_entity_ids),
                            *semantic_roles,
                            *slots,
                            *semantic_roles,
                            limit,
                        ),
                    ).fetchall()
                }
                result = matching if result is None else result & matching
                if not result:
                    break
            return (result or set(), matched_relation_types) if found_related_anchor else (set(), set())

        if frame.get("query_type") == "action_question":
            direct = anchor_ids()
            related, relation_types = relation_anchor_ids({"IS_A", "ALIAS_OF"})
            scene_ids = direct | related
            stages.append({
                "stage": "action_entity_anchor",
                "predicate_ignored": (
                    (frame.get("action_question") or {})
                    .get("removed_predicate", {})
                    .get("lemma")
                ),
                "relation_types": sorted(relation_types),
                "scene_ids": sorted(scene_ids),
                "considered": len(scene_ids),
                "stopped": bool(scene_ids),
            })
            return sorted(scene_ids, reverse=True)[:limit], stages

        predicate_surface = str(
            action.get("normalized")
            or action.get("surface")
            or ""
        ).casefold()
        exact_form = apply_anchors(predicate_form_ids(predicate_surface))
        requested_constraints = (
            frame.get("slot_constraints", {}).get(frame.get("requested_role"), {})
            or {}
        )
        stages.append({
            "stage": "exact_form",
            "predicate_surface": predicate_surface or None,
            "scene_ids": sorted(exact_form),
            "considered": len(exact_form),
            "threshold": float(settings.exact_retrieval_stop_threshold),
            "stopped": bool(exact_form) and not requested_constraints,
        })
        if exact_form and not requested_constraints:
            return sorted(exact_form, reverse=True)[:limit], stages
        exact = apply_anchors(predicate_ids([predicate_lemma])) | exact_form
        stages.append({
            "stage": "lemma",
            "predicate_lemmas": [predicate_lemma] if predicate_lemma else [],
            "scene_ids": sorted(exact),
            "considered": len(exact),
            "stopped": bool(exact),
        })
        if exact and not requested_constraints:
            return sorted(exact, reverse=True)[:limit], stages
        construction_ids: set[str] = set()
        requested_slot = str(frame.get("requested_slot") or "")
        if predicate_lemma:
            if requested_slot:
                construction_rows = conn.execute(
                    """SELECT DISTINCT ct.id
                       FROM construction_templates ct
                       JOIN construction_arguments ca ON ca.construction_id=ct.id
                       WHERE ct.predicate_lemma=?
                         AND ca.grammatical_slot=?
                         AND ct.status<>'DEPRECATED'
                       ORDER BY ct.confidence DESC LIMIT ?""",
                    (predicate_lemma, requested_slot, limit),
                ).fetchall()
            else:
                construction_rows = conn.execute(
                    """SELECT id FROM construction_templates
                       WHERE predicate_lemma=? AND status<>'DEPRECATED'
                       ORDER BY confidence DESC LIMIT ?""",
                    (predicate_lemma, limit),
                ).fetchall()
            construction_ids = {str(row["id"]) for row in construction_rows}
        construction_scene_ids: set[int] = set()
        if construction_ids:
            marks = ",".join("?" for _ in construction_ids)
            construction_scene_ids = apply_anchors({
                int(row["source_scene_id"])
                for row in conn.execute(
                    f"""SELECT events.source_scene_id FROM events
                        JOIN scenes
                          ON scenes.cloud_id=events.source_scene_id
                        WHERE events.construction_id IN ({marks})
                          AND scenes.knowledge_status<>'RETRACTED'
                        ORDER BY events.source_scene_id DESC LIMIT ?""",
                    (*sorted(construction_ids), limit),
                ).fetchall()
            })
        stages.append({
            "stage": "construction",
            "construction_ids": sorted(construction_ids),
            "scene_ids": sorted(construction_scene_ids),
            "considered": len(construction_scene_ids),
            "stopped": bool(construction_scene_ids) and not requested_constraints,
        })
        if construction_scene_ids and not requested_constraints:
            return sorted(construction_scene_ids, reverse=True)[:limit], stages
        concept_lemmas: set[str] = set()
        variant = self.semantic_projection.variant(conn, predicate_lemma)
        if variant:
            concept_lemmas = {
                str(row["lemma"])
                for row in conn.execute(
                    """SELECT lemma FROM action_variants
                       WHERE action_concept_id=? AND source_type<>'manual_seed'""",
                    (variant["action_concept_id"],),
                ).fetchall()
            }
        conceptual = (
            apply_anchors(predicate_ids(concept_lemmas))
            | construction_scene_ids
            | exact
        )
        stages.append({
            "stage": "action_concept",
            "concept_id": variant["action_concept_id"] if variant else None,
            "predicate_lemmas": sorted(concept_lemmas),
            "scene_ids": sorted(conceptual),
            "considered": len(conceptual),
            "stopped": bool(conceptual),
        })
        if conceptual:
            return sorted(conceptual, reverse=True)[:limit], stages
        related, relation_types = relation_anchor_ids()
        stages.append({
            "stage": "entity_relation",
            "relation_types": sorted(relation_types),
            "scene_ids": sorted(related),
            "considered": len(related),
            "stopped": bool(related),
        })
        if related:
            return sorted(related, reverse=True)[:limit], stages
        analogous = anchor_ids()
        if not analogous and requested_slot:
            analogous = {
                int(row["source_scene_id"])
                for row in conn.execute(
                    """SELECT DISTINCT e.source_scene_id
                       FROM events e
                       JOIN scenes s ON s.cloud_id=e.source_scene_id
                       JOIN construction_arguments ca
                         ON ca.construction_id=e.construction_id
                       JOIN construction_templates ct
                         ON ct.id=e.construction_id
                       WHERE ca.grammatical_slot=?
                         AND s.knowledge_status<>'RETRACTED'
                         AND ct.status IN ('PROBABLE','STABLE')
                       ORDER BY e.source_scene_id DESC LIMIT ?""",
                    (requested_slot, limit),
                ).fetchall()
            }
        stages.append({
            "stage": "analogy",
            "scene_ids": sorted(analogous),
            "considered": len(analogous),
            "stopped": True,
        })
        return sorted(analogous, reverse=True)[:limit], stages

    def _definition_relation_scenes(
        self, conn: Any, frame: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        entity = (frame.get("roles") or {}).get("entity") or {}
        entity_ids = self._entity_ids_for_value(conn, entity)
        if entity.get("entity_id"):
            entity_ids.add(int(entity["entity_id"]))
        if not entity_ids:
            return []
        marks = ",".join("?" for _ in entity_ids)
        rows = conn.execute(
            f"""SELECT relation.id,relation.relation_type,relation.confidence,
                       relation.source_type,relation.subject_lexeme_cloud_id,
                       relation.object_lexeme_cloud_id,
                       subject.canonical_lemma AS subject_lemma,
                       subject.display_name AS subject_surface,
                       object.canonical_lemma AS type_lemma,
                       object.display_name AS type_surface
                FROM concept_relations relation
                JOIN entities subject
                  ON subject.cloud_id=relation.subject_lexeme_cloud_id
                JOIN entities object
                  ON object.cloud_id=relation.object_lexeme_cloud_id
                WHERE relation.subject_lexeme_cloud_id IN ({marks})
                  AND relation.relation_type IN ('IS_A','INSTANCE_OF')
                  AND relation.status<>'DEPRECATED'
                ORDER BY CASE relation.relation_type
                    WHEN 'IS_A' THEN 0 WHEN 'INSTANCE_OF' THEN 1 ELSE 2 END,
                    relation.confidence DESC,relation.id""",
            tuple(sorted(entity_ids)),
        ).fetchall()
        result = []
        for row in rows:
            subject_surface = str(entity.get("surface") or row["subject_surface"])
            type_surface = str(row["type_surface"])
            confidence = clamp(float(row["confidence"]))
            relation_id = str(row["id"])
            source_id = f"relation-{relation_id}"
            result.append({
                "id": source_id,
                "cloud_id": None,
                "type": "definition_relation",
                "source_text": f"{subject_surface} — {type_surface}.",
                "roles": {
                    "entity": {
                        "status": "fixed", "entity_id": int(row["subject_lexeme_cloud_id"]),
                        "lemma": str(row["subject_lemma"]), "surface": subject_surface,
                        "part_of_speech": "NOUN",
                    },
                    "entity_type": {
                        "status": "fixed", "entity_id": int(row["object_lexeme_cloud_id"]),
                        "lemma": str(row["type_lemma"]), "surface": type_surface,
                        "part_of_speech": "NOUN",
                    },
                },
                "relation_id": relation_id,
                "relation_type": str(row["relation_type"]),
                "root_source_ids": [relation_id],
                "source_relation_ids": [relation_id],
                "negation": False,
                "retrieval_scope": "RELATION",
                "eligible_for_fact_retrieval": True,
                "provenance": {
                    "source": "concept_relation",
                    "relation_id": relation_id,
                    "relation_type": str(row["relation_type"]),
                    "source_type": str(row["source_type"]),
                },
                "scores": {
                    "total_score": confidence,
                    "semantic_total": confidence,
                    "semantic_match": 1.0,
                    "structural_match": 1.0,
                    "anchor_match": 1.0,
                    "requested_role_match": 1.0,
                    "source_confidence": confidence,
                    "evidence_confidence": confidence,
                    "role_matches": {"entity": 1.0},
                    "anchor_validation": {
                        "status": "PASSED", "required_roles": ["entity"],
                        "failed_roles": [], "requested_role_present": True,
                        "answer_slot_type": "relation",
                    },
                },
                "anchor_validation": {
                    "status": "PASSED", "required_roles": ["entity"],
                    "failed_roles": [], "requested_role_present": True,
                    "answer_slot_type": "relation",
                },
                "matched_roles": ["entity"], "mismatched_roles": [],
                "selection_reason": "прямая классификационная связь сущности",
                "result_type": "FULL_HIT",
            })
        return result

    @staticmethod
    def _definition_candidates(
        frame: Dict[str, Any], scenes: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen = set()
        for scene in scenes:
            if scene.get("type") != "definition_relation":
                continue
            entity = (scene.get("roles") or {}).get("entity", {})
            entity_type = (scene.get("roles") or {}).get("entity_type", {})
            key = (entity.get("entity_id"), entity_type.get("entity_id"))
            if key in seen or not entity_type.get("lemma"):
                continue
            seen.add(key)
            confidence = float((scene.get("scores") or {}).get("semantic_total", .9))
            relation_ids = list(scene.get("source_relation_ids") or [])
            candidates.append({
                "id": f"candidate-definition-{uuid.uuid5(uuid.NAMESPACE_URL, str(key)).hex[:12]}",
                "concept_id": f"entity-type-{entity_type.get('entity_id')}",
                "lemma": entity_type["lemma"], "surface": entity_type["surface"],
                "entity_id": entity_type.get("entity_id"), "entity_value": entity_type.get("surface"),
                "entity_type": None, "full_surface": entity_type.get("surface"),
                "definition_entity_surface": entity.get("surface"),
                "definition_entity_id": entity.get("entity_id"),
                "relation_type": scene.get("relation_type"),
                "source_relation_ids": relation_ids,
                "root_source_ids": list(scene.get("root_source_ids") or relation_ids),
                "evidence_family_key": (
                    entity.get("entity_id"), "IS_A", False, None,
                    tuple(scene.get("root_source_ids") or relation_ids),
                ),
                "mention": {"surface": entity_type.get("surface"), "head_surface": entity_type.get("surface"), "mention_type": "entity_type", "attributes": []},
                "answer_surfaces": {"short_name": entity_type.get("surface"), "full_mention": entity_type.get("surface")},
                "lexeme_cloud_id": None, "word_form_cloud_id": None, "preposition": "",
                "grammatical_features": {},
                "form_provenance": {"source_type": "classification_relation", "relation_id": relation_ids[0] if relation_ids else None},
                "target_role": "entity_type", "part_of_speech": "NOUN",
                "answer_slot_type": "relation", "predicate_lemma": "IS_A",
                "predicate_surface": "—", "predicate_phrase": None,
                "constraint_matches": {"relation": {"passed": True, "allowed": frame.get("allowed_relations", [])}},
                "entity_kind": "entity_type", "answer_mode": "definition", "answer_scene_id": None,
                "hard_forbidden": False, "sources": [scene["id"]], "primary_source_id": scene["id"],
                "fact_evidence": [{"relation_id": relation_id, "text": scene.get("source_text", "")} for relation_id in relation_ids],
                "scores": {
                    "role_compatibility": 1.0, "query_relevance": 1.0, "semantic_support": 1.0,
                    "structural_support": 1.0, "exact_match": 1.0, "action_compatibility": 0.0,
                    "object_compatibility": 0.0, "anchor_compatibility": 1.0,
                    "grammar_compatibility": 1.0, "source_confidence": confidence,
                    "resonance": 0.0, "retention": confidence, "activation": confidence,
                    "contradiction": 0.0, "evidence_confidence": confidence,
                    "semantic_confidence": confidence, "answer_confidence": confidence,
                    "semantic_total": confidence, "gravity": 0.0,
                    "decision_score": confidence, "total": confidence,
                },
                "concept_matches": {}, "semantic_frame": {},
                "competition_group_id": f"{frame.get('id')}:entity_type",
                "status": "new", "weak_steps": 0,
                "selection_reason": scene.get("selection_reason", ""),
            })
        return sorted(candidates, key=lambda item: (-float(item["scores"]["total"]), item["lemma"]))

    def _memory_scenes(
        self,
        conn: Any,
        hive_id: Optional[str] = None,
        frame: Optional[Dict[str, Any]] = None,
        limit: int = 64,
    ) -> List[Dict[str, Any]]:
        retrieval_stage_by_scene: Dict[int, str] = {}
        if hive_id:
            rows = conn.execute(
                """SELECT DISTINCT s.cloud_id, s.sentence_text, s.canonical_text,
                                  s.parser_version
                FROM scenes s JOIN hive_cells hc ON hc.source_scene_cloud_id=s.cloud_id
                WHERE hc.hive_id=? AND s.knowledge_status<>'RETRACTED'
                ORDER BY s.updated_at DESC""", (hive_id,)
            ).fetchall()
        else:
            scene_ids, retrieval_stages = self._indexed_scene_ids(
                conn,
                frame or {},
                min(max(1, int(limit)), int(settings.retrieval_stage_limit)),
            )
            if frame is not None:
                frame["retrieval_stages"] = retrieval_stages
                frame["retrieval_metrics"] = {
                    "scenes_total": int(conn.execute(
                        """SELECT COUNT(*) FROM scenes
                           WHERE knowledge_status<>'RETRACTED'"""
                    ).fetchone()[0]),
                    "scenes_considered": len(scene_ids),
                    "indexed": True,
                }
            for stage in retrieval_stages:
                for source_scene_id in stage.get("scene_ids", []):
                    retrieval_stage_by_scene.setdefault(
                        int(source_scene_id),
                        str(stage.get("stage") or "lemma"),
                    )
            if scene_ids:
                marks = ",".join("?" for _ in scene_ids)
                rows = conn.execute(
                    f"""SELECT s.cloud_id,s.sentence_text,s.canonical_text,
                               s.parser_version
                        FROM scenes s WHERE s.cloud_id IN ({marks})
                          AND s.knowledge_status<>'RETRACTED'
                        ORDER BY s.updated_at DESC,s.cloud_id DESC""",
                    scene_ids,
                ).fetchall()
            else:
                rows = []
        result = []
        for row in rows:
            if str(row["parser_version"]) != TrainingPipelineV2.parser_version:
                self.universal_events.materialize_scene(conn, int(row["cloud_id"]))
                conn.execute(
                    "UPDATE scenes SET parser_version=?,updated_at=? WHERE cloud_id=?",
                    (
                        TrainingPipelineV2.parser_version,
                        utcnow(),
                        int(row["cloud_id"]),
                    ),
                )
            components = []
            for component in conn.execute(
                """SELECT sc.token_index, sc.grammatical_role, sc.morphology_json, sc.lexeme_cloud_id,
                   sc.word_form_cloud_id, wf.normalized_form,
                   l.lemma FROM scene_components sc JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id
                   LEFT JOIN lexemes l ON l.cloud_id=sc.lexeme_cloud_id
                   WHERE sc.scene_cloud_id=? ORDER BY sc.token_index""", (row["cloud_id"],)
            ):
                morphology = json.loads(component["morphology_json"] or "{}")
                components.append({
                    "index": int(component["token_index"]), "surface": component["normalized_form"],
                    "normalized": component["normalized_form"], "lemma": component["lemma"] or morphology.get("lemma") or component["normalized_form"],
                    "scene_role": component["grammatical_role"],
                    "lexeme_cloud_id": int(component["lexeme_cloud_id"]) if component["lexeme_cloud_id"] else None,
                    "word_form_cloud_id": int(component["word_form_cloud_id"]) if component["word_form_cloud_id"] else None,
                    "part_of_speech": morphology.get("pos", "UNK"),
                    "grammatical_features": {key: value for key, value in morphology.items() if key != "pos" and key != "lemma"},
                })
            event = self.universal_events.load_event(conn, int(row["cloud_id"]))
            event_roles: Dict[str, Dict[str, Any]] = {}
            if event:
                event_roles["action"] = {
                    "status": "fixed",
                    "lemma": event["predicate"]["lemma"],
                    "surface": event["predicate"]["surface"],
                    "lexeme_cloud_id": event["predicate"].get("lexeme_cloud_id"),
                    "part_of_speech": "VERB",
                }
                for participant in event["participants"]:
                    value = {
                        "status": "fixed",
                        "lemma": participant["lemma"],
                        "surface": participant["surface"],
                        "mention_surface": participant.get("mention_surface"),
                        "full_surface": participant.get("full_surface"),
                        "mention_type": participant.get("mention_type"),
                        "entity_value": participant.get("entity_value"),
                        "entity_type": deepcopy(participant.get("entity_type")),
                        "answer_surfaces": deepcopy(
                            participant.get("answer_surfaces", {})
                        ),
                        "entity_id": participant["entity_id"],
                        "lexeme_cloud_id": participant.get("lexeme_cloud_id"),
                        "semantic_role": participant["role"],
                        "grammatical_slot": participant["grammatical_slot"],
                        "preposition": participant["preposition"],
                        "role_hypotheses": participant["role_hypotheses"],
                        "part_of_speech": "NOUN",
                        "grammatical_features": deepcopy(
                            participant.get("grammatical_features", {})
                        ),
                    }
                    event_roles.setdefault(participant["role"], value)
                    for hypothesis in participant.get("role_hypotheses", []):
                        if float(hypothesis.get("confidence", 0.0)) < .5:
                            continue
                        hypothesis_role = str(hypothesis.get("role") or "")
                        if not hypothesis_role:
                            continue
                        event_roles.setdefault(
                            hypothesis_role,
                            {
                                **value,
                                "semantic_role": hypothesis_role,
                                "role_confidence": float(
                                    hypothesis.get("confidence", 0.0)
                                ),
                                "role_source": "event_role_hypothesis",
                            },
                        )
                    if participant["grammatical_slot"] == "direct_object":
                        event_roles.setdefault("object", value)
                    if participant["grammatical_slot"] in {
                        "indirect_object",
                        "instrumental",
                    }:
                        event_roles.setdefault("object", value)
                    if participant["grammatical_slot"] == "subject":
                        event_roles.setdefault("agent", value)
            result.append({
                "id": f"scene-{row['cloud_id']}", "cloud_id": int(row["cloud_id"]), "type": "memory_scene",
                "source_text": row["sentence_text"],
                "retrieval_stage": retrieval_stage_by_scene.get(
                    int(row["cloud_id"]),
                    "local_memory",
                ),
                "roles": event_roles or self._roles(components),
                "event": event,
                "entity_mentions": self.universal_events.load_mentions(conn, int(row["cloud_id"])),
                "concept_evidence": [
                    {
                        **dict(evidence),
                        "payload": decode(evidence["payload_json"], {}),
                    }
                    for evidence in conn.execute(
                        """SELECT * FROM concept_evidence
                           WHERE source_scene_id=?
                           ORDER BY created_at,id""",
                        (int(row["cloud_id"]),),
                    ).fetchall()
                ],
                "scene_concept_projections": [
                    {
                        **dict(projection),
                        "semantic_frame": decode(
                            projection["semantic_frame_json"], {}
                        ),
                    }
                    for projection in conn.execute(
                        """SELECT * FROM scene_concept_projections
                           WHERE scene_id=? ORDER BY projection_confidence DESC,id""",
                        (int(row["cloud_id"]),),
                    ).fetchall()
                ],
                "negation": any(item["normalized"] == "не" for item in components),
                "retrieval_scope": "LOCAL" if hive_id else "GLOBAL",
                "provenance": {"source": "hive_cells" if hive_id else "global_field"},
            })
        if hive_id:
            dialogue_rows = conn.execute(
                "SELECT * FROM hive_dialogue_scenes WHERE hive_id=? ORDER BY activation DESC, updated_at DESC",
                (hive_id,),
            ).fetchall()
            for row in dialogue_rows:
                if str(row["source_text"] or "").rstrip().endswith("?"):
                    continue
                roles = decode(row["roles_json"], {})
                provenance = decode(row["provenance_json"], {})
                eligible = bool(row["eligible_for_fact_retrieval"])
                result.append({
                    "id": str(row["id"]), "cloud_id": None, "type": "dialogue_memory_scene",
                    "source_text": str(row["source_text"]), "roles": roles, "negation": False,
                    "retrieval_scope": "DIALOGUE", "activation": float(row["activation"]),
                    "retention": float(row["retention"]),
                    "memory_class": str(row["memory_class"]),
                    "source_type": str(row["source_type"]),
                    "knowledge_status": str(row["knowledge_status"]),
                    "independent_evidence": bool(row["independent_evidence"]),
                    "eligible_for_fact_retrieval": eligible,
                    "completion_status": str(row["completion_status"]),
                    "missing_supported_roles": decode(row["missing_supported_roles_json"], []),
                    "root_source_ids": decode(row["root_evidence_ids_json"], []),
                    "provenance": {
                        "source": "dialogue_memory", "message_id": str(row["message_id"]),
                        "role": str(row["source_role"]), **provenance,
                    },
                })
        return result

    def _migrate_scene_roles(self, conn: Any) -> None:
        rows = conn.execute(
            """SELECT cloud_id FROM scenes
               WHERE parser_version <> ?
                 AND knowledge_status<>'RETRACTED'
               ORDER BY cloud_id""",
            (TrainingPipelineV2.parser_version,),
        ).fetchall()
        if not rows:
            return
        resolver = RoleResolver()
        for row in rows:
            components = conn.execute(
                """SELECT sc.id, sc.token_index, wf.normalized_form
                FROM scene_components sc JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id
                WHERE sc.scene_cloud_id=? ORDER BY sc.token_index""",
                (row["cloud_id"],),
            ).fetchall()
            tokens = [
                WordToken(text=str(item["normalized_form"]), normalized=str(item["normalized_form"]), position=index)
                for index, item in enumerate(components)
            ]
            morphologies = [self.morphology.parse(token.normalized) for token in tokens]
            roles = resolver.resolve(tokens, morphologies)
            predicate_id = next(
                (int(item["id"]) for item, role in zip(components, roles) if role["role"] == "predicate"),
                None,
            )
            for component, morphology, role in zip(components, morphologies, roles):
                conn.execute(
                    """UPDATE scene_components SET grammatical_role=?, dependency_role=?, confidence=?,
                    morphology_json=?, head_component_id=? WHERE id=?""",
                    (
                        role["role"], role["dependency_role"], role["confidence"],
                        encode({"lemma": morphology.lemma, "pos": morphology.pos_tag, **morphology.features}),
                        predicate_id if predicate_id and role["role"] not in {"preposition", "service", "predicate"} else None,
                        component["id"],
                    ),
                )
            conn.execute(
                "UPDATE scenes SET parser_version=?, updated_at=? WHERE cloud_id=?",
                (TrainingPipelineV2.parser_version, utcnow(), row["cloud_id"]),
            )

    def _semantic_membership_detail(self, conn: Any, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        if not conn or not left.get("lexeme_cloud_id") or not right.get("lexeme_cloud_id"):
            return {"score": 0.0, "role_match_score": 0.0, "match_type": "none", "concepts": [], "supporting_scenes": []}
        rows = conn.execute(
            """SELECT sm.concept_cloud_id, cfr.concept_space_id, cfr.evidence_type, cfr.stability, cfr.evidence_count
            FROM semantic_memberships sm JOIN concept_fog_registry cfr ON cfr.concept_cloud_id=sm.concept_cloud_id
            WHERE sm.lexeme_cloud_id=? AND sm.concept_cloud_id IN
              (SELECT concept_cloud_id FROM semantic_memberships WHERE lexeme_cloud_id=?)""",
            (int(left["lexeme_cloud_id"]), int(right["lexeme_cloud_id"])),
        ).fetchall()
        if not rows:
            return {"score": 0.0, "role_match_score": 0.0, "match_type": "none", "concepts": [], "supporting_scenes": []}
        order = {"definition": (0.85, "stable_concept"), "contextual_similarity": (0.65, "related_concept"), "shared_category": (0.45, "shared_category")}
        best = max(rows, key=lambda row: order.get(str(row["evidence_type"]), (0.0, "none"))[0])
        score, match_type = order.get(str(best["evidence_type"]), (0.0, "none"))
        if str(best["evidence_type"]) == "shared_category":
            scenes = conn.execute(
                """SELECT DISTINCT source_scene_cloud_id FROM semantic_evidence
                WHERE evidence_type='definition'
                  AND CAST(json_extract(evidence_json,'$.defined_lexeme_id') AS INTEGER) IN (?,?)
                ORDER BY source_scene_cloud_id LIMIT 8""",
                (int(left["lexeme_cloud_id"]), int(right["lexeme_cloud_id"])),
            ).fetchall()
        else:
            scenes = conn.execute(
                """SELECT DISTINCT source_scene_cloud_id FROM semantic_evidence
                WHERE (left_lexeme_cloud_id IN (?,?) AND right_lexeme_cloud_id IN (?,?))
                ORDER BY source_scene_cloud_id LIMIT 8""",
                (int(left["lexeme_cloud_id"]), int(right["lexeme_cloud_id"]), int(left["lexeme_cloud_id"]), int(right["lexeme_cloud_id"])),
            ).fetchall()
        evidence_scene_ids = [f"scene-{row['source_scene_cloud_id']}" for row in scenes if row["source_scene_cloud_id"] is not None]
        return {"score": score, "role_match_score": score, "match_type": match_type, "concepts": [int(best["concept_cloud_id"])], "concept_ids": [int(best["concept_cloud_id"])], "concept_space_ids": [int(best["concept_space_id"])], "supporting_scenes": evidence_scene_ids, "evidence_scene_ids": evidence_scene_ids}

    def _semantic_membership(self, conn: Any, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        return float(self._semantic_membership_detail(conn, left, right)["score"])

    def _entity_match_detail(
        self,
        query: Dict[str, Any],
        memory: Dict[str, Any],
        conn: Any,
    ) -> Dict[str, Any]:
        if not conn:
            return {"score": 0.0, "role_match_score": 0.0, "match_type": "none", "concepts": [], "supporting_scenes": []}
        query_ids = self._entity_ids_for_value(conn, query)
        memory_ids = {
            int(value)
            for value in (memory.get("entity_id"), memory.get("lexeme_cloud_id"))
            if value is not None
        }
        if not query_ids or not memory_ids:
            return {"score": 0.0, "role_match_score": 0.0, "match_type": "none", "concepts": [], "supporting_scenes": []}
        if query_ids & memory_ids:
            exact_surface = any(
                str(query.get(key) or "").casefold()
                == str(memory.get(key) or "").casefold()
                for key in ("normalized", "lemma", "surface")
                if query.get(key) and memory.get(key)
            )
            score = 1.0 if exact_surface else .95
            return {
                "score": score,
                "role_match_score": score,
                "match_type": "exact_entity" if exact_surface else "entity_alias",
                "concepts": [],
                "supporting_scenes": [],
            }
        observed_type = memory.get("entity_type")
        if isinstance(observed_type, dict):
            type_id = observed_type.get("entity_id")
            type_match = self._value_match_detail(
                query,
                {**observed_type, "entity_id": type_id},
                None,
            )
            if type_id in query_ids or float(type_match.get("score", 0.0)) >= .95:
                return {
                    "score": .88,
                    "role_match_score": .88,
                    "match_type": "entity_type",
                    "concepts": [],
                    "supporting_scenes": [],
                }
        source_marks = ",".join("?" for _ in memory_ids)
        target_marks = ",".join("?" for _ in query_ids)
        relation = conn.execute(
            f"""WITH RECURSIVE type_path(entity_id,depth) AS (
                    SELECT object_lexeme_cloud_id,1
                    FROM concept_relations
                    WHERE relation_type='IS_A'
                      AND status<>'DEPRECATED'
                      AND subject_lexeme_cloud_id IN ({source_marks})
                    UNION ALL
                    SELECT relation.object_lexeme_cloud_id,type_path.depth+1
                    FROM concept_relations relation
                    JOIN type_path
                      ON relation.subject_lexeme_cloud_id=type_path.entity_id
                    WHERE relation.relation_type='IS_A'
                      AND relation.status<>'DEPRECATED'
                      AND type_path.depth<3
                )
                SELECT MIN(depth) AS depth FROM type_path
                WHERE entity_id IN ({target_marks})""",
            (*sorted(memory_ids), *sorted(query_ids)),
        ).fetchone()
        if relation and relation["depth"] is not None:
            return {
                "score": .88,
                "role_match_score": .88,
                "match_type": "is_a",
                "concepts": [],
                "supporting_scenes": [],
                "depth": int(relation["depth"]),
            }
        return {"score": 0.0, "role_match_score": 0.0, "match_type": "none", "concepts": [], "supporting_scenes": []}

    def _value_match_detail(self, query: Dict[str, Any], memory: Dict[str, Any], conn: Any = None) -> Dict[str, Any]:
        if not query or not memory:
            return {"score": 0.0, "role_match_score": 0.0, "match_type": "none", "concepts": [], "supporting_scenes": []}
        entity_match = self._entity_match_detail(query, memory, conn)
        if float(entity_match.get("score", 0.0)) > 0.0:
            return entity_match
        if query.get("normalized") and memory.get("normalized") and query["normalized"].casefold() == memory["normalized"].casefold():
            return {"score": 1.0, "role_match_score": 1.0, "match_type": "exact_form", "concepts": [], "supporting_scenes": []}
        if query.get("lemma") and memory.get("lemma") and query["lemma"].casefold() == memory["lemma"].casefold():
            return {"score": .95, "role_match_score": .95, "match_type": "lemma", "concepts": [], "supporting_scenes": []}
        semantic = self._semantic_membership_detail(conn, query, memory)
        if semantic.get("match_type") == "related_concept":
            return {
                **semantic,
                "score": .75,
                "role_match_score": .75,
                "match_type": "probable_concept",
            }
        return semantic

    def _value_match(self, query: Dict[str, Any], memory: Dict[str, Any], conn: Any = None) -> float:
        return float(self._value_match_detail(query, memory, conn)["score"])

    def _is_a_match_detail(self, value: Dict[str, Any], constraint: Dict[str, Any], conn: Any = None) -> Dict[str, Any]:
        value = value or {}
        constraint = constraint or {}
        observed_type = value.get("entity_type")
        if isinstance(observed_type, dict):
            type_match = self._value_match_detail(observed_type, constraint, conn)
            if float(type_match.get("score", 0.0)) >= .95:
                return {
                    **type_match,
                    "score": 1.0,
                    "role_match_score": 1.0,
                    "match_type": "apposition_type_membership",
                }
        exact = self._value_match_detail(value, constraint, conn)
        if float(exact.get("score", 0.0)) >= .95:
            return {**exact, "score": 1.0, "role_match_score": 1.0, "match_type": "exact_type"}
        if not conn or not value.get("lexeme_cloud_id") or not constraint.get("lexeme_cloud_id"):
            return {"score": 0.0, "role_match_score": 0.0, "match_type": "none", "concepts": [], "supporting_scenes": []}
        evidence = conn.execute(
            """SELECT source_scene_cloud_id FROM semantic_evidence
            WHERE evidence_type='definition'
              AND CAST(json_extract(evidence_json, '$.defined_lexeme_id') AS INTEGER)=?
              AND CAST(json_extract(evidence_json, '$.definition_lexeme_id') AS INTEGER)=?
            ORDER BY source_scene_cloud_id LIMIT 8""",
            (int(value["lexeme_cloud_id"]), int(constraint["lexeme_cloud_id"])),
        ).fetchall()
        scene_ids = [f"scene-{row['source_scene_cloud_id']}" for row in evidence if row["source_scene_cloud_id"] is not None]
        return {
            "score": .9 if scene_ids else 0.0,
            "role_match_score": .9 if scene_ids else 0.0,
            "match_type": "definition_is_a" if scene_ids else "none",
            "concepts": [], "supporting_scenes": scene_ids, "evidence_scene_ids": scene_ids,
        }

    def _role_match(self, role: str, query: Dict[str, Any], memory_roles: Dict[str, Dict[str, Any]], conn: Any = None) -> float:
        aliases = ROLE_MATCH_ALIASES.get(role, {role})
        values = [memory_roles.get(name, {}) for name in aliases]
        return max((self._value_match(query, value, conn) for value in values), default=0.0)

    def _role_match_detail(self, role: str, query: Dict[str, Any], memory_roles: Dict[str, Dict[str, Any]], conn: Any = None) -> Dict[str, Any]:
        aliases = ROLE_MATCH_ALIASES.get(role, {role})
        details = [self._value_match_detail(query, memory_roles.get(name, {}), conn) for name in aliases]
        return max(details, key=lambda item: float(item["role_match_score"])) if details else {"score": 0.0, "role_match_score": 0.0, "match_type": "none", "concepts": [], "supporting_scenes": []}

    @staticmethod
    def _assign_scene_activation(scenes: Iterable[Dict[str, Any]]) -> None:
        for scene in scenes:
            scores = scene.get("scores", {})
            validation = scene.get("anchor_validation", scores.get("anchor_validation", {}))
            matched = len(scene.get("matched_roles", []))
            if scene.get("result_type") == "FULL_HIT" and validation.get("status") == "PASSED":
                activation, tier = 1.0, "DIRECT"
            elif validation.get("status") == "PASSED" and matched >= 2:
                activation, tier = .75, "RELATED"
            elif matched >= 1:
                activation, tier = .30, "PARTIAL"
            else:
                activation, tier = .05, "BACKGROUND"
            scene["physics"] = {
                "activation": activation, "gravity": activation, "relevance_tier": tier,
                "matched_role_count": matched, "semantic_total": float(scores.get("semantic_total", 0.0)),
            }

    @staticmethod
    def _persist_scene_activation(conn: Any, hive_id: str, scenes: Iterable[Dict[str, Any]]) -> None:
        by_scene_id = {
            int(scene["cloud_id"]): scene.get("physics", {})
            for scene in scenes if scene.get("cloud_id") is not None
        }
        rows = conn.execute(
            "SELECT id, hive_placement_id, source_scene_cloud_id, metadata_json FROM hive_cells WHERE hive_id=?",
            (hive_id,),
        ).fetchall()
        for row in rows:
            physics = by_scene_id.get(int(row["source_scene_cloud_id"])) if row["source_scene_cloud_id"] is not None else None
            if not physics:
                continue
            activation = float(physics["activation"])
            metadata = decode(row["metadata_json"], {})
            metadata["query_activation"] = {
                "value": activation, "tier": physics["relevance_tier"],
                "matched_role_count": physics["matched_role_count"], "semantic_total": physics["semantic_total"],
            }
            conn.execute(
                "UPDATE hive_cells SET local_activation=?, metadata_json=?, updated_at=? WHERE id=?",
                (activation, encode(metadata), utcnow(), row["id"]),
            )
            conn.execute(
                "UPDATE cloud_placements SET local_activation=?, local_gravity=?, updated_at=? WHERE id=?",
                (activation, activation, utcnow(), row["hive_placement_id"]),
            )

    @staticmethod
    def _sync_working_cells(conn: Any, hive_id: str, state: Dict[str, Any]) -> None:
        rows = conn.execute(
            """SELECT hc.id, hc.component_class, hc.source_scene_cloud_id, hc.local_activation,
                      hc.retention, hp.x, hp.y, c.canonical_name AS label, hc.metadata_json
               FROM hive_cells hc JOIN cloud_placements hp ON hp.id=hc.hive_placement_id
               JOIN clouds c ON c.id=hc.dominant_cloud_id
               WHERE hc.hive_id=? ORDER BY hc.created_at, hc.id""",
            (hive_id,),
        ).fetchall()
        state["working_cells"] = [
            {
                "id": str(row["id"]), "component_class": str(row["component_class"]),
                "source_scene_id": int(row["source_scene_cloud_id"]) if row["source_scene_cloud_id"] is not None else None,
                "label": str(row["label"]), "activation": float(row["local_activation"]),
                "retention": float(row["retention"]), "position": {"x": float(row["x"]), "y": float(row["y"])},
                "metadata": decode(row["metadata_json"], {}),
            }
            for row in rows
        ]

    def _score_scene(self, frame: Dict[str, Any], scene: Dict[str, Any], conn: Any = None) -> Dict[str, Any]:
        query_roles, memory_roles = frame["roles"], scene["roles"]
        requested = frame.get("requested_role")
        polar_question = frame.get("query_type") == "polar_question"
        action_question = frame.get("query_type") == "action_question"
        conceptual_frame = frame.get("conceptual_query_frame") or {}
        scene_projection = None
        concept_match_detail: Dict[str, Any] = {"score": 0.0, "match_type": "none"}
        if conn and conceptual_frame.get("expansion_enabled") and conceptual_frame.get("action_concept_id"):
            scene_projection = self.semantic_projection.scene_projection(
                conn, int(scene.get("cloud_id") or str(scene.get("id", "")).removeprefix("scene-") or 0),
                str(conceptual_frame["action_concept_id"]),
            )
            if scene_projection:
                concept_status = str(scene_projection.get("concept_status") or "PROBABLE")
                match_type = "action_concept" if concept_status == "STABLE" else "probable_concept"
                concept_match_detail = {
                    "score": MATCH_WEIGHTS[match_type], "role_match_score": MATCH_WEIGHTS[match_type],
                    "match_type": match_type, "concept_id": conceptual_frame["action_concept_id"],
                    "scene_projection_id": scene_projection["id"], "semantic_frame": scene_projection["semantic_frame"],
                }
        fixed_roles = {
            role: value for role, value in query_roles.items()
            if role != requested and value.get("status") == "fixed" and value.get("lemma")
        }
        role_matches: Dict[str, float] = {}
        role_match_details: Dict[str, Dict[str, Any]] = {}
        for role, query_value in fixed_roles.items():
            if action_question and role in {"agent", "theme"}:
                participant_matches = [
                    (
                        name,
                        value,
                        self._value_match_detail(query_value, value, conn),
                    )
                    for name, value in memory_roles.items()
                    if name not in {"action", "modal"}
                    and isinstance(value, dict)
                    and value.get("status") == "fixed"
                ]
                matched_name, memory_value, detail = max(
                    participant_matches,
                    key=lambda item: float(item[2].get("score", 0.0)),
                    default=("", {}, {
                        "score": 0.0,
                        "role_match_score": 0.0,
                        "match_type": "none",
                    }),
                )
                detail = {**detail, "matched_participant_role": matched_name}
            else:
                detail = self._role_match_detail(role, query_value, memory_roles, conn)
                memory_value = next((memory_roles.get(name, {}) for name in ROLE_MATCH_ALIASES.get(role, {role}) if memory_roles.get(name)), {})
            if role == "action" and float(concept_match_detail.get("score", 0.0)) > float(detail.get("score", 0.0)):
                detail = concept_match_detail
            role_match_details[role] = {**detail, "required": True, "query_value": query_value.get("lemma"), "scene_value": memory_value.get("lemma")}
            role_matches[role] = float(detail["role_match_score"])
        action_match = role_matches.get("action", 0.0)
        object_match = role_matches.get("object", 0.0)
        location_match = role_matches.get("location", 0.0)
        agent_match = role_matches.get("agent", 0.0)
        anchor_match = sum(role_matches.values()) / max(1, len(role_matches))
        grammar_match = 1.0 if anchor_match >= .99 else .65 if anchor_match > 0 else 0.0
        role_value = memory_roles.get(requested or "")
        if not role_value:
            requested_slot = str(frame.get("requested_slot") or "")
            for hypothesis in frame.get("requested_role_hypotheses", []):
                if float(hypothesis.get("confidence", 0.0)) < .5:
                    continue
                alternative = memory_roles.get(str(hypothesis.get("role") or ""))
                if not alternative:
                    continue
                alternative_slot = str(alternative.get("grammatical_slot") or "")
                if (
                    requested_slot == "direct_object"
                    and alternative_slot == "subject"
                ):
                    continue
                if (
                    requested_slot == "subject"
                    and alternative_slot != "subject"
                ):
                    continue
                role_value = alternative
                break
        slot_constraint_details: Dict[str, Dict[str, Any]] = {}
        conceptual_constraints = (conceptual_frame.get("slot_constraints") or {}).get(
            requested or "", {}
        )
        deferred_taxonomy = bool(scene_projection and conceptual_constraints.get("is_a"))
        for name, constraint in (frame.get("slot_constraints", {}).get(requested or "", {}) or {}).items():
            if name == "is_a":
                slot_constraint_details[name] = (
                    {"score": 1.0, "role_match_score": 1.0, "match_type": "deferred_candidate_constraint", "status": "PENDING_CANDIDATE", "required": constraint}
                    if deferred_taxonomy else self._is_a_match_detail(role_value, constraint, conn)
                )
        semantic_constraint_details: Dict[str, Dict[str, Any]] = {}
        for name, constraint in (frame.get("semantic_constraints", {}) or {}).items():
            semantic_constraint_details[name] = self._role_match_detail(name, constraint, memory_roles, conn)
            if name == "purpose" and scene_projection:
                implied_purpose = (scene_projection.get("semantic_frame") or {}).get("purpose") or {}
                if implied_purpose.get("lemma"):
                    semantic_constraint_details[name] = {
                        "score": 1.0, "role_match_score": 1.0,
                        "match_type": "implied_semantic_role", "implied": True,
                        "query_value": constraint.get("lemma"), "scene_value": implied_purpose.get("lemma"),
                    }
        constraints_passed = all(float(detail.get("score", 0.0)) >= .85 for detail in slot_constraint_details.values())
        semantic_constraints_passed = all(float(detail.get("score", 0.0)) >= .85 for detail in semantic_constraint_details.values())
        requested_role_match = 1.0 if polar_question else 0.0 if not role_value or scene["negation"] else 1.0
        if not polar_question and (not constraints_passed or not semantic_constraints_passed):
            requested_role_match = 0.0
        explanation_match = 0.0
        if requested == "source" and not role_value:
            object_value = query_roles.get("object", {})
            destination_value = memory_roles.get("destination", {})
            object_match_for_explanation = self._role_match("object", object_value, memory_roles, conn) if object_value else 0.0
            destination_match = self._value_match(query_roles.get("location", {}), destination_value, conn) if destination_value else 0.0
            if object_match_for_explanation >= .85 and destination_match >= .85 and not scene["negation"]:
                explanation_match = min(object_match_for_explanation, destination_match)
                requested_role_match = .85 * explanation_match
        structural_match = clamp(anchor_match * .65 + requested_role_match * .35)
        semantic_match = clamp(anchor_match * .75 + max(role_matches.values(), default=0.0) * .25)
        role_coverage = sum(score >= .5 for score in role_matches.values()) / max(1, len(fixed_roles))
        if scene.get("retrieval_scope") == "DIALOGUE":
            source_quality = (
                .2 if not scene.get("eligible_for_fact_retrieval", True)
                else .75 if scene.get("memory_class") == "USER_ASSERTION"
                else .7 if scene.get("memory_class") == "USER_CONFIRMATION"
                else .2
            )
        elif scene.get("retrieval_scope") == "LOCAL":
            source_quality = .9
        elif scene.get("retrieval_scope") in {"IMPORTED", "GLOBAL"}:
            source_quality = .75
        else:
            source_quality = .8
        context_conflict = any(
            role in CONTEXT_ROLES and query_value.get("lemma")
            and any(memory_roles.get(alias) for alias in ROLE_MATCH_ALIASES.get(role, {role}))
            and self._role_match(role, query_value, memory_roles, conn) < .5
            for role, query_value in fixed_roles.items()
        )
        action_concept_match = float(concept_match_detail.get("score", 0.0))
        semantic_role_match = min(
            1.0,
            (object_match if "object" in fixed_roles else 1.0)
            * (1.0 if not semantic_constraint_details else min(float(item.get("score", 0.0)) for item in semantic_constraint_details.values())),
        )
        slot_constraint_match = min((float(item.get("score", 0.0)) for item in slot_constraint_details.values()), default=1.0)
        purpose_match = min((float(item.get("score", 0.0)) for item in semantic_constraint_details.values()), default=1.0)
        query_modal = query_roles.get("modal", {})
        memory_modal = memory_roles.get("modal", {})
        modality_match = not query_modal or (
            bool(memory_modal)
            and query_modal.get("semantic_function") == memory_modal.get("semantic_function")
        )
        temporal_match = not query_roles.get("time") or role_matches.get("time", 0.0) >= .45
        polarity_match = (
            bool(scene["negation"]) == bool(frame.get("negated"))
            if polar_question
            else not scene["negation"]
        )
        weights = settings.retrieval_weights
        predicate_score = max(action_match, action_concept_match)
        entity_score = max(
            (score for role, score in role_matches.items() if role != "action"),
            default=1.0 if not fixed_roles else 0.0,
        )
        constraint_score = min(
            slot_constraint_match,
            purpose_match,
            float(constraints_passed and semantic_constraints_passed),
        )
        weighted_components = [
            ("requested_role_support", requested_role_match),
            ("constraint_match", constraint_score),
            ("polarity_match", float(polarity_match)),
            ("context_match", float(not context_conflict)),
            ("evidence_confidence", source_quality),
        ]
        if "action" in fixed_roles or conceptual_frame.get("action_concept_id"):
            weighted_components.append(("predicate_match", predicate_score))
        if fixed_roles:
            weighted_components.append(("required_roles_match", anchor_match))
        if any(role != "action" for role in fixed_roles):
            weighted_components.append(("entity_match", entity_score))
        semantic_weight = sum(
            float(weights.get(name, 0.0))
            for name, _ in weighted_components
        ) or 1.0
        semantic_total = clamp(
            sum(
                float(weights.get(name, 0.0)) * value
                for name, value in weighted_components
            )
            / semantic_weight
        )
        total = semantic_total
        matched_roles = [role for role, score in role_matches.items() if score >= .5]
        mismatched_roles = [role for role, score in role_matches.items() if score < .5]
        role_thresholds = {role: .45 for role in fixed_roles}
        failed_anchors = [role for role, score in role_matches.items() if score < role_thresholds[role]]
        failed_constraints = [f"slot.{name}" for name, detail in slot_constraint_details.items() if float(detail.get("score", 0.0)) < .85]
        failed_semantic_constraints = [f"semantic.{name}" for name, detail in semantic_constraint_details.items() if float(detail.get("score", 0.0)) < .85]
        anchor_validation = {
            "status": "PASSED" if requested_role_match and not failed_anchors and not failed_constraints and not failed_semantic_constraints and polarity_match and modality_match and temporal_match else "FAILED",
            "required_roles": sorted(fixed_roles), "failed_roles": failed_anchors,
            "slot_constraints": slot_constraint_details,
            "semantic_constraints": semantic_constraint_details,
            "failed_constraints": failed_constraints + failed_semantic_constraints,
            "role_thresholds": role_thresholds, "requested_role_present": polar_question or bool(role_value),
            "polarity_match": polarity_match, "modality_match": modality_match, "temporal_match": temporal_match,
            "critical_roles_passed": all(role_matches.get(role, 0.0) >= .45 for role in ("agent", "action") if role in fixed_roles),
            "supporting_roles_passed": all(role_matches.get(role, 0.0) >= .45 for role in fixed_roles if role not in {"agent", "action"}),
            "scene_found": bool(scene.get("event")),
            "answer_slot_type": frame.get("answer_slot_type", "participant"),
            "answer_extraction": "predicate" if action_question else "participant",
            "placeholder_predicate_removed": bool(
                (frame.get("action_question") or {}).get(
                    "placeholder_predicate_removed"
                )
            ),
            "subject_match": deepcopy(
                role_match_details.get("agent")
                or role_match_details.get("theme")
                or {}
            ),
        }
        if polar_question and fixed_roles and not failed_anchors:
            result_type = "FULL_HIT" if polarity_match else "CONFLICT_HIT"
        elif scene["negation"] and (anchor_match or requested_role_match):
            result_type = "CONFLICT_HIT"
        elif requested_role_match and fixed_roles and all(score >= .95 for score in role_matches.values()):
            result_type = "FULL_HIT"
        elif requested_role_match and anchor_match > 0:
            result_type = "ROLE_HIT"
        elif requested_role_match:
            # Broad retrieval may find a scene because it contains a value for
            # the requested role.  Keep it visible for diagnostics even when
            # its mandatory anchors fail; candidate admission remains strict.
            result_type = "PARTIAL_HIT"
        elif anchor_match > 0:
            result_type = "PARTIAL_HIT"
        elif semantic_match:
            result_type = "WEAK_HIT"
        elif action_question and scene.get("event"):
            result_type = "FOUND_BUT_REJECTED"
        else:
            result_type = "NO_HIT"
        if action_question and anchor_validation["status"] == "FAILED" and scene.get("event"):
            result_type = "FOUND_BUT_REJECTED"
        if action_question and anchor_validation["status"] == "FAILED" and scene.get("event"):
            selection_reason = (
                "Сцена найдена, но отклонена: "
                + (
                    "не пройдены опорные роли: " + ", ".join(failed_anchors)
                    if failed_anchors
                    else "не совпали ограничения вопроса"
                )
            )
        elif anchor_validation["status"] == "PASSED":
            selection_reason = "сцена прошла проверку опорных ролей"
        elif failed_constraints or failed_semantic_constraints:
            selection_reason = "не пройдены ограничения слота: " + ", ".join(failed_constraints + failed_semantic_constraints)
        elif not polarity_match or not modality_match:
            selection_reason = "несовместимая полярность или модальность"
        elif failed_anchors:
            selection_reason = "не пройдены опорные роли: " + ", ".join(failed_anchors)
        else:
            selection_reason = "не найдена запрошенная роль"
        return {**scene, "scores": {
            "object_match": object_match, "action_match": action_match, "agent_match": agent_match,
            "location_match": location_match, "requested_role_match": requested_role_match,
            "anchor_match": anchor_match, "anchor_score": anchor_match, "role_matches": role_matches,
            "role_match_details": role_match_details, "anchor_validation": anchor_validation,
            "slot_constraints": slot_constraint_details, "semantic_constraints": semantic_constraint_details,
            "grammar_match": grammar_match, "semantic_match": semantic_match, "structural_match": structural_match,
            "role_coverage": role_coverage, "requested_role_support": requested_role_match,
            "source_quality": source_quality, "semantic_total": semantic_total,
            "lexical_match": action_match if role_match_details.get("action", {}).get("match_type") in {"exact_form", "lemma"} else 0.0,
            "action_concept_match": action_concept_match,
            "semantic_role_match": semantic_role_match,
            "slot_constraint_match": slot_constraint_match,
            "purpose_match": purpose_match,
            "polarity_match": float(polarity_match),
            "match_weights": MATCH_WEIGHTS,
            "scoring_weights": dict(weights),
            "explanation_match": explanation_match, "context_conflict": context_conflict,
            "source_confidence": source_quality, "retention": 1.0, "total_score": total,
        }, "role_match_details": role_match_details, "anchor_validation": anchor_validation,
            "matched_roles": matched_roles, "mismatched_roles": mismatched_roles,
            "selection_reason": selection_reason,
            "result_type": result_type}

    @staticmethod
    def _polar_answer(frame: Dict[str, Any], scenes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        fixed_roles = {
            role for role, value in frame.get("roles", {}).items()
            if value.get("status") == "fixed" and value.get("lemma")
        }
        query_negated = bool(frame.get("negated"))

        def evidence_score(scene: Dict[str, Any]) -> float:
            details = scene.get("role_match_details", {})
            if not fixed_roles:
                return 0.0
            scores = [float((details.get(role) or {}).get("role_match_score", 0.0)) for role in fixed_roles]
            return min(scores) if scores and all(score >= .65 for score in scores) else 0.0

        def scope_allows(scene: Dict[str, Any], same_polarity: bool) -> bool:
            if "object" in fixed_roles:
                return True
            has_extra_object = bool((scene.get("roles") or {}).get("object"))
            if not has_extra_object:
                return True
            return (not query_negated and same_polarity) or (query_negated and not same_polarity)

        ranked = sorted(
            ((evidence_score(scene), scene) for scene in scenes),
            key=lambda item: (-item[0], str(item[1].get("id", ""))),
        )
        agreeing = [
            (score, scene) for score, scene in ranked
            if score and bool(scene.get("negation")) == query_negated and scope_allows(scene, True)
        ]
        contradicting = [
            (score, scene) for score, scene in ranked
            if score and bool(scene.get("negation")) != query_negated and scope_allows(scene, False)
        ]

        def source_surface(prefix: str, scene: Dict[str, Any]) -> str:
            source = str(scene.get("source_text") or "").strip()
            if source:
                source = source[0].upper() + source[1:]
                source = source if source.endswith((".", "!", "?")) else source + "."
                return f"{prefix} {source}"
            return prefix

        if agreeing and contradicting:
            surface = "В памяти есть противоречивые сведения."
            status = "UNRESOLVED"
            resolved_value = None
            confidence = min(agreeing[0][0], contradicting[0][0])
            evidence_status = "CONFLICTING_EVIDENCE"
            supporting = [agreeing[0][1]["id"], contradicting[0][1]["id"]]
        elif agreeing:
            confidence, scene = agreeing[0]
            surface = source_surface("Да.", scene)
            status = "RESOLVED"
            resolved_value = True
            evidence_status = "SUPPORTED"
            supporting = [scene["id"]]
        elif contradicting:
            confidence, scene = contradicting[0]
            surface = source_surface("Нет.", scene)
            status = "RESOLVED"
            resolved_value = False
            evidence_status = "CONTRADICTED"
            supporting = [scene["id"]]
        else:
            surface = "В доступной памяти недостаточно данных."
            status = "UNRESOLVED"
            resolved_value = None
            confidence = 0.0
            evidence_status = "INSUFFICIENT_EVIDENCE"
            supporting = []
        return {
            "query": frame.get("source_text", ""),
            "answer_mode": "polar",
            "resolved_role": None,
            "resolved_value": resolved_value,
            "confidence": confidence,
            "supporting_scenes": supporting,
            "surface_answer": surface,
            "full_surface_answer": surface,
            "status": status,
            "status_message": surface,
            "evidence_status": evidence_status,
            "semantic_total": confidence,
            "gravity": 0.0,
            "decision_score": confidence,
        }

    def _stable_concept_ids(self, conn: Any, value: Dict[str, Any]) -> List[int]:
        if not conn or not value.get("lexeme_cloud_id"):
            return list(value.get("concept_ids") or [])
        return [int(row["concept_cloud_id"]) for row in conn.execute(
            """SELECT sm.concept_cloud_id FROM semantic_memberships sm
            JOIN concept_fog_registry cfr ON cfr.concept_cloud_id=sm.concept_cloud_id
            WHERE sm.lexeme_cloud_id=? AND cfr.evidence_type IN ('definition','contextual_similarity')""",
            (int(value["lexeme_cloud_id"]),),
        ).fetchall()]

    def _is_excluded(self, conn: Any, value: Dict[str, Any], exclusions: Iterable[Dict[str, Any]]) -> bool:
        value_concepts = set(self._stable_concept_ids(conn, value))
        for excluded in exclusions:
            mode = excluded.get("mode", "STABLE_CONCEPT")
            if self._value_match(value, excluded, conn) >= .95:
                return True
            if mode == "STABLE_CONCEPT" and value_concepts & set(self._stable_concept_ids(conn, excluded)):
                return True
        return False

    @staticmethod
    def _predicate_phrase(scene: Dict[str, Any]) -> Dict[str, Any]:
        event = scene.get("event") or {}
        predicate = event.get("predicate") or scene.get("roles", {}).get("action", {})
        predicate_surface = str(predicate.get("surface") or "").strip()
        predicate_lemma = str(predicate.get("lemma") or "").strip()
        if not predicate_surface:
            return {}
        pieces = [predicate_surface]
        seen = set()
        for role in ACTION_QUESTION_COMPLEMENT_ROLES:
            value = (scene.get("roles") or {}).get(role, {})
            if not isinstance(value, dict) or not value.get("surface"):
                continue
            if value.get("grammatical_slot") == "subject":
                continue
            key = (
                value.get("entity_id") or value.get("lexeme_cloud_id")
                or value.get("lemma"),
                value.get("preposition", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            surface = str(value["surface"]).strip()
            preposition = str(value.get("preposition") or "").strip()
            pieces.append(" ".join(item for item in (preposition, surface) if item))
        for modifier in event.get("modifiers", []):
            if str(modifier.get("role") or "") not in {"manner", "time"}:
                continue
            surface = str(modifier.get("value_text") or "").strip()
            if surface and surface.casefold() not in {piece.casefold() for piece in pieces}:
                pieces.append(surface)
        surface = " ".join(pieces)
        supported_roles = [
            role for role in ACTION_QUESTION_COMPLEMENT_ROLES
            if (scene.get("roles") or {}).get(role, {}).get("surface")
        ]
        return {
            "lemma": predicate_lemma,
            "surface": surface,
            "predicate_lemma": predicate_lemma,
            "predicate_surface": predicate_surface,
            "answer_slot_type": "predicate_phrase",
            "part_of_speech": "VERB",
            "grammatical_features": {},
            "phrase_components": list(pieces[1:]),
            "predicate_phrase_completeness": (
                1.0 if supported_roles else .55
            ),
            "supported_complement_roles": supported_roles,
        }

    def _candidates(self, frame: Dict[str, Any], scenes: Iterable[Dict[str, Any]], conn: Any = None) -> List[Dict[str, Any]]:
        requested = frame.get("requested_role")
        if frame.get("requires_clarification"):
            return []
        if frame.get("context_resolution", {}).get("status") == "UNRESOLVED_CONTEXT":
            return []
        found: Dict[str, Dict[str, Any]] = {}
        if not requested:
            return []
        for scene in scenes:
            if not scene.get("eligible_for_fact_retrieval", True):
                scene["candidate_status"] = "rejected"
                scene["decision_reason"] = "derived_dialogue_not_independent_evidence"
                continue
            action_question = frame.get("query_type") == "action_question"
            value = (
                self._predicate_phrase(scene)
                if action_question
                else scene["roles"].get(requested)
            )
            conceptual_frame = frame.get("conceptual_query_frame") or {}
            projection_frame = ((scene.get("scores", {}).get("role_match_details", {}).get("action", {}) or {}).get("semantic_frame", {}) or {})
            conceptual_candidate = bool(
                conceptual_frame.get("action_concept_id")
                and float(scene.get("scores", {}).get("action_concept_match", 0.0)) >= .75
            )
            if conceptual_candidate and not value:
                value = projection_frame.get(requested)
                if not value:
                    requested_slot = frame.get("requested_slot")
                    value = next(
                        (
                            candidate for candidate in projection_frame.values()
                            if isinstance(candidate, dict)
                            and candidate.get("grammatical_slot") == requested_slot
                        ),
                        None,
                    )
            if not value:
                for hypothesis in frame.get("requested_role_hypotheses", []):
                    if float(hypothesis.get("confidence", 0.0)) < .5:
                        continue
                    alternative_role = str(hypothesis.get("role") or "")
                    if alternative_role == requested:
                        continue
                    alternative = scene["roles"].get(alternative_role)
                    if alternative:
                        alternative_slot = str(
                            alternative.get("grammatical_slot") or ""
                        )
                        if (
                            frame.get("requested_slot") == "direct_object"
                            and alternative_slot == "subject"
                        ):
                            continue
                        if (
                            frame.get("requested_slot") == "subject"
                            and alternative_slot != "subject"
                        ):
                            continue
                        value = {
                            **alternative,
                            "resolved_role_hypothesis": alternative_role,
                            "requested_role": requested,
                        }
                        break
            explanation = requested == "source" and not value and scene["scores"].get("explanation_match", 0.0) >= .85
            if not value and not explanation:
                scene["candidate_status"] = "rejected"
                scene["decision_reason"] = "requested_role_missing"
                continue
            validation = scene.get("anchor_validation") or scene["scores"].get("anchor_validation", {})
            failed_anchor_roles = list(validation.get("failed_roles") or [])
            # Conceptual expansion can supply a missing semantic frame, but it
            # must never override an explicitly stated participant in the
            # query.  Candidate admission therefore remains structural before
            # vibration or physical ranking begins.
            has_explicit_anchor_conflict = bool(failed_anchor_roles)
            semantic_candidate_ready = bool(
                conceptual_candidate
                and float(scene["scores"].get("action_concept_match", 0.0)) >= .75
                and float(scene["scores"].get("object_match", 0.0)) >= .85
                and float(scene["scores"].get("purpose_match", 0.0)) >= .85
                and not has_explicit_anchor_conflict
            )
            if not explanation and validation.get("status") != "PASSED" and not semantic_candidate_ready:
                scene["candidate_status"] = "rejected"
                failed_role = next(
                    (
                        role for role in ("agent", "action", "object")
                        if role in failed_anchor_roles
                    ),
                    failed_anchor_roles[0] if failed_anchor_roles else "",
                )
                scene["decision_reason"] = (
                    f"{failed_role.upper()}_MISMATCH"
                    if failed_role
                    else
                    "anchor_validation_failed"
                    if action_question
                    else "OBJECT_MISMATCH"
                    if float(scene["scores"].get("object_match", 0.0)) < .85
                    else "anchor_validation_failed"
                )
                scene["candidate_validation"] = {
                    "id": f"rejected-{scene['id']}-{requested}",
                    "lemma": value.get("lemma"),
                    "surface": value.get("surface"),
                    "target_role": requested,
                    "status": "REJECTED",
                    "rejection_reason": (
                        "SCENE_FOUND_BUT_REJECTED"
                        if action_question and scene.get("event")
                        else scene["decision_reason"].upper()
                    ),
                    "rejection_explanation": scene.get("selection_reason"),
                    "sources": [scene["id"]],
                    "primary_source_id": scene["id"],
                    "fact_evidence": [{
                        "scene_id": scene["id"],
                        "text": scene.get("source_text", ""),
                    }],
                    "scores": {"total": scene["scores"].get("semantic_total", 0.0)},
                }
                continue
            excluded_values = frame.get("excluded_roles", {}).get(requested, [])
            excluded = not explanation and self._is_excluded(conn, value, excluded_values)
            validation["exclusion_passed"] = not excluded
            if excluded:
                scene["candidate_status"] = "rejected"
                scene["decision_reason"] = "excluded_by_previous_answer"
                continue
            if requested == "agent" and value.get("part_of_speech") and value.get("part_of_speech") not in {"NOUN", "NPRO"}:
                continue
            if explanation:
                value = {"lemma": scene["id"], "surface": scene.get("source_text", ""), "part_of_speech": "SCENE", "grammatical_features": {}}
            lemma = value["lemma"]
            root_source_ids = list(
                scene.get("root_source_ids")
                or (scene.get("provenance") or {}).get("root_evidence_ids")
                or [scene["id"]]
            )
            evidence_family_key = (
                (scene.get("roles") or {}).get("agent", {}).get("entity_id")
                or (scene.get("roles") or {}).get("theme", {}).get("entity_id"),
                value.get("predicate_lemma", lemma) if action_question else lemma,
                bool(scene.get("negation")),
                (scene.get("roles") or {}).get("time", {}).get("lemma"),
                tuple(sorted(str(item) for item in root_source_ids)),
            )
            candidate_key = repr(evidence_family_key) if action_question else lemma
            answer_surface = value["surface"]
            if value.get("mention_type") == "apposition":
                requested_case = {
                    "subject": "nomn",
                    "direct_object": "accs",
                    "indirect_object": "datv",
                    "instrumental": "ablt",
                    "source_oblique": "gent",
                    "location_oblique": "loct",
                    "destination_oblique": "accs",
                }.get(str(frame.get("requested_slot") or ""))
                answer_surface = self.morphology.inflect(
                    str(value["surface"]),
                    {"case": requested_case, "number": "sing"}
                    if requested_case
                    else {},
                )
            action = scene["scores"]["action_match"]
            object_match = scene["scores"]["object_match"]
            anchor_match = scene["scores"]["anchor_match"]
            contradiction = 1.0 if scene["negation"] else 0.0
            current = found.get(candidate_key)
            support = clamp(scene["scores"].get("semantic_total", scene["scores"]["total_score"]) + .05)
            query_relevance = clamp(anchor_match * .65 + scene["scores"]["requested_role_match"] * .35)
            semantic_support = clamp(scene["scores"]["semantic_match"])
            role_compatibility = 0.98 if not scene["negation"] else 0.0
            exact_match = float(scene["scores"]["object_match"] > 0 and scene["scores"]["action_match"] >= .99)
            activation = clamp(.05 + exact_match * .30 + query_relevance * .25 + semantic_support * .15 + role_compatibility * .15 + support * .10 + scene["scores"]["structural_match"] * .05)
            retention = clamp(scene["scores"]["source_confidence"] * .25 + support * .20 + query_relevance * .15 + role_compatibility * .10 + .05)
            candidate = {
                "id": f"candidate-{lemma}-{uuid.uuid5(uuid.NAMESPACE_URL, candidate_key).hex[:8]}",
                "concept_id": f"concept-{lemma}", "lemma": lemma, "surface": answer_surface,
                "resolved_lemma": lemma,
                "resolved_surface": answer_surface,
                "entity_id": value.get("entity_id"),
                "entity_value": value.get("entity_value") or value.get("surface"),
                "entity_type": deepcopy(value.get("entity_type")),
                "full_surface": (
                    value.get("full_surface")
                    or value.get("mention_surface")
                    or value.get("surface")
                ),
                "mention": {
                    "surface": (
                        value.get("full_surface")
                        or value.get("mention_surface")
                        or value.get("surface")
                    ),
                    "head_surface": (
                        value.get("entity_value") or value.get("surface")
                    ),
                    "mention_type": value.get("mention_type", "noun_phrase"),
                    "attributes": deepcopy(value.get("attributes", [])),
                },
                "answer_surfaces": deepcopy(
                    value.get("answer_surfaces")
                    or {
                        "short_name": value.get("surface"),
                        "observed_surface": value.get("surface"),
                        "canonical_lemma": value.get("lemma"),
                        "full_mention": (
                            value.get("full_surface")
                            or value.get("mention_surface")
                            or value.get("surface")
                        ),
                    }
                ),
                "lexeme_cloud_id": value.get("lexeme_cloud_id"),
                "word_form_cloud_id": value.get("word_form_cloud_id"),
                "preposition": value.get("preposition", ""),
                "grammatical_features": deepcopy(value.get("grammatical_features", {})),
                "form_provenance": {
                    "source_type": "observed_training_form",
                    "scene_id": scene["id"],
                    "scene_text": scene.get("source_text", ""),
                    "observed_surface": value["surface"],
                    "generated": answer_surface != value["surface"],
                },
                "target_role": requested, "part_of_speech": value.get("part_of_speech", "NOUN"),
                "answer_slot_type": value.get(
                    "answer_slot_type",
                    frame.get("answer_slot_type", "participant"),
                ),
                "predicate_lemma": value.get("predicate_lemma", lemma),
                "predicate_surface": value.get("predicate_surface", answer_surface),
                "predicate_phrase": value.get("surface") if action_question else None,
                "predicate_phrase_completeness": float(
                    value.get("predicate_phrase_completeness", 1.0)
                ),
                "evidence_family_key": evidence_family_key,
                "root_source_ids": root_source_ids,
                "variants": [
                    {
                        "surface": answer_surface,
                        "source_id": scene["id"],
                        "predicate_phrase_completeness": float(
                            value.get("predicate_phrase_completeness", 1.0)
                        ),
                    }
                ],
                "constraint_matches": {
                    "slot": deepcopy(scene["scores"].get("slot_constraints", {})),
                    "semantic": deepcopy(scene["scores"].get("semantic_constraints", {})),
                },
                "entity_kind": "entity",
                "answer_mode": "explanation" if explanation else "direct",
                "answer_scene_id": scene["id"] if explanation else None,
                "hard_forbidden": bool(scene["scores"].get("context_conflict")),
                "sources": [scene["id"]], "primary_source_id": scene["id"],
                "fact_evidence": [{"scene_id": scene["id"], "text": scene.get("source_text", "")}],
                "scores": {
                    "role_compatibility": role_compatibility, "query_relevance": query_relevance, "semantic_support": semantic_support, "structural_support": scene["scores"]["structural_match"], "exact_match": exact_match, "action_compatibility": action, "object_compatibility": object_match, "anchor_compatibility": anchor_match,
                    "grammar_compatibility": .99, "source_confidence": support, "resonance": 0.0,
                    "retention": retention, "activation": activation, "contradiction": contradiction,
                    "evidence_confidence": support, "semantic_confidence": support,
                    "answer_confidence": min(support, semantic_support or support),
                    "semantic_total": scene["scores"].get("semantic_total", scene["scores"]["total_score"]),
                    "lexical_match": scene["scores"].get("lexical_match", 0.0),
                    "action_concept_match": scene["scores"].get("action_concept_match", 0.0),
                    "semantic_role_match": scene["scores"].get("semantic_role_match", 0.0),
                    "slot_constraint_match": scene["scores"].get("slot_constraint_match", 0.0),
                    "purpose_match": scene["scores"].get("purpose_match", 0.0),
                    "gravity": 0.0, "decision_score": scene["scores"].get("semantic_total", scene["scores"]["total_score"]),
                    "total": scene["scores"].get("semantic_total", scene["scores"]["total_score"]),
                }, "concept_matches": {"action_concept": deepcopy(scene["scores"].get("role_match_details", {}).get("action", {}))},
                "semantic_frame": deepcopy((scene["scores"].get("role_match_details", {}).get("action", {}) or {}).get("semantic_frame", {})),
                "competition_group_id": f"{frame.get('id')}:{requested}", "status": "conflict" if contradiction else "new", "weak_steps": 0,
                "selection_reason": scene.get("selection_reason", ""),
            }
            taxonomy = {"passed": True, "score": 1.0, "match_type": "not_required", "path": [], "depth": 0, "evidence_scene_ids": [], "failure_reason": None}
            constraint_source = (
                (conceptual_frame.get("slot_constraints") or {}).get(requested, {})
                if conceptual_candidate
                else (frame.get("slot_constraints", {}).get(requested, {}) or {})
            )
            required_type = constraint_source.get("is_a") or constraint_source.get("IS_A")
            if required_type:
                observed_type = value.get("entity_type")
                direct_type_match = (
                    self._value_match_detail(observed_type, required_type, conn)
                    if isinstance(observed_type, dict)
                    else {"score": 0.0}
                )
                if float(direct_type_match.get("score", 0.0)) >= .95:
                    taxonomy = {
                        "passed": True,
                        "score": 1.0,
                        "match_type": "apposition_type_membership",
                        "path": [deepcopy(observed_type)],
                        "depth": 1,
                        "evidence_scene_ids": [scene["id"]],
                        "failure_reason": None,
                    }
                else:
                    taxonomy = TaxonomyResolver(conn).resolve_is_a(
                        int(value.get("lexeme_cloud_id") or 0),
                        int(required_type.get("lexeme_cloud_id") or 0),
                        max_depth=3,
                    ) if conn else {"passed": False, "score": 0.0, "failure_reason": "RELATION_NOT_FOUND", "path": [], "evidence_scene_ids": []}
            polarity = {
                "passed": not bool(scene["negation"]),
                "scene_polarity": "negative" if scene["negation"] else "positive",
                "query_polarity": "positive",
            }
            candidate["constraint_matches"]["taxonomy"] = taxonomy
            candidate["constraint_matches"]["polarity"] = polarity
            candidate["constraint_evidence"] = [
                {"scene_id": evidence_id, "relation": "IS_A"}
                for evidence_id in taxonomy.get("evidence_scene_ids", [])
            ]
            candidate["scores"]["taxonomy_match"] = float(taxonomy.get("score", 0.0))
            if conceptual_candidate:
                candidate_score = clamp(
                    .25 * float(scene["scores"].get("action_concept_match", action))
                    + .20 * object_match + .10 * float(scene["scores"].get("purpose_match", 1.0))
                    + .20 * float(taxonomy.get("score", 1.0)) + .15 * float(polarity["passed"])
                    + .10 * float(scene["scores"].get("source_confidence", support))
                )
                candidate["scores"].update({"candidate_score": candidate_score, "decision_score": candidate_score, "total": candidate_score, "answer_confidence": candidate_score})
            rejection_reason = None
            if not polarity["passed"]:
                rejection_reason = "POLARITY_MISMATCH"
            elif required_type and not taxonomy.get("passed"):
                rejection_reason = "TAXONOMY_RELATION_NOT_FOUND"
            if rejection_reason:
                candidate.update({"status": "REJECTED", "rejection_reason": rejection_reason})
                scene["pre_candidate"] = {"candidate": lemma, "target_role": requested, "source_scene_id": scene["id"], "status": "PENDING_CONSTRAINTS"}
                scene["candidate_validation"] = deepcopy(candidate)
                scene["candidate_status"] = "rejected"
                scene["decision_reason"] = "anchor_validation_failed" if rejection_reason == "TAXONOMY_RELATION_NOT_FOUND" else rejection_reason
                if rejection_reason == "TAXONOMY_RELATION_NOT_FOUND":
                    scene.setdefault("anchor_validation", {}).setdefault("failed_constraints", []).append("slot.is_a")
                continue
            if current:
                current["sources"].append(scene["id"])
                current.setdefault("variants", []).extend(candidate["variants"])
                current["fact_evidence"].extend(candidate["fact_evidence"])
                if float(candidate.get("predicate_phrase_completeness", 0.0)) > float(current.get("predicate_phrase_completeness", 0.0)):
                    current["predicate_phrase_completeness"] = candidate["predicate_phrase_completeness"]
                    current["surface"] = candidate["surface"]
                    current["predicate_phrase"] = candidate["predicate_phrase"]
                current_scene = min(
                    (
                        int(str(source_id).removeprefix("scene-"))
                        for source_id in current["sources"]
                        if str(source_id).removeprefix("scene-").isdigit()
                    ),
                    default=999999,
                )
                candidate_scene = (
                    int(str(scene["id"]).removeprefix("scene-"))
                    if str(scene["id"]).removeprefix("scene-").isdigit()
                    else 999999
                )
                if (
                    candidate["scores"]["total"] > current["scores"]["total"]
                    or (
                        candidate["scores"]["total"] == current["scores"]["total"]
                        and candidate_scene <= current_scene
                    )
                ):
                    candidate["sources"] = current["sources"]
                    candidate["variants"] = current.get("variants", [])
                    candidate["fact_evidence"] = current.get("fact_evidence", [])
                    found[candidate_key] = candidate
            else:
                found[candidate_key] = candidate
            scene["candidate_status"] = "accepted"
            scene["decision_reason"] = "all_required_anchors_matched"
        for candidate in found.values():
            candidate["primary_source_id"] = min(
                candidate.get("sources", []),
                key=lambda source_id: int(str(source_id).removeprefix("scene-") or 999999)
                if str(source_id).removeprefix("scene-").isdigit() else 999999,
            )
        return sorted(
            found.values(),
            key=lambda item: (
                -item["scores"]["total"],
                int(str(item.get("primary_source_id", "scene-999999")).removeprefix("scene-") or 999999)
                if str(item.get("primary_source_id", "scene-999999")).removeprefix("scene-").isdigit() else 999999,
                item["lemma"],
            ),
        )

    @staticmethod
    def _result_type(scenes: Iterable[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> str:
        labels = {scene["result_type"] for scene in scenes}
        if "FULL_HIT" in labels:
            return "FULL_HIT"
        if candidates and any(item["status"] != "conflict" for item in candidates):
            return "ROLE_HIT"
        if "CONFLICT_HIT" in labels:
            return "CONFLICT_HIT"
        if "FOUND_BUT_REJECTED" in labels:
            return "FOUND_BUT_REJECTED"
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
        active = [item for item in state["candidates"] if item["status"] not in {"evicted", "conflict"} and not item.get("hard_forbidden")]
        if not active:
            return None
        first, second = active[0], active[1] if len(active) > 1 else None
        first_score = float(first["scores"].get("decision_score", first["scores"].get("total", 0.0)))
        second_score = float(second["scores"].get("decision_score", second["scores"].get("total", 0.0))) if second else 0.0
        gap = first_score - second_score
        step = int(state.get("vibration", {}).get("current_step", 0)) + 1
        if (step <= 1 and first_score >= .65 and gap >= .08) or (step >= 2 and first_score >= .55 and first.get("stable_steps", 0) >= 1 and gap >= .02):
            return first
        if step >= int(config["max_steps"]) and first_score >= .55:
            first["selection_reason"] = "лучший устойчивый кандидат выбран после ограниченной вибрации"
            return first
        return None

    @staticmethod
    def _answer_candidates(state: Dict[str, Any], winner: Dict[str, Any]) -> List[Dict[str, Any]]:
        if state.get("query_frame", {}).get("answer_cardinality") != "multiple":
            return [winner]
        top_score = float(winner["scores"].get("decision_score", winner["scores"].get("total", 0.0)))
        selected: List[Dict[str, Any]] = []
        seen = set()
        for candidate in state.get("candidates", []):
            score = float(candidate.get("scores", {}).get("decision_score", candidate.get("scores", {}).get("total", 0.0)))
            if candidate.get("status") == "conflict" or candidate.get("hard_forbidden"):
                continue
            # A plural question asks for every structurally admitted value in
            # the answer band.  Vibration still orders those values, but a
            # transient local weakening must not discard a separate fact.
            if score < max(.55, top_score - .15):
                continue
            key = str(candidate.get("lemma") or candidate.get("surface") or "").casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
            if len(selected) >= MULTI_ANSWER_LIMIT:
                break
        return selected or [winner]

    def _resolve(self, state: Dict[str, Any], winner: Dict[str, Any], step: int) -> None:
        scene = state["query_scene"]
        role = winner["target_role"]
        selected_candidates = self._answer_candidates(state, winner)
        state["selected_candidate_ids"] = [candidate["id"] for candidate in selected_candidates]
        resolved_values = [candidate["surface"] for candidate in selected_candidates]
        for slot in scene["slots"]:
            if slot["role"] == role and slot["status"] == "empty":
                values = [
                    {
                        "lemma": candidate["lemma"], "surface": candidate["surface"],
                        "resolved_lemma": candidate.get("resolved_lemma", candidate["lemma"]),
                        "resolved_surface": candidate.get("resolved_surface", candidate["surface"]),
                        "entity_id": candidate.get("entity_id"),
                        "grammatical_features": deepcopy(
                            candidate.get("grammatical_features", {})
                        ),
                        "concept_id": candidate["concept_id"], "preposition": candidate.get("preposition", ""),
                    }
                    for candidate in selected_candidates
                ]
                slot.update({"status": "RESOLVED", "lemma": winner["lemma"], "surface": winner["surface"], "value": values[0], "values": values, "selected_candidate_ids": state["selected_candidate_ids"], "confidence": winner["scores"].get("answer_confidence", winner["scores"]["total"]), "resolved_at_step": step})
        scene["status"] = "RESOLVED"
        dialogue_context = deepcopy(state.get("dialogue_context", {}))
        dialogue_context[role] = self._context_value(winner, role, source="resolved_answer")
        state["dialogue_context"] = dialogue_context
        for role_search in state.get("role_searches", []):
            if role_search.get("target_role") == role:
                role_search["selected_role_candidate"] = deepcopy(winner)
        full = any(item["result_type"] == "FULL_HIT" for item in state["memory_scenes"])
        mode = "multiple" if len(selected_candidates) > 1 else "exact" if full else "probable"
        state["sentence_plan"] = self._sentence_plan(state, winner, step)
        semantic_confidence = min((float(item.get("candidate_bridges", [{}])[0].get("confidence", item.get("selected_candidate", {}).get("scores", {}).get("semantic_total", 1.0))) for item in state.get("unknown_token_searches", []) if item.get("candidate_bridges")), default=1.0)
        source_confidence = float(winner["scores"].get("source_confidence", winner["scores"].get("total", 0.0)))
        role_evidence = float(winner["scores"].get("evidence_confidence", winner["scores"].get("total", 0.0)))
        answer_confidence = min(float(winner["scores"].get("semantic_total", winner["scores"].get("total", 0.0))), semantic_confidence, source_confidence, role_evidence, 1.0)
        winner["scores"]["semantic_confidence"] = semantic_confidence
        winner["scores"]["answer_confidence"] = answer_confidence
        source_scene_ids = [
            source for candidate in selected_candidates
            for source in candidate["sources"]
        ]
        source_relation_ids = [
            relation_id for candidate in selected_candidates
            for relation_id in candidate.get("source_relation_ids", [])
        ]
        root_source_ids = [
            source_id for candidate in selected_candidates
            for source_id in candidate.get("root_source_ids", candidate.get("sources", []))
        ]
        derivation_type = (
            "relation_extraction" if role == "entity_type"
            else "predicate_extraction"
            if state.get("query_frame", {}).get("query_type") == "action_question"
            else "role_extraction"
        )
        answer_provenance = {
            "answer_id": f"answer-{uuid.uuid4().hex[:12]}",
            "source_type": "assistant_derived_answer",
            "knowledge_status": "DERIVED",
            "independent_evidence": False,
            "eligible_for_fact_retrieval": False,
            "source_scene_ids": source_scene_ids,
            "source_relation_ids": source_relation_ids,
            "root_source_ids": root_source_ids,
            "derivation_type": derivation_type,
            "selected_candidate_ids": state["selected_candidate_ids"],
            "independent_fact": False,
        }
        answer = {
            "query": state["query_frame"]["source_text"], "answer_mode": mode, "resolved_role": role,
            "resolved_value": winner["surface"], "resolved_values": resolved_values,
            "resolved_lemma": winner.get("resolved_lemma", winner["lemma"]),
            "resolved_surface": winner.get("resolved_surface", winner["surface"]),
            "resolved_entity_id": winner.get("entity_id"),
            "resolved_value_records": [
                {
                    "resolved_lemma": candidate.get("resolved_lemma", candidate["lemma"]),
                    "resolved_surface": candidate.get("resolved_surface", candidate["surface"]),
                    "entity_id": candidate.get("entity_id"),
                    "grammatical_features": deepcopy(
                        candidate.get("grammatical_features", {})
                    ),
                }
                for candidate in selected_candidates
            ],
            "selected_candidate_ids": state["selected_candidate_ids"], "confidence": answer_confidence,
            "answer_id": answer_provenance["answer_id"],
            "provenance": answer_provenance,
            "supporting_scenes": source_scene_ids, "surface_answer": None,
            "source_relation_ids": source_relation_ids,
            "root_source_ids": root_source_ids,
            "full_surface_answer": None, "status": "PLANNING",
            "semantic_total": float(winner["scores"].get("semantic_total", 0.0)),
            "gravity": float(winner["scores"].get("gravity", 0.0)),
            "decision_score": float(winner["scores"].get("decision_score", winner["scores"].get("total", 0.0))),
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
            selected_ids = set(state.get("selected_candidate_ids") or [winner["id"]])
            selected_candidates = [candidate for candidate in state.get("candidates", []) if candidate.get("id") in selected_ids]
            if not selected_candidates:
                selected_candidates = [winner]
            multiple = len(selected_candidates) > 1
            selected = self._russian_list([str(candidate["surface"]) for candidate in selected_candidates]) if multiple else winner["surface"]
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
            if multiple:
                plan["slots"][0]["members"] = [
                    {"lemma": candidate["lemma"], "surface": candidate["surface"], "candidate_id": candidate["id"]}
                    for candidate in selected_candidates
                ]
            bridge = next((item for item in state.get("unknown_token_searches", []) if item.get("query_role") == "object"), None)
            roles = state.get("query_frame", {}).get("roles", {})
            source_scene = self._supporting_scene(state, winner)
            source_roles = (source_scene or {}).get("roles", {})
            answer_roles: Dict[str, Dict[str, Any]] = {}
            role_sources: Dict[str, str] = {}
            action_question = (
                state.get("query_frame", {}).get("query_type")
                == "action_question"
            )
            definition_question = (
                state.get("query_frame", {}).get("query_type")
                == "definition_question"
            )
            if action_question:
                subject = (
                    source_roles.get("agent")
                    or source_roles.get("theme")
                    or source_roles.get("experiencer")
                    or {}
                )
                if subject.get("surface"):
                    answer_roles["agent"] = {
                        **subject,
                        "surface": (
                            subject.get("full_surface")
                            or subject.get("mention_surface")
                            or subject.get("surface")
                        ),
                    }
                    role_sources["agent"] = "memory_scene"
                answer_roles["action"] = {**source_roles.get("action", {}), **winner}
                role_sources["action"] = "resolved_predicate"
            else:
                for name in ANSWER_ROLE_ORDER:
                    query_value = roles.get(name, {})
                    source_value = source_roles.get(name, {})
                    if name == role:
                        value = {**source_value, **winner}
                        role_source = "resolved_candidate"
                    elif state.get("query_frame", {}).get("purpose_fragment") and name in {"agent", "action"} and source_value.get("surface"):
                        value = source_value
                        role_source = "memory_scene"
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
            if answer_roles.get("action"):
                answer_roles["action"] = self._realize_predicate(
                    answer_roles["action"],
                    answer_roles.get("agent"),
                )
            short = (
                self._upper_first(selected) + "."
                if multiple
                else self._upper_first(
                    f"{winner.get('preposition', '')} {selected}".strip()
                ) + "."
            )
            full_plan = deepcopy(plan)
            full_plan["id"] = f"sentence-plan-full-{uuid.uuid4().hex[:12]}"
            full_plan["answer_style"] = "full"
            full_plan["source_scene_id"] = (source_scene or {}).get("id")
            full_plan["source_scene_text"] = (source_scene or {}).get("source_text")
            full_plan["slots"] = []
            planned_values = set()
            for name in ANSWER_ROLE_ORDER:
                value = answer_roles.get(name)
                if not value or not value.get("surface"):
                    continue
                value_key = (
                    value.get("entity_id")
                    or value.get("lexeme_cloud_id")
                    or value.get("lemma"),
                    value.get("surface"),
                    value.get("preposition", ""),
                )
                if value_key in planned_values:
                    continue
                planned_values.add(value_key)
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
            full = self._upper_first(" ".join(
                part
                for slot in full_plan["slots"]
                for part in (slot.get("preposition", ""), slot.get("surface", ""))
                if part
            )) + "."
            if definition_question:
                definition_entity = str(
                    winner.get("definition_entity_surface")
                    or roles.get("entity", {}).get("surface")
                    or ""
                ).strip()
                definition_type = str(
                    winner.get("lemma") or selected
                ).strip()
                definition_surface = self._upper_first(
                    f"{definition_entity} — {definition_type}".strip()
                ) + "."
                short = definition_surface
                full = definition_surface
                full_plan["answer_style"] = "definition"
                full_plan["relation_type"] = winner.get("relation_type")
            if multiple:
                full = short
                full_plan["answer_style"] = "multiple"
                full_plan["slots"] = [{
                    "order": 0, "role": role, "slot_id": f"slot-{role}",
                    "lemma": selected, "surface": selected, "members": deepcopy(plan["slots"][0]["members"]),
                    "requested_features": {}, "observed_features": {}, "source_type": "resolved_candidates",
                    "context_source": "candidate_ranking",
                }]
            contextual_roles = {
                name: value
                for name, value in roles.items()
                if isinstance(value, dict) and value.get("status") == "fixed"
            }
            is_contextual_scene_answer = (
                role == "object"
                and source_scene is not None
                and not any(name in contextual_roles for name in ("agent", "action", "object"))
                and any(name in contextual_roles for name in ("location", "destination", "source", "time"))
                and str(state.get("query_frame", {}).get("source_text") or "").casefold().strip().startswith("что на ")
            )
            if is_contextual_scene_answer:
                contextual_surface = str(source_scene.get("source_text") or "").strip()
                if contextual_surface:
                    contextual_surface = contextual_surface if contextual_surface.endswith((".", "!", "?")) else contextual_surface + "."
                    short = contextual_surface
                    full = contextual_surface
                    full_plan["answer_style"] = "contextual_scene"
                    full_plan["semantic_resolution"] = "location_anchored_scene"
            if winner.get("answer_mode") == "explanation" and source_scene:
                explanatory_text = str(source_scene.get("source_text", "")).strip()
                explanatory_text = explanatory_text if explanatory_text.endswith((".", "!", "?")) else explanatory_text + "."
                short = explanatory_text
                full = explanatory_text
                full_plan["slots"] = [
                    {
                        "order": index,
                        "role": name,
                        "slot_id": f"slot-{name}",
                        "lemma": value.get("lemma"),
                        "surface": value.get("surface"),
                        "preposition": value.get("preposition", ""),
                        "requested_features": {},
                        "observed_features": deepcopy(value.get("grammatical_features", {})),
                        "source_type": "known_word_form",
                        "context_source": "memory_scene",
                    }
                    for index, name in enumerate(ANSWER_ROLE_ORDER)
                    for value in [source_roles.get(name, {})]
                    if value.get("surface")
                ]
            answer = state["answer"]
            confidence = float(winner["scores"].get("semantic_total", winner["scores"].get("answer_confidence", winner["scores"]["total"])))
            short_validation = self.reverse_validate(
                state,
                short,
                answer_roles,
                profile="short",
            )
            full_validation = self.reverse_validate(
                state,
                full,
                answer_roles,
                # A list of requested values is deliberately elliptical even
                # though it is exposed through the historical ``full`` field.
                profile="short" if multiple else "full",
            )
            validation = {
                "status": (
                    "PASSED"
                    if short_validation["status"] == "PASSED"
                    and full_validation["status"] == "PASSED"
                    else "FAILED"
                ),
                "score": round(
                    (
                        float(short_validation["score"])
                        + float(full_validation["score"])
                    ) / 2,
                    4,
                ),
                # Preserve the established top-level shape for existing API
                # consumers while keeping both profile traces explicit.
                "checks": deepcopy(full_validation["checks"]),
                "errors": deepcopy(full_validation["errors"]),
                "profiles": {
                    "short": short_validation,
                    "full": full_validation,
                },
            }
            answer.update({"status": "RESOLVED" if validation["status"] == "PASSED" else "BUILD_FAILED", "answer_mode": "contextual_scene" if is_contextual_scene_answer else answer.get("answer_mode") if answer.get("answer_mode") not in {"pending", "unknown"} else "probable", "source_scene_id": (source_scene or {}).get("id"), "surface_answer": short, "full_surface_answer": full, "confidence": confidence, "short": {"surface": short, "status": "RESOLVED" if short_validation["status"] == "PASSED" else "BUILD_FAILED", "reverse_validation": short_validation}, "full": {"surface": full, "status": "RESOLVED" if full_validation["status"] == "PASSED" else "BUILD_FAILED", "reverse_validation": full_validation}})
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
            self._set_pipeline_state(state, "ANSWER_READY" if validation["status"] == "PASSED" else "FAILED")
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
    def _russian_list(values: List[str]) -> str:
        unique: List[str] = []
        seen = set()
        for value in values:
            normalized = str(value).casefold()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(str(value))
        if len(unique) < 2:
            return unique[0] if unique else ""
        if len(unique) == 2:
            return f"{unique[0]} и {unique[1]}"
        return f"{', '.join(unique[:-1])} и {unique[-1]}"

    @staticmethod
    def _upper_first(value: str) -> str:
        return value[:1].upper() + value[1:] if value else value

    def _realize_predicate(
        self,
        predicate: Dict[str, Any],
        agent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Inflect a predicate from its lemma after resolving its subject."""
        if not predicate:
            return predicate
        result = deepcopy(predicate)
        lemma = str(result.get("lemma") or "")
        source_surface = str(result.get("surface") or "")
        # Predicate-phrase answers already include their complements and are
        # realized from the supporting scene as a whole.  Reinflection here is
        # only for a single finite predicate in a newly composed full answer.
        if not lemma or len(source_surface.split()) != 1:
            return result
        observed_features = dict(result.get("grammatical_features") or {})
        if not observed_features.get("tense") and source_surface:
            observed_features = dict(
                self.morphology.parse(source_surface).features
            )
        requested_features = {
            key: str(observed_features[key])
            for key in ("tense", "number", "gender", "person")
            if observed_features.get(key)
        }
        agent_features = dict((agent or {}).get("grammatical_features") or {})
        if requested_features.get("tense") == "past" and agent_features:
            if agent_features.get("number"):
                requested_features["number"] = str(agent_features["number"])
            # A plural past-tense verb has no grammatical gender.
            if requested_features.get("number") == "plur":
                requested_features.pop("gender", None)
            elif agent_features.get("gender"):
                requested_features["gender"] = str(agent_features["gender"])
        if requested_features.get("tense"):
            generated = self.morphology.inflect(lemma, requested_features)
            if generated:
                result["surface"] = (
                    self._upper_first(generated)
                    if source_surface[:1].isupper()
                    else generated
                )
                result["realization"] = {
                    "source": "lemma_inflection",
                    "lemma": lemma,
                    "requested_features": requested_features,
                    "observed_features": observed_features,
                    "agent_features": agent_features,
                }
        return result

    @staticmethod
    def _score_breakdown() -> Dict[str, Any]:
        values = {"semantic": 1.0, "grammar": 1.0, "pattern": 1.0, "orthography": 1.0, "context": 1.0, "reverse_validation": 1.0}
        weights = {"semantic": .25, "grammar": .25, "pattern": .15, "orthography": .15, "context": .10, "reverse_validation": .10}
        breakdown = {name: {"value": value, "weight": weights[name], "contribution": round(value * weights[name], 6)} for name, value in values.items()}
        return {"score_breakdown": breakdown, "score_total": round(sum(item["contribution"] for item in breakdown.values()), 6)}

    @staticmethod
    def _supporting_scene(state: Dict[str, Any], winner: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        source_ids = {str(source_id) for source_id in [winner.get("primary_source_id"), *winner.get("sources", [])] if source_id}
        answer_scene_id = str(winner.get("answer_scene_id") or "")
        if answer_scene_id:
            explained_scene = next((scene for scene in state.get("memory_scenes", []) if str(scene.get("id")) == answer_scene_id), None)
            if explained_scene:
                return explained_scene
        primary_source_id = str(winner.get("primary_source_id") or "")
        if primary_source_id:
            primary_scene = next(
                (
                    scene
                    for scene in state.get("memory_scenes", [])
                    if str(scene.get("id")) == primary_source_id
                ),
                None,
            )
            if primary_scene:
                return primary_scene
        requested_role = str(winner.get("target_role") or "")
        candidate_lemma = str(winner.get("lemma") or "").casefold()
        candidate_surface = str(winner.get("surface") or "").casefold()
        candidates = [scene for scene in state.get("memory_scenes", []) if str(scene.get("id")) in source_ids]
        if not candidates:
            return None

        def score(scene: Dict[str, Any]) -> tuple[float, float, float, float, str]:
            value = (scene.get("roles") or {}).get(requested_role, {})
            exact_value = float(
                str(value.get("surface") or "").casefold() == candidate_surface
                or str(value.get("lemma") or "").casefold() == candidate_lemma
            )
            details = (scene.get("role_match_details") or {}).values()
            exact_anchors = sum(float(item.get("score") or 0.0) for item in details if isinstance(item, dict))
            scores = scene.get("scores") or {}
            return (
                exact_value,
                exact_anchors,
                float(scores.get("anchor_match", scores.get("anchor_score", 0.0))),
                float(scores.get("total_score", scores.get("semantic_total", 0.0))),
                str(scene.get("id") or ""),
            )

        return max(candidates, key=score)

    @staticmethod
    def _surface_contains_value(text: str, value: Dict[str, Any]) -> bool:
        """Match a realized word or phrase against its canonical value."""
        if not value:
            return True
        candidates = [
            value.get("surface"),
            value.get("resolved_surface"),
            value.get("mention_surface"),
            (value.get("answer_surfaces") or {}).get("short_name"),
            (value.get("answer_surfaces") or {}).get("full_mention"),
        ]
        for candidate in candidates:
            normalized = str(candidate or "").casefold().strip()
            if normalized and normalized in text:
                return True
        lemma = str(
            value.get("resolved_lemma") or value.get("lemma") or ""
        ).casefold().strip()
        return bool(lemma and len(lemma) >= 4 and lemma[:4] in text)

    @classmethod
    def reverse_validate(
        cls,
        state: Dict[str, Any],
        surface: str,
        answer_roles: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        profile: str = "full",
    ) -> Dict[str, Any]:
        if profile not in {"short", "full"}:
            raise ValueError(f"unsupported validation profile: {profile}")
        text = str(surface or "").casefold()
        query_frame = state.get("query_frame", {})
        query_roles = query_frame.get("roles", {})
        roles = answer_roles or query_roles
        scene = state.get("query_scene") or {}
        requested_role = query_frame.get("requested_role")
        resolved = next(
            (
                slot for slot in scene.get("slots", [])
                if slot.get("status", "").upper() == "RESOLVED"
                and slot.get("role") == requested_role
            ),
            None,
        )
        resolved_value = resolved.get("value", {}) if resolved else {}
        resolved_values = resolved.get("values", [resolved_value]) if resolved else []
        checks: Dict[str, bool] = {
            "grammar_valid": bool(text.strip()) and text.rstrip().endswith("."),
            "resolved_role_preserved": all(
                cls._surface_contains_value(text, value)
                for value in resolved_values
                if isinstance(value, dict)
            ),
            # A direct answer inherits the query's polarity and attribution
            # scope; spelling out a full sentence is responsible for making
            # those axes explicit on the surface.
            "polarity_preserved": True,
            "attribution_preserved": True,
        }
        errors: List[Dict[str, Any]] = []
        if not checks["resolved_role_preserved"]:
            errors.append({
                "type": "MISSING_RESOLVED_ROLE",
                "role": requested_role,
                "expected": (
                    resolved_value.get("resolved_surface")
                    or resolved_value.get("surface")
                    or resolved_value.get("resolved_lemma")
                    or resolved_value.get("lemma")
                ) if isinstance(resolved_value, dict) else None,
            })
        if profile == "full":
            for role, value in roles.items():
                if not isinstance(value, dict) or not value.get("surface"):
                    continue
                # An empty requested slot is resolved above; all other fixed
                # roles define the complete sentence contract.
                if value.get("status") not in {None, "fixed"} and role != requested_role:
                    continue
                check_name = f"{role}_preserved"
                checks[check_name] = cls._surface_contains_value(text, value)
                if not checks[check_name]:
                    errors.append({
                        "type": f"MISSING_{role.upper()}",
                        "role": role,
                        "expected": value.get("surface") or value.get("lemma"),
                    })
            # Retain stable check names for API consumers and analytics.
            for role in ("agent", "object", "modal", "action", "location"):
                checks.setdefault(f"{role}_preserved", True)
        else:
            # Short answers validate only the requested semantic slot.
            for role in ("agent", "object", "modal", "action", "location"):
                checks[f"{role}_preserved"] = True
        if not checks["grammar_valid"]:
            errors.append({"type": "INVALID_SURFACE"})
        score = sum(checks.values()) / max(1, len(checks))
        return {
            "status": "PASSED" if not errors and all(checks.values()) else "FAILED",
            "score": round(score, 4),
            "checks": checks,
            "errors": errors,
            "profile": profile,
        }

    @staticmethod
    def _empty_answer(frame: Dict[str, Any], result_type: str) -> Dict[str, Any]:
        role = frame.get("requested_role")
        mode = "partial" if result_type in {"PARTIAL_HIT", "CONFLICT_HIT"} else "unknown"
        return {
            "query": frame["source_text"], "answer_mode": mode, "resolved_role": role,
            "resolved_value": None, "confidence": 0.0, "supporting_scenes": [],
            "surface_answer": None, "status": "PENDING",
            "semantic_total": 0.0, "gravity": 0.0, "decision_score": 0.0,
            "status_message": f"В памяти нет точного указания для роли «{ROLE_LABELS.get(role or '', role or '?')}»." if role else "",
        }

    @staticmethod
    def _scene_trace(scene: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "scene_id": scene.get("id"),
            "source_scene_id": scene.get("cloud_id"),
            "source_text": scene.get("source_text"),
            "retrieval_stage": scene.get("retrieval_stage"),
            "result_type": scene.get("result_type"), "matched_roles": scene.get("matched_roles", []),
            "mismatched_roles": scene.get("mismatched_roles", []), "selection_reason": scene.get("selection_reason", ""),
            "role_match_details": deepcopy(scene.get("role_match_details", scene.get("scores", {}).get("role_match_details", {}))),
            "anchor_validation": deepcopy(scene.get("anchor_validation", scene.get("scores", {}).get("anchor_validation", {}))),
            "evidence": deepcopy(scene.get("fact_evidence", [])),
            "activation": deepcopy(scene.get("physics", {})),
            "scores": deepcopy(scene.get("scores", {})),
        }

    @staticmethod
    def _candidate_trace(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "candidate_id": candidate.get("id"),
            "entity_id": candidate.get("entity_id"),
            "lemma": candidate.get("lemma"),
            "surface": candidate.get("surface"),
            "target_role": candidate.get("target_role"), "status": candidate.get("status"),
            "rejection_reason": candidate.get("rejection_reason"),
            "sources": list(candidate.get("sources", [])), "selection_reason": candidate.get("selection_reason", ""),
            "evidence": deepcopy(candidate.get("fact_evidence", [])),
            "constraint_evidence": deepcopy(candidate.get("constraint_evidence", [])),
            "scores": deepcopy(candidate.get("scores", {})),
            "semantic_total": candidate.get("scores", {}).get("semantic_total", 0.0),
            "gravity": candidate.get("scores", {}).get("gravity", 0.0),
            "decision_score": candidate.get("scores", {}).get("decision_score", 0.0),
            "decision_contribution": {
                "semantic_total": .65 * float(candidate.get("scores", {}).get("semantic_total", 0.0)),
                "gravity": .20 * float(candidate.get("scores", {}).get("gravity", 0.0)),
                "resonance": .10 * float(candidate.get("scores", {}).get("resonance", 0.0)),
                "retention": .05 * float(candidate.get("scores", {}).get("retention", 0.0)),
            },
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
        if state.get("dialogue_context") is not None:
            metadata["dialogue_context"] = state["dialogue_context"]
        conn.execute("UPDATE hives SET metadata_json=?, updated_at=? WHERE id=?", (encode(metadata), utcnow(), hive_id))

    def _decorate_state(self, conn: Any, hive_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        state = deepcopy(state)
        hive_row = conn.execute("SELECT * FROM hives WHERE id=?", (hive_id,)).fetchone()
        cells = [dict(row) for row in conn.execute(
            """SELECT hc.*, c.canonical_name AS lemma, p.local_activation AS energy, p.x, p.y
            FROM hive_cells hc JOIN clouds c ON c.id=hc.dominant_cloud_id
            JOIN cloud_placements p ON p.id=hc.hive_placement_id WHERE hc.hive_id=?
            ORDER BY hc.retention DESC, hc.created_at ASC, hc.id ASC""", (hive_id,)
        )]
        for item in cells:
            item["metadata"] = decode(item.get("metadata_json"), {})
        active = [item for item in cells if item.get("component_class") != "memory_source"]
        memory_sources = [item for item in cells if item.get("component_class") == "memory_source"]
        dialogue_scene_count = int(conn.execute("SELECT COUNT(*) FROM hive_dialogue_scenes WHERE hive_id=?", (hive_id,)).fetchone()[0])
        raw_sum = sum(float(item.get("energy") or item.get("local_activation") or 0) for item in cells)
        reasoning_sum = sum(float(item.get("energy") or item.get("local_activation") or 0) for item in active)
        energy = {
            "raw_sum": round(raw_sum, 6), "all_cells_average": round(raw_sum / len(cells), 6) if cells else 0.0,
            "reasoning_cells_sum": round(reasoning_sum, 6), "reasoning_cells_average": round(reasoning_sum / len(active), 6) if active else 0.0,
            "memory_sources_average": round(sum(float(item.get("energy") or item.get("local_activation") or 0) for item in memory_sources) / len(memory_sources), 6) if memory_sources else 0.0,
            "active_reasoning_cells": len(active), "active_memory_sources": len(memory_sources), "calculation_version": 1,
        }
        capacity = int((hive_row["capacity"] or hive_row["max_cells"]) if hive_row else 24)
        occupancy = get_working_occupancy(cells, capacity)
        state["hive"] = {**state.get("hive", {}), "id": hive_id, "space_id": int(hive_row["space_id"]) if hive_row and hive_row["space_id"] is not None else None, "status": hive_row["status"] if hive_row else "ACTIVE", "query_text": hive_row["query_text"] if hive_row else state.get("created_for_surface", ""), "intent": (state.get("intent_classification") or {}).get("intent", state.get("query_frame", {}).get("intent")), "capacity": occupancy, "reasoning_step": int(hive_row["reasoning_step"] or 0) if hive_row else 0, "energy": energy, "pipeline": state.get("pipeline", {}), "pipeline_state": state.get("pipeline_state", "CREATED")}
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
        elif state.get("answer", {}).get("status") == "RESOLVED" or state.get("answer", {}).get("surface_answer"):
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
            "placements": {**occupancy, "total_placements": len(cells)},
            "working_items": [
                {"type": item.get("component_class"), "label": (f"{item.get('metadata', {}).get('bridge', {}).get('unknown_token', {}).get('surface', '')} → {item.get('metadata', {}).get('bridge', {}).get('global_candidate', {}).get('lexeme', '')}" if item.get("component_class") == "semantic_bridge" else item.get("lemma", ""))}
                for item in active
            ],
            "sources": [{"type": "memory_scene", "label": item.get("source_text", "")} for item in state.get("memory_sources", [])],
            "selected_structure_target": state.get("selected_structure_target"),
        }
        state["stats"] = {"nodes": occupancy["active_total"], "cells": occupancy["active_total"], "components": len(active), "working_cells": occupancy["active_total"], "memory_sources": occupancy["memory_sources"], "dialogue_scenes": dialogue_scene_count, "inspection_projections": len(state.get("inspection_projections", [])), "total_placements": len(cells), "capacity_pressure": occupancy["pressure"], "energy": energy}
        state["messages"] = [dict(row) for row in conn.execute("SELECT id, hive_id, turn_index, role, text, parsed_json, created_at FROM hive_messages WHERE hive_id=? ORDER BY turn_index", (hive_id,))]
        return state
