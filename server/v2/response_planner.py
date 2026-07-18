"""Content-first response planning and semantic reverse validation."""

from __future__ import annotations

import re
import uuid
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .language.models import ResponsePlan, ResponseType
from .repository import encode, utcnow


def _stable_id(prefix: str, *parts: object) -> str:
    key = "|".join(str(part) for part in parts)
    return f"{prefix}-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:20]}"


class ResponsePlanner:
    def plan(
        self,
        *,
        interpretation_status: str = "STABLE",
        query_frame: Optional[Mapping[str, Any]] = None,
        answer: Optional[Mapping[str, Any]] = None,
        candidates: Sequence[Mapping[str, Any]] = (),
        dialogue_state: Optional[Mapping[str, Any]] = None,
        source_evidence: Sequence[Mapping[str, Any]] = (),
    ) -> ResponsePlan:
        frame = query_frame or {}
        current_answer = answer or {}
        state = dialogue_state or {}
        acts = frame.get("dialogue_acts") or []
        target_act_id = next(
            (
                item.get("id") for item in acts
                if item.get("act_type") in {"QUESTION", "CORRECTION", "REQUEST"}
            ),
            None,
        )
        correction = frame.get("correction")
        clarification = (
            frame.get("pending_clarification")
            or state.get("pending_clarification")
        )
        if clarification:
            response_type = ResponseType.CLARIFICATION
            content = {
                "question": clarification.get("question"),
                "slot": clarification.get("slot"),
                "candidates": deepcopy(clarification.get("candidates") or []),
            }
        elif correction:
            response_type = ResponseType.CORRECTION_ACK
            content = {
                "correction": deepcopy(correction),
                "message": "Исправление принято.",
                "value": deepcopy(current_answer.get("resolved_value")),
                "surface": (
                    current_answer.get("surface_answer")
                    or current_answer.get("surface")
                ),
            }
        elif interpretation_status == "CONFLICTED":
            response_type = ResponseType.CONFLICT
            content = {
                "message": (
                    "В диалоге есть два несовместимых утверждения."
                ),
                "conflicts": deepcopy(
                    current_answer.get("conflicts") or []
                ),
            }
        elif interpretation_status == "AMBIGUOUS":
            response_type = ResponseType.AMBIGUOUS
            content = {
                "message": "Реплика допускает несколько интерпретаций.",
                "alternatives": deepcopy(
                    current_answer.get("alternatives") or []
                ),
            }
        elif current_answer.get("evidence_status") in {
            "CONFIRMED",
            "SUPPORTED",
            "POLAR_CONFIRMED",
        } and frame.get("query_type") == "polar_question":
            response_type = ResponseType.CONFIRMATION
            content = {
                "value": current_answer.get("resolved_value"),
                "surface": (
                    current_answer.get("surface_answer")
                    or current_answer.get("surface")
                ),
            }
        elif (
            str(current_answer.get("status") or "").startswith("RESOLVED")
            or candidates
        ):
            response_type = (
                ResponseType.DIRECT
                if frame.get("requested_role") else ResponseType.FULL
            )
            content = {
                "value": current_answer.get("resolved_value")
                or (deepcopy(candidates[0]) if candidates else None),
                "surface": (
                    current_answer.get("surface_answer")
                    or current_answer.get("surface")
                ),
            }
        else:
            response_type = ResponseType.UNKNOWN
            content = {
                "message": (
                    "В доступной памяти нет достаточного свидетельства."
                ),
                "requested_role": frame.get("requested_role"),
            }
        attribution = current_answer.get("attribution")
        if not attribution:
            quoted = next(
                (
                    clause.get("quoted_speaker")
                    for clause in frame.get("clauses") or []
                    if clause.get("quoted_speaker")
                ),
                None,
            )
            if quoted:
                attribution = {"speaker": quoted}
        return ResponsePlan(
            response_type=response_type,
            target_act_id=target_act_id,
            focus_role=frame.get("requested_role"),
            content_slots=content,
            source_evidence=[deepcopy(item) for item in source_evidence],
            uncertainty=(
                {
                    "interpretation_status": interpretation_status,
                    "confidence": current_answer.get("confidence", 0.0),
                }
                if response_type in {
                    ResponseType.UNKNOWN,
                    ResponseType.AMBIGUOUS,
                    ResponseType.CONFLICT,
                    ResponseType.CLARIFICATION,
                }
                else None
            ),
            attribution=deepcopy(attribution) if attribution else None,
            surface_constraints={
                "language": "ru",
                "answer_style": (
                    "short"
                    if response_type in {
                        ResponseType.DIRECT,
                        ResponseType.CONFIRMATION,
                    }
                    else "full"
                ),
                "preserve": [
                    "roles",
                    "polarity",
                    "modality",
                    "actuality",
                    "attribution",
                ],
            },
        )

    @staticmethod
    def _russian_list(values: Iterable[str]) -> str:
        items = [str(value) for value in values if value]
        if len(items) < 2:
            return items[0] if items else ""
        if len(items) == 2:
            return f"{items[0]} или {items[1]}"
        return f"{', '.join(items[:-1])} или {items[-1]}"

    def realize(self, plan: ResponsePlan) -> str:
        slots = plan.content_slots
        existing = str(slots.get("surface") or "").strip()
        if existing:
            return existing
        if plan.response_type == ResponseType.CLARIFICATION:
            question = str(slots.get("question") or "").strip()
            if question:
                return question
            candidates = [
                item.get("surface") or item.get("lemma")
                for item in slots.get("candidates") or []
            ]
            return (
                f"Уточните значение для роли «{slots.get('slot') or '?'}»: "
                f"{self._russian_list(candidates)}?"
            )
        if plan.response_type == ResponseType.CORRECTION_ACK:
            return str(slots.get("message") or "Исправление принято.")
        if plan.response_type == ResponseType.UNKNOWN:
            return str(
                slots.get("message")
                or "В доступной памяти нет достаточного свидетельства."
            )
        if plan.response_type == ResponseType.AMBIGUOUS:
            return str(
                slots.get("message")
                or "Реплика допускает несколько интерпретаций."
            )
        if plan.response_type == ResponseType.CONFLICT:
            return str(
                slots.get("message")
                or "В диалоге есть несовместимые утверждения."
            )
        if plan.response_type == ResponseType.CONFIRMATION:
            return "Да."
        value = slots.get("value")
        if isinstance(value, dict):
            result = value.get("surface") or value.get("lemma")
        else:
            result = value
        surface = str(result or "").strip()
        if plan.attribution and plan.attribution.get("speaker") and surface:
            surface = (
                f"По словам {plan.attribution['speaker']}, "
                f"{surface[:1].lower() + surface[1:]}"
            )
        return surface or "В доступной памяти нет достаточного свидетельства."

    @staticmethod
    def reverse_validate(
        plan: ResponsePlan,
        surface: str,
        *,
        semantic_axes: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = str(surface or "").casefold()
        axes = semantic_axes or {}
        checks: Dict[str, bool] = {
            "surface_present": bool(text.strip()),
            "roles_preserved": True,
            "polarity_preserved": True,
            "modality_preserved": True,
            "actuality_preserved": True,
            "attribution_preserved": True,
        }
        errors = []
        if plan.response_type == ResponseType.DIRECT:
            focus_value = plan.content_slots.get("value")
            if isinstance(focus_value, Mapping):
                focus_surface = str(
                    focus_value.get("surface")
                    or focus_value.get("lemma")
                    or ""
                ).casefold()
            else:
                focus_surface = str(focus_value or "").casefold()
            if focus_surface:
                checks["roles_preserved"] = (
                    focus_surface in text
                    or (
                        len(focus_surface) >= 4
                        and focus_surface[:4] in text
                    )
                )
                if not checks["roles_preserved"]:
                    errors.append({
                        "type": "MISSING_ROLE",
                        "role": plan.focus_role,
                    })
        else:
            for role, value in (axes.get("roles") or {}).items():
                if not isinstance(value, dict) or not value.get("surface"):
                    continue
                lemma = str(value.get("lemma") or "").casefold()
                role_present = (
                    str(value["surface"]).casefold() in text
                    or bool(lemma and lemma[:4] in text)
                )
                checks["roles_preserved"] = (
                    checks["roles_preserved"] and role_present
                )
                if not role_present:
                    errors.append({"type": "MISSING_ROLE", "role": role})
        elliptical = plan.response_type in {
            ResponseType.DIRECT,
            ResponseType.CONFIRMATION,
        }
        polarity = str(axes.get("polarity") or "POSITIVE").upper()
        if polarity == "NEGATIVE" and not elliptical:
            checks["polarity_preserved"] = bool(
                re.search(r"\b(?:не|нет|нельзя)\b", text)
            )
            if not checks["polarity_preserved"]:
                errors.append({"type": "MISSING_NEGATION"})
        modality = str(axes.get("modality") or "").upper()
        modality_markers = {
            "CAN": ("мож",),
            "MUST": ("долж", "надо", "нужно"),
            "MAY": ("можно", "возможно"),
            "SHOULD": ("следует", "стоило"),
            "WANT": ("хоч",),
            "INTEND": ("намер", "собира"),
            "TRY": ("пыта",),
            "BELIEVE": ("счита", "вер"),
            "KNOW": ("зна",),
        }
        if modality in modality_markers and not elliptical:
            checks["modality_preserved"] = any(
                marker in text for marker in modality_markers[modality]
            )
            if not checks["modality_preserved"]:
                errors.append({"type": "MISSING_MODALITY", "modality": modality})
        actuality = str(axes.get("actuality") or "ACTUAL").upper()
        if (
            actuality in {
                "POSSIBLE",
                "HYPOTHETICAL",
                "COUNTERFACTUAL",
            }
            and not elliptical
        ):
            checks["actuality_preserved"] = (
                "если" in text
                or "бы" in text
                or any(
                    marker in text
                    for marker in (
                        "наверн",
                        "возможн",
                        "вероятн",
                        "похоже",
                        "кажется",
                    )
                )
            )
            if not checks["actuality_preserved"]:
                errors.append({"type": "MISSING_HYPOTHETICAL_MARKER"})
        attribution = plan.attribution or {}
        if attribution.get("speaker"):
            speaker = str(attribution["speaker"]).casefold()
            checks["attribution_preserved"] = (
                speaker in text and "слов" in text
            )
            if not checks["attribution_preserved"]:
                errors.append({
                    "type": "MISSING_ATTRIBUTION",
                    "speaker": attribution["speaker"],
                })
        score = sum(checks.values()) / len(checks)
        return {
            "status": "PASSED" if all(checks.values()) else "FAILED",
            "score": round(score, 6),
            "checks": checks,
            "errors": errors,
        }

    def persist(
        self,
        conn: Any,
        plan: ResponsePlan,
        *,
        conversation_id: str = "",
        source_utterance_id: Optional[str] = None,
        surface: Optional[str] = None,
        independent_source_count: int = 0,
        semantic_axes: Optional[Mapping[str, Any]] = None,
        persist_derived: bool = True,
    ) -> Dict[str, Any]:
        surface_text = surface or self.realize(plan)
        validation = self.reverse_validate(
            plan,
            surface_text,
            semantic_axes=semantic_axes,
        )
        plan_id = _stable_id(
            "response-plan",
            conversation_id,
            source_utterance_id,
            plan.response_type.value,
            surface_text,
        )
        conn.execute(
            """INSERT OR REPLACE INTO response_plans
               (id,conversation_id,source_utterance_id,response_type,
                target_act_id,focus_role,plan_json,reverse_validation_json,
                status,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                plan_id,
                conversation_id or None,
                source_utterance_id,
                plan.response_type.value,
                plan.target_act_id,
                plan.focus_role,
                encode(plan.as_dict()),
                encode(validation),
                (
                    "PLANNED"
                    if not persist_derived
                    else "VALIDATED"
                    if validation["status"] == "PASSED"
                    else "NEEDS_REBUILD"
                ),
                utcnow(),
            ),
        )
        answer_id = None
        if persist_derived:
            answer_id = _stable_id("derived-answer", plan_id, surface_text)
            conn.execute(
                """INSERT OR REPLACE INTO derived_answers
                   (id,response_plan_id,conversation_id,surface_text,
                    full_surface_text,source_evidence_json,
                    independent_source_count,attribution_json,status,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    answer_id,
                    plan_id,
                    conversation_id or None,
                    surface_text,
                    surface_text,
                    encode(plan.source_evidence),
                    int(independent_source_count),
                    encode(plan.attribution) if plan.attribution else None,
                    "DERIVED_ANSWER",
                    utcnow(),
                ),
            )
        return {
            "id": plan_id,
            "plan": plan.as_dict(),
            "surface": surface_text,
            "reverse_validation": validation,
            "derived_answer_id": answer_id,
            "independent_source_count": int(independent_source_count),
            "status": "DERIVED_ANSWER" if persist_derived else "PLANNED",
        }
