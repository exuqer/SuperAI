from __future__ import annotations

from typing import Any, Mapping

from .answer_modes import AnswerMode, ResultStatus


class ResponseBuilder:
    def classify(
        self,
        answer: Mapping[str, Any] | None,
        candidates: list[Mapping[str, Any]],
        *,
        lexical_status: ResultStatus | None = None,
    ) -> dict[str, Any]:
        answer = dict(answer or {})
        confidence = float(answer.get("confidence") or 0.0)
        resolved = answer.get("status") in {"RESOLVED", "RESOLVED_GREETING"} or bool(
            answer.get("surface_answer")
        )
        if resolved and confidence >= 0.72:
            mode = AnswerMode.CONFIRMED
            status = ResultStatus.RETRIEVED
        elif resolved:
            mode = AnswerMode.PROBABLE
            status = ResultStatus.PREDICTED
        elif candidates:
            mode = (
                AnswerMode.PROBABLE
                if max(float(item.get("confidence") or 0.0) for item in candidates) >= 0.58
                else AnswerMode.PARTIAL
            )
            status = (
                ResultStatus.PREDICTED if mode == AnswerMode.PROBABLE else ResultStatus.UNVERIFIED
            )
        else:
            mode = (
                AnswerMode.PARTIAL if answer.get("answer_mode") == "partial" else AnswerMode.UNKNOWN
            )
            status = ResultStatus.UNVERIFIED
        if lexical_status == ResultStatus.COMPOSED and resolved:
            mode = AnswerMode.COMPOSITE
            status = lexical_status
        return {
            "mode": mode.value,
            "status": status.value,
            "confidence": confidence,
            "surface": answer.get("full_surface_answer") or answer.get("surface_answer"),
            "fact_supported": status in {ResultStatus.RETRIEVED, ResultStatus.PREDICTED},
            "lower_levels_created_fact": False,
        }
