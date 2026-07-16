from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class PromptFrame:
    source_text: str
    intent: str
    roles: dict[str, dict[str, Any]]
    constraints: list[dict[str, Any]] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    exclusions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    continuation: bool = False
    polarity: str = "positive"
    modality: str = "fact"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromptFactory:
    token_pattern = re.compile(r"[a-zа-яё0-9-]+", re.IGNORECASE)
    question_roles = {
        "кто": "agent",
        "что": "object",
        "где": "location",
        "куда": "destination",
        "откуда": "source",
        "когда": "time",
        "чем": "instrument",
        "почему": "cause",
    }
    service_tokens = {
        "а",
        "и",
        "но",
        "или",
        "же",
        "ли",
        "ещё",
        "еще",
        "там",
        "тут",
        "это",
        "этот",
        "эта",
        "эти",
        "он",
        "она",
        "оно",
        "они",
        "его",
        "её",
    }

    def build(self, text: str, query_frame: Mapping[str, Any] | None = None) -> PromptFrame:
        source = text.strip()
        supplied = dict(query_frame or {})
        raw_roles = (
            supplied.get("roles", {}) if isinstance(supplied.get("roles", {}), Mapping) else {}
        )
        roles = {
            str(role).casefold(): self._normalize_role_value(value)
            for role, value in raw_roles.items()
            if value
        }
        requested = supplied.get("requested_role")
        tokens = [token.casefold() for token in self.token_pattern.findall(source)]
        if not requested:
            requested = next(
                (self.question_roles[token] for token in tokens if token in self.question_roles),
                None,
            )
        missing = [str(requested)] if requested else []
        supplied_missing = supplied.get("missing_slots") or supplied.get("unresolved_roles") or []
        for item in supplied_missing:
            role = item.get("role") if isinstance(item, Mapping) else item
            if role and str(role) not in missing:
                missing.append(str(role))
        intent = str(
            supplied.get("intent")
            or ("QUESTION" if requested or source.endswith("?") else "STATEMENT")
        )
        exclusions = supplied.get("excluded_roles") or supplied.get("exclusions") or {}
        topics = sorted(
            {
                str(
                    value.get("lemma") or value.get("normalized") or value.get("surface") or ""
                ).casefold()
                for value in roles.values()
                if value
            }
            | {
                token
                for token in tokens
                if token not in self.question_roles
                and token not in self.service_tokens
                and len(token) > 1
            }
        )
        polarity = str(supplied.get("polarity") or ("negative" if "не" in tokens else "positive"))
        return PromptFrame(
            source_text=source,
            intent=intent,
            roles=roles,
            constraints=list(supplied.get("constraints", [])),
            missing_slots=missing,
            exclusions={str(key): list(value) for key, value in exclusions.items()}
            if isinstance(exclusions, Mapping)
            else {},
            topics=[topic for topic in topics if topic],
            continuation=bool(
                supplied.get("continuation")
                or supplied.get("resolved_mode") == "FOLLOW_UP"
                or (tokens and tokens[0] == "а")
            ),
            polarity=polarity,
            modality=str(supplied.get("modality") or "fact"),
        )

    @staticmethod
    def _normalize_role_value(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            result = dict(value)
            result.setdefault(
                "lemma", result.get("normalized") or result.get("surface") or result.get("value")
            )
            return result
        return {"lemma": str(value).casefold(), "surface": str(value)}
