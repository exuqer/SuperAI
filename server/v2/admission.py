"""Rules controlling which query objects may enter physical dynamics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


FUNCTION_PARTS = {"PREPOSITION", "CONJUNCTION", "PARTICLE", "AUXILIARY", "ADP", "CCONJ", "SCONJ", "PART", "AUX"}
REASONING_CLASSES = {"semantic_bridge", "role_candidate", "competing_hypothesis", "lexical_seed", "resolved_role", "reasoning_support"}


class DynamicsAdmissionService:
    minimum_relevance = 0.12

    def admit(self, item: Dict[str, Any], *, query_session_id: Optional[str] = None, selected_memory: bool = False, existing_ids: Iterable[str] = ()) -> Dict[str, Any]:
        reasons = []
        component = str(item.get("component_class") or item.get("type") or "")
        metadata = item.get("metadata") or {}
        item_session = item.get("query_session_id") or metadata.get("query_session_id")
        if query_session_id and item_session and item_session != query_session_id:
            reasons.append("other_query_session")
        if item.get("id") in set(existing_ids) or item.get("duplicate_of"):
            reasons.append("duplicate_projection")
        pos = str(item.get("part_of_speech") or item.get("operator_type") or "").upper()
        if component == "function_operator" or pos in FUNCTION_PARTS:
            reasons.append("function_word")
        if component in {"search_hit", "inspection_projection", "SEARCH_HIT", "INSPECTION_PROJECTION"}:
            reasons.append("search_only")
        relevance = float(item.get("query_relevance", (item.get("scores") or {}).get("query_relevance", (item.get("scores") or {}).get("total", 0.0))) or 0.0)
        support = float(item.get("semantic_support", (item.get("scores") or {}).get("semantic_support", (item.get("scores") or {}).get("semantic_confidence", 0.0))) or 0.0)
        compatibility = float(item.get("role_compatibility", (item.get("scores") or {}).get("role_compatibility", 0.0)) or 0.0)
        if component == "memory_source":
            if not (selected_memory or item.get("selection_status") == "SELECTED"):
                reasons.append("not_selected_as_memory_source")
        elif component not in REASONING_CLASSES:
            reasons.append("not_reasoning_cell")
        elif relevance < self.minimum_relevance and support <= 0 and compatibility <= 0:
            reasons.append("not_relevant_to_query")
        if reasons:
            status = "REJECTED_FUNCTION_WORD" if "function_word" in reasons else "REJECTED_SEARCH_ONLY" if "search_only" in reasons else "REJECTED_DUPLICATE" if "duplicate_projection" in reasons else "REJECTED_OTHER_QUERY" if "other_query_session" in reasons else "REJECTED_LOW_RELEVANCE" if "not_relevant_to_query" in reasons else "REJECTED_UNSUPPORTED"
            return {"object_id": item.get("id"), "status": status, "reasons": reasons}
        return {"object_id": item.get("id"), "status": "ADMITTED_MEMORY_SOURCE" if component == "memory_source" else "ADMITTED_REASONING_CELL", "reasons": []}
