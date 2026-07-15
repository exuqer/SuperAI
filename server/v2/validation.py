"""Invariant validator for the canonical model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .repository import V2Repository, decode


class StateConsistencyValidator:
    """Validates mutually exclusive scene and resonance presentation state."""

    def validate(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        status = state.get("display_status")
        frame = state.get("query_frame") or {}
        scene = state.get("query_scene") or {}
        probe = state.get("resonance_probe") or next((item for item in state.get("resonance_probes", []) if item.get("id") == state.get("active_resonance_probe_id")), None)
        slots = scene.get("slots", [])
        resolved = [slot for slot in slots if slot.get("status", "").upper() == "RESOLVED"]
        candidates = state.get("candidates", [])
        if status == "ROLE_RESOLVED" and (not frame.get("requested_role") or not scene or not resolved):
            errors.append({"type": "StateConsistencyError", "rule": "ROLE_RESOLVED requires a query scene, requested role, and resolved slot"})
        if status == "ANSWER_READY" and (not scene or not state.get("sentence_plan") or not state.get("answer")):
            errors.append({"type": "StateConsistencyError", "rule": "ANSWER_READY requires resolved scene, sentence plan, and answer frame"})
        if status in {"GLOBAL_MATCHES_FOUND", "LOCAL_MATCHES_FOUND", "RESONANCE_MATCHES_FOUND"} and (not probe or not (probe.get("local_results") or probe.get("global_results"))):
            errors.append({"type": "StateConsistencyError", "rule": "resonance match status requires a probe with matches"})
        if status == "ROLE_CANDIDATES_FOUND" and not candidates:
            errors.append({"type": "StateConsistencyError", "rule": "ROLE_CANDIDATES_FOUND requires candidates"})
        vibration = state.get("vibration", {})
        cells = state.get("cells", [])
        reasoning_cells = [cell for cell in cells if cell.get("component_class") != "memory_source"]
        if vibration.get("enabled") and not reasoning_cells:
            errors.append({"type": "StateConsistencyError", "rule": "vibration requires at least one reasoning cell"})
        if probe and state.get("header_input") not in {None, probe.get("input")}:
            errors.append({"type": "StateConsistencyError", "rule": "header input must equal resonance probe input"})
        dynamics = state.get("dynamics") or {}
        reasoning_cells = [cell for cell in cells if cell.get("component_class") in {"semantic_bridge", "role_candidate", "reasoning_support", "lexical_seed", "resolved_role"}]
        if dynamics.get("status") in {"STABILIZING", "MOVING", "CALCULATING_FORCES"} and not reasoning_cells:
            errors.append({"type": "StateConsistencyError", "rule": "active dynamics requires reasoning cells"})
        for candidate in candidates:
            scores = candidate.get("scores") or {}
            if float(scores.get("activation", 0) or 0) > .8 and not (scores.get("query_relevance", 0) > .7 or scores.get("exact_match") or scores.get("role_compatibility", 0) > .8 or scores.get("semantic_support", 0) > .7 or candidate.get("pinned")):
                errors.append({"type": "StateConsistencyError", "rule": "high activation requires evidence", "candidate_id": candidate.get("id")})
            if candidate.get("competition_group_id") and sum(1 for item in candidates if item.get("competition_group_id") == candidate.get("competition_group_id")) < 2:
                errors.append({"type": "StateConsistencyError", "rule": "competition requires at least two group candidates", "candidate_id": candidate.get("id")})
        if frame.get("intent") in {"SMALL_TALK", "GREETING", "GREETING_WITH_SMALL_TALK"} and (scene or frame.get("requested_role") or reasoning_cells):
            errors.append({"type": "StateConsistencyError", "rule": "small talk cannot create scene-role dynamics"})
        return errors


class ModelInvariantValidator:
    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()

    def validate(self) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        checks = 0
        with self.repository.transaction() as conn:
            def check(name: str, sql: str) -> None:
                nonlocal checks
                checks += 1
                for row in conn.execute(sql).fetchall():
                    violations.append({"invariant": name, "row": dict(row)})

            check("unique_component_index", """SELECT parent_cloud_id, component_index, COUNT(*) count
                FROM structural_components GROUP BY parent_cloud_id, component_index HAVING count > 1""")
            check("word_structure_matches_text", """SELECT wf.cloud_id, LENGTH(wf.normalized_form) expected,
                COUNT(sc.id) actual FROM word_forms wf LEFT JOIN structural_components sc
                ON sc.parent_cloud_id = wf.cloud_id GROUP BY wf.cloud_id HAVING expected <> actual""")
            check("one_word_structure_space", """SELECT owner_cloud_id, COUNT(*) count FROM spaces
                WHERE space_type = 'word_structure_space' GROUP BY owner_cloud_id HAVING count <> 1""")
            check("scene_component_role", """SELECT id FROM scene_components
                WHERE grammatical_role = '' OR confidence < 0 OR confidence > 1""")
            check("word_form_lexeme_distinct", "SELECT cloud_id FROM word_forms WHERE cloud_id = lexeme_cloud_id")
            check("scene_token_once", """SELECT scene_cloud_id, token_index, COUNT(*) count
                FROM scene_components GROUP BY scene_cloud_id, token_index HAVING count > 1""")
            check("scene_placement_space", """SELECT sc.id FROM scene_components sc
                JOIN scenes s ON s.cloud_id = sc.scene_cloud_id
                JOIN cloud_placements p ON p.id = sc.placement_id
                WHERE p.space_id <> s.scene_space_id""")
            check("placement_has_space", """SELECT p.id FROM cloud_placements p
                LEFT JOIN spaces s ON s.id = p.space_id WHERE s.id IS NULL""")
            check("characters_not_global", """SELECT p.id FROM cloud_placements p
                JOIN clouds c ON c.id = p.cloud_id JOIN spaces s ON s.id = p.space_id
                WHERE c.cloud_type = 'character' AND s.space_type = 'global_field'""")
            check("hive_placement_space", """SELECT hc.id FROM hive_cells hc
                JOIN hives h ON h.id = hc.hive_id JOIN cloud_placements p ON p.id = hc.hive_placement_id
                WHERE p.space_id <> h.space_id""")
            check("hive_source_not_local", """SELECT hc.id FROM hive_cells hc
                WHERE hc.source_placement_id = hc.hive_placement_id""")
            check("hive_composition_sum", """SELECT cell_id, SUM(composition_share) total
                FROM hive_cell_components GROUP BY cell_id HAVING total < .999 OR total > 1.001""")
            checks += 1
            for row in conn.execute("SELECT id, metadata_json FROM hives").fetchall():
                working = decode(row["metadata_json"], {}).get("query_working_memory", {})
                frame = working.get("query_frame", {})
                surfaces = {str(token.get("surface", "")).casefold() for token in frame.get("tokens", [])}
                for search in working.get("unknown_token_searches", []):
                    for bridge in search.get("candidate_bridges", []):
                        surface = str(bridge.get("unknown_token", {}).get("surface", "")).casefold()
                        if surface and surface not in surfaces:
                            violations.append({"invariant": "TEMPORARY_OBJECT_QUERY_MISMATCH", "row": {"hive_id": row["id"], "bridge_id": bridge.get("id"), "surface": surface}})
        return {"valid": not violations, "violations": violations, "checked": checks}
