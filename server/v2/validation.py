"""Architectural invariant checks for V2."""

from __future__ import annotations

from typing import Any, Dict, List

from .repository import V2Repository


class ModelInvariantValidator:
    def __init__(self, repository: V2Repository | None = None) -> None:
        self.repository = repository or V2Repository()

    def validate(self) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        with self.repository.transaction() as conn:
            def check(name: str, sql: str, params: tuple = ()) -> None:
                rows = conn.execute(sql, params).fetchall()
                violations.extend({"invariant": name, "row": dict(row)} for row in rows)

            check("unique_component_index", """SELECT parent_cloud_id, component_index, COUNT(*) AS count
                FROM v2_structural_components GROUP BY parent_cloud_id, component_index HAVING count > 1""")
            check("character_component_once", """SELECT parent_cloud_id, component_index, COUNT(*) AS count
                FROM v2_structural_components sc JOIN v2_clouds parent ON parent.id = sc.parent_cloud_id
                JOIN v2_clouds child ON child.id = sc.child_cloud_id
                WHERE parent.cloud_type = 'word_form' AND child.cloud_type = 'character'
                GROUP BY parent_cloud_id, component_index HAVING count <> 1""")
            check("scene_component_role", "SELECT id FROM v2_scene_components WHERE grammatical_role = '' OR confidence < 0 OR confidence > 1")
            check("word_form_lexeme_distinct", "SELECT wf.cloud_id FROM v2_word_forms wf WHERE wf.cloud_id = wf.lexeme_cloud_id")
            check("concept_lexeme_distinct", """SELECT sm.id FROM v2_semantic_memberships sm
                WHERE sm.lexeme_cloud_id = sm.concept_cloud_id""")
            check("placement_single_space", """SELECT p.id FROM v2_cloud_placements p
                LEFT JOIN v2_spaces s ON s.id = p.space_id WHERE s.id IS NULL""")
            check("scene_token_once", """SELECT scene_cloud_id, token_index, COUNT(*) AS count
                FROM v2_scene_components GROUP BY scene_cloud_id, token_index HAVING count > 1""")
            check("hive_composition_sum", """SELECT cell_id, SUM(composition_share) AS total
                FROM v2_hive_cell_components GROUP BY cell_id HAVING total < .999 OR total > 1.001""")
            check("hive_activation_range", """SELECT id FROM v2_hive_cells
                WHERE local_activation < 0 OR local_activation > 1 OR stored_strength < 0
                OR stored_strength > 1 OR retention < 0 OR retention > 1
                OR conversation_focus < 0 OR conversation_focus > 1""")
            check("hive_component_activation_range", """SELECT id FROM v2_hive_cell_components
                WHERE local_activation < 0 OR local_activation > 1""")
            check("hive_resonance_reference", """SELECT e.id FROM v2_hive_resonance_events e
                LEFT JOIN v2_hive_cells c ON c.id = e.cell_id WHERE c.id IS NULL""")
            check("hive_not_global_coordinates", """SELECT hc.id FROM v2_hive_cells hc
                JOIN v2_hives h ON h.id = hc.hive_id
                JOIN v2_cloud_placements p ON p.id = hc.source_placement_id
                WHERE ABS(hc.x - p.x) < .000001 AND ABS(hc.y - p.y) < .000001""")
            check("no_cross_space_physics", """SELECT p.id FROM v2_cloud_placements p
                JOIN v2_spaces s ON s.id = p.space_id WHERE p.space_id <> s.id""")
        return {"valid": not violations, "violations": violations, "checked": 10}
