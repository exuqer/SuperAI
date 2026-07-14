"""Invariant validator for the canonical model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .repository import V2Repository


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
        return {"valid": not violations, "violations": violations, "checked": checks}
