"""Evidence-backed resolution of entity taxonomy relations."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional


class TaxonomyResolver:
    def __init__(self, conn: Any, depth_decay: float = .9) -> None:
        self.conn = conn
        self.depth_decay = depth_decay

    def resolve_is_a(
        self,
        subject_lexeme_id: int,
        required_type_lexeme_id: int,
        max_depth: int = 1,
    ) -> Dict[str, Any]:
        subject = self._lemma(subject_lexeme_id)
        required = self._lemma(required_type_lexeme_id)
        if not subject or not required:
            return self._failure(subject, required, "LEXEME_NOT_FOUND")
        queue = deque([(int(subject_lexeme_id), [], 0, 1.0)])
        visited = {int(subject_lexeme_id)}
        while queue:
            current, path, depth, confidence = queue.popleft()
            if depth >= max_depth:
                continue
            rows = self.conn.execute(
                """SELECT id, object_lexeme_cloud_id, confidence
                   FROM concept_relations
                   WHERE subject_lexeme_cloud_id=? AND relation_type='IS_A' AND status='STABLE'
                   ORDER BY confidence DESC, id""",
                (current,),
            ).fetchall()
            for row in rows:
                target = int(row["object_lexeme_cloud_id"])
                step = {
                    "subject": self._lemma(current),
                    "relation": "IS_A",
                    "object": self._lemma(target),
                    "relation_id": row["id"],
                }
                next_path = [*path, step]
                next_depth = depth + 1
                next_confidence = confidence * float(row["confidence"])
                if target == int(required_type_lexeme_id):
                    evidence_scene_ids = self._evidence_scene_ids(next_path)
                    score = max(0.0, min(1.0, next_confidence * self.depth_decay ** (next_depth - 1)))
                    return {
                        "passed": True,
                        "score": score,
                        "match_type": "direct_is_a" if next_depth == 1 else "transitive_is_a",
                        "subject": subject,
                        "required_type": required,
                        "path": next_path,
                        "depth": next_depth,
                        "evidence_scene_ids": evidence_scene_ids,
                        "failure_reason": None,
                    }
                if target not in visited:
                    visited.add(target)
                    queue.append((target, next_path, next_depth, next_confidence))
        return self._failure(subject, required, "RELATION_NOT_FOUND")

    def _lemma(self, lexeme_id: int) -> Optional[str]:
        row = self.conn.execute("SELECT lemma FROM lexemes WHERE cloud_id=?", (lexeme_id,)).fetchone()
        return str(row["lemma"]) if row else None

    def _evidence_scene_ids(self, path: List[Dict[str, Any]]) -> List[str]:
        relation_ids = [str(item["relation_id"]) for item in path]
        if not relation_ids:
            return []
        marks = ",".join("?" for _ in relation_ids)
        rows = self.conn.execute(
            f"""SELECT DISTINCT source_scene_id FROM concept_relation_evidence
                WHERE concept_relation_id IN ({marks})
                  AND status<>'RETRACTED'
                  AND source_scene_id IS NOT NULL
                ORDER BY source_scene_id""",
            relation_ids,
        ).fetchall()
        return [f"scene-{int(row['source_scene_id'])}" for row in rows]

    @staticmethod
    def _failure(subject: Optional[str], required: Optional[str], reason: str) -> Dict[str, Any]:
        return {
            "passed": False,
            "score": 0.0,
            "match_type": "none",
            "subject": subject,
            "required_type": required,
            "path": [],
            "depth": 0,
            "evidence_scene_ids": [],
            "failure_reason": reason,
        }
