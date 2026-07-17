"""Training and backfill for classification relations between lexemes."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

from .repository import encode, utcnow


CLASSIFICATION_MARKERS = {"это", "является", "являться", "относится", "относиться", "представляет", "представлять"}


class ConceptRelationTrainer:
    confidence = .95

    def classify_scene(self, conn: Any, scene_id: int) -> Optional[Dict[str, Any]]:
        scene = conn.execute("SELECT sentence_text FROM scenes WHERE cloud_id=?", (scene_id,)).fetchone()
        if not scene:
            return None
        rows = conn.execute(
            """SELECT sc.token_index, sc.lexeme_cloud_id, sc.grammatical_role, l.lemma, l.pos_tag,
                      wf.normalized_form
               FROM scene_components sc
               JOIN lexemes l ON l.cloud_id=sc.lexeme_cloud_id
               JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id
               WHERE sc.scene_cloud_id=? AND sc.lexeme_cloud_id IS NOT NULL
               ORDER BY sc.token_index""",
            (scene_id,),
        ).fetchall()
        words = [dict(row) for row in rows]
        marker_index = next((int(word["token_index"]) for word in words if str(word["normalized_form"]).casefold() in CLASSIFICATION_MARKERS), None)
        has_dash = "—" in str(scene["sentence_text"]) or " - " in str(scene["sentence_text"])
        if marker_index is None and not has_dash:
            return None
        nouns = [word for word in words if word.get("pos_tag") in {"NOUN", "NPRO"}]
        if len(nouns) < 2:
            return None
        if marker_index is None:
            subject, target = nouns[0], nouns[-1]
        else:
            left = [word for word in nouns if int(word["token_index"]) < marker_index]
            right = [word for word in nouns if int(word["token_index"]) > marker_index]
            if not left or not right:
                return None
            subject, target = left[-1], right[0]
        if int(subject["lexeme_cloud_id"]) == int(target["lexeme_cloud_id"]):
            return None
        return {"subject_lexeme_cloud_id": int(subject["lexeme_cloud_id"]), "object_lexeme_cloud_id": int(target["lexeme_cloud_id"])}

    def materialize(self, conn: Any, scene_id: int, observation_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        classification = self.classify_scene(conn, scene_id)
        if not classification:
            return None
        subject_id = classification["subject_lexeme_cloud_id"]
        object_id = classification["object_lexeme_cloud_id"]
        relation_id = "relation-" + hashlib.sha256(f"IS_A:{subject_id}:{object_id}".encode()).hexdigest()[:20]
        now = utcnow()
        conn.execute(
            """INSERT INTO concept_relations
               (id, relation_type, subject_lexeme_cloud_id, object_lexeme_cloud_id, confidence,
                status, direct, depth, evidence_count, source_type, created_at, updated_at)
               VALUES (?, 'IS_A', ?, ?, ?, 'STABLE', 1, 1, 0, 'CLASSIFICATION_DEFINITION', ?, ?)
               ON CONFLICT(relation_type, subject_lexeme_cloud_id, object_lexeme_cloud_id) DO UPDATE SET
                 confidence=MAX(concept_relations.confidence, excluded.confidence), updated_at=excluded.updated_at""",
            (relation_id, subject_id, object_id, self.confidence, now, now),
        )
        relation = conn.execute(
            """SELECT id FROM concept_relations WHERE relation_type='IS_A'
               AND subject_lexeme_cloud_id=? AND object_lexeme_cloud_id=?""",
            (subject_id, object_id),
        ).fetchone()
        relation_id = str(relation["id"])
        evidence_id = "evidence-" + hashlib.sha256(f"{relation_id}:{scene_id}:{observation_id or ''}".encode()).hexdigest()[:20]
        existing = conn.execute(
            """SELECT 1 FROM concept_relation_evidence
               WHERE concept_relation_id=? AND source_scene_id=?""",
            (relation_id, scene_id),
        ).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO concept_relation_evidence
                   (id, concept_relation_id, source_scene_id, source_training_observation_id, evidence_type,
                    relation_type, weight, confidence, status, evidence_json, created_at)
                   VALUES (?, ?, ?, ?, 'CLASSIFICATION_DEFINITION', 'IS_A', ?, ?, 'STABLE', ?, ?)""",
                (evidence_id, relation_id, scene_id, observation_id, self.confidence, self.confidence,
                 encode({"subject_lexeme_cloud_id": subject_id, "object_lexeme_cloud_id": object_id}), now),
            )
        count = int(conn.execute(
            "SELECT COUNT(*) FROM concept_relation_evidence WHERE concept_relation_id=?", (relation_id,)
        ).fetchone()[0])
        conn.execute("UPDATE concept_relations SET evidence_count=?, updated_at=? WHERE id=?", (count, now, relation_id))
        return {"id": relation_id, "relation_type": "IS_A", **classification, "confidence": self.confidence, "evidence_scene_ids": [f"scene-{scene_id}"]}

    def rebuild(self, conn: Any) -> Dict[str, Any]:
        count = 0
        for row in conn.execute("SELECT cloud_id FROM scenes ORDER BY cloud_id").fetchall():
            observation = conn.execute(
                "SELECT id FROM training_observations WHERE scene_cloud_id=? ORDER BY id LIMIT 1", (row["cloud_id"],)
            ).fetchone()
            if self.materialize(conn, int(row["cloud_id"]), int(observation["id"]) if observation else None):
                count += 1
        by_type = {str(row["relation_type"]): int(row["count"]) for row in conn.execute(
            "SELECT relation_type, COUNT(*) AS count FROM concept_relations GROUP BY relation_type"
        ).fetchall()}
        return {"processed_classification_scenes": count, "concept_relations_total": sum(by_type.values()), "concept_relations_by_type": by_type}
