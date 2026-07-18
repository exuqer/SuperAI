"""Evidence-backed action concepts and generic scene projections."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .event_core import UniversalEventPipeline, stable_id
from .repository import decode, encode, utcnow


PROJECTION_VERSION = 2
MATCH_WEIGHTS = {
    "exact_form": 1.00,
    "lemma": 0.95,
    "construction": 0.90,
    "action_concept": 0.85,
    "probable_concept": 0.75,
    "entity_relation": 0.65,
    "analogy": 0.45,
    "none": 0.00,
}


class SemanticProjectionService:
    def ensure_seeded(self, conn: Any) -> None:
        return None

    @staticmethod
    def classify_observation(text: str) -> str:
        return UniversalEventPipeline.classify_observation(text)

    def record_observation(self, conn: Any, scene_id: int, text: str) -> str:
        return self.classify_observation(text)

    @staticmethod
    def variant(conn: Any, lemma: Optional[str]) -> Optional[Dict[str, Any]]:
        if not lemma:
            return None
        row = conn.execute(
            """SELECT av.*,ac.status AS concept_status,ac.canonical_name,
                      ac.display_name
               FROM action_variants av
               JOIN action_concepts ac ON ac.id=av.action_concept_id
               WHERE av.lemma=? AND av.source_type<>'manual_seed'
                 AND ac.status IN ('PROBABLE','STABLE')
               ORDER BY CASE ac.status WHEN 'STABLE' THEN 0 ELSE 1 END,
                        av.weight DESC LIMIT 1""",
            (lemma.casefold(),),
        ).fetchone()
        return dict(row) if row else None

    def project_scene(self, conn: Any, scene_id: int) -> Optional[Dict[str, Any]]:
        scene = conn.execute(
            """SELECT 1 FROM scenes
               WHERE cloud_id=? AND knowledge_status<>'RETRACTED'""",
            (scene_id,),
        ).fetchone()
        if not scene:
            return None
        event = UniversalEventPipeline.load_event(conn, scene_id)
        if not event:
            return None
        variant = self.variant(conn, event["predicate"]["lemma"])
        if not variant:
            return None
        frame = {
            participant["role"]: {
                "entity_id": participant["entity_id"],
                "lemma": participant["lemma"],
                "surface": participant["surface"],
                "grammatical_slot": participant["grammatical_slot"],
            }
            for participant in event["participants"]
        }
        projection_id = stable_id(
            "scene-projection",
            scene_id,
            variant["action_concept_id"],
            PROJECTION_VERSION,
        )
        now = utcnow()
        conn.execute(
            """INSERT INTO scene_concept_projections
               (id,scene_id,action_concept_id,semantic_frame_json,
                projection_confidence,projection_version,source_type,created_at,updated_at)
               VALUES(?,?,?,?,?,?,'learned_evidence',?,?)
               ON CONFLICT(scene_id,action_concept_id,projection_version) DO UPDATE SET
                 semantic_frame_json=excluded.semantic_frame_json,
                 projection_confidence=excluded.projection_confidence,
                 source_type=excluded.source_type,updated_at=excluded.updated_at""",
            (
                projection_id,
                scene_id,
                variant["action_concept_id"],
                encode(frame),
                float(variant["weight"]),
                PROJECTION_VERSION,
                now,
                now,
            ),
        )
        return {
            "id": projection_id,
            "scene_id": scene_id,
            "action_concept_id": variant["action_concept_id"],
            "semantic_frame": frame,
            "projection_confidence": float(variant["weight"]),
        }

    def rebuild(self, conn: Any) -> Dict[str, int]:
        conn.execute(
            "DELETE FROM scene_concept_projections WHERE source_type='manual_seed'"
        )
        projected = 0
        for row in conn.execute(
            """SELECT cloud_id FROM scenes
               WHERE knowledge_status<>'RETRACTED' ORDER BY cloud_id"""
        ).fetchall():
            if self.project_scene(conn, int(row["cloud_id"])):
                projected += 1
        return {
            "action_concepts": int(conn.execute(
                "SELECT COUNT(*) FROM action_concepts"
            ).fetchone()[0]),
            "action_variants": int(conn.execute(
                "SELECT COUNT(*) FROM action_variants"
            ).fetchone()[0]),
            "semantic_constructions": int(conn.execute(
                "SELECT COUNT(*) FROM construction_templates"
            ).fetchone()[0]),
            "scene_concept_projections": projected,
        }

    def query_frame(self, conn: Any, frame: Dict[str, Any]) -> Dict[str, Any]:
        action = frame.get("roles", {}).get("action", {})
        variant = self.variant(conn, action.get("lemma"))
        if not variant:
            return {}
        return {
            "missing_role": frame.get("requested_role"),
            "requested_slot": frame.get("requested_slot"),
            "action_concept_id": variant["action_concept_id"],
            "action_concept": variant["canonical_name"],
            "action_concept_status": variant["concept_status"],
            "slot_constraints": frame.get("slot_constraints", {}),
            "expansion_enabled": True,
        }

    @staticmethod
    def scene_projection(
        conn: Any,
        scene_id: int,
        concept_id: str,
    ) -> Optional[Dict[str, Any]]:
        row = conn.execute(
            """SELECT p.*,c.status AS concept_status
               FROM scene_concept_projections p
               JOIN action_concepts c ON c.id=p.action_concept_id
               WHERE p.scene_id=? AND p.action_concept_id=?
                 AND p.projection_version=?""",
            (scene_id, concept_id, PROJECTION_VERSION),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["semantic_frame"] = decode(result.pop("semantic_frame_json"), {})
        return result
