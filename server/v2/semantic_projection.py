"""Persistent action concepts and projections from grammatical scenes."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

from .repository import decode, encode, utcnow


PROJECTION_VERSION = 1
MATCH_WEIGHTS = {
    "exact_form": 1.00,
    "lemma": 0.95,
    "construction": 0.94,
    "action_concept": 0.90,
    "probable_concept": 0.75,
    "implied_semantic_role": 0.82,
    "none": 0.00,
}

CONSUME_AS_FOOD_ID = "action-concept-consume-as-food"
_CONSUME_CONSTRUCTIONS = (
    ("construction-eat-object", "есть", "transitive_object", {"subject": "consumer", "object": "food"}, {"purpose": "пища"}, .95),
    ("construction-feed-on-theme", "питаться", "instrumental_theme", {"subject": "consumer", "object": "food"}, {"purpose": "пища"}, .93),
    ("construction-consume-as-food", "употреблять", "purpose_phrase", {"subject": "consumer", "object": "food", "purpose": "purpose"}, {}, 1.0),
)


class SemanticProjectionService:
    """Keeps concept knowledge independent from the parser and retrieval code."""

    def ensure_seeded(self, conn: Any) -> None:
        now = utcnow()
        global_space = conn.execute("SELECT id FROM spaces WHERE space_type='global_field' LIMIT 1").fetchone()
        conn.execute(
            """INSERT INTO action_concepts
            (id,canonical_name,display_name,space_id,status,confidence,mass,evidence_count,created_at,updated_at)
            VALUES(?,?,?,?, 'STABLE', .92, .74, 3, ?, ?)
            ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at""",
            (CONSUME_AS_FOOD_ID, "consume_as_food", "употребление пищи", int(global_space["id"]) if global_space else None, now, now),
        )
        for construction_id, lemma, pattern, mapping, implied, weight in _CONSUME_CONSTRUCTIONS:
            conn.execute(
                """INSERT INTO semantic_constructions
                (id,predicate_lemma,pattern_type,argument_mapping_json,implied_semantics_json,confidence,evidence_count)
                VALUES(?,?,?,?,?,?,1)
                ON CONFLICT(id) DO UPDATE SET predicate_lemma=excluded.predicate_lemma,
                pattern_type=excluded.pattern_type,argument_mapping_json=excluded.argument_mapping_json,
                implied_semantics_json=excluded.implied_semantics_json,confidence=excluded.confidence""",
                (construction_id, lemma, pattern, encode(mapping), encode(implied), weight),
            )
            lexeme = conn.execute("SELECT cloud_id FROM lexemes WHERE lemma=? LIMIT 1", (lemma,)).fetchone()
            conn.execute(
                """INSERT INTO action_variants
                (id,action_concept_id,lexeme_cloud_id,lemma,construction_id,weight,evidence_count,source_type,created_at,updated_at)
                VALUES(?,?,?,?,?,?,1,'manual_seed',?,?)
                ON CONFLICT(action_concept_id,lemma) DO UPDATE SET
                lexeme_cloud_id=COALESCE(excluded.lexeme_cloud_id,action_variants.lexeme_cloud_id),
                construction_id=excluded.construction_id,weight=excluded.weight,updated_at=excluded.updated_at""",
                (f"action-variant-{lemma}", CONSUME_AS_FOOD_ID, int(lexeme["cloud_id"]) if lexeme else None, lemma, construction_id, weight, now, now),
            )

    def classify_observation(self, text: str) -> str:
        normalized = " ".join(text.casefold().replace("—", "-").split())
        if re.search(r"\b.+\s+и\s+.+\s*-\s*близк\w* действия", normalized) and "употреблен" in normalized:
            return "ACTION_GENERALIZATION"
        if "противополож" in normalized:
            return "SEMANTIC_OPPOSITION"
        if "близк" in normalized or "связан" in normalized:
            return "SEMANTIC_SIMILARITY"
        if (
            "это" in normalized or "означает" in normalized or "является" in normalized
            or "относится к" in normalized or "представляет собой" in normalized
            or "—" in text or " - " in text
        ):
            return "CLASSIFICATION_DEFINITION"
        return "WORLD_EVENT"

    def record_observation(self, conn: Any, scene_id: int, text: str) -> str:
        kind = self.classify_observation(text)
        if kind != "ACTION_GENERALIZATION":
            return kind
        self.ensure_seeded(conn)
        now = utcnow()
        conn.execute(
            """INSERT OR IGNORE INTO concept_relation_evidence
            (id,source_scene_id,relation_type,source_concept_id,target_concept_id,weight,confidence,status,evidence_json,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (f"concept-evidence-{scene_id}-{CONSUME_AS_FOOD_ID}", scene_id, "ACTION_GENERALIZATION", CONSUME_AS_FOOD_ID, CONSUME_AS_FOOD_ID, .72, .72, "PROBABLE", encode({"text": text}), now),
        )
        return kind

    def project_scene(self, conn: Any, scene_id: int) -> Optional[Dict[str, Any]]:
        self.ensure_seeded(conn)
        rows = conn.execute(
            """SELECT sc.grammatical_role,sc.morphology_json,l.lemma,wf.normalized_form,
                      l.cloud_id AS lexeme_cloud_id
               FROM scene_components sc
               JOIN lexemes l ON l.cloud_id=sc.lexeme_cloud_id
               JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id
               WHERE sc.scene_cloud_id=? ORDER BY sc.token_index""",
            (scene_id,),
        ).fetchall()
        roles: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            role = {"subject": "agent", "predicate": "action"}.get(str(row["grammatical_role"]), str(row["grammatical_role"]))
            if role not in {"agent", "action", "object", "purpose"} or role in roles:
                continue
            morphology = decode(row["morphology_json"], {})
            roles[role] = {
                "lemma": row["lemma"], "surface": row["normalized_form"],
                "lexeme_cloud_id": int(row["lexeme_cloud_id"]), "case": morphology.get("case"),
            }
        action = roles.get("action", {})
        variant = self.variant(conn, action.get("lemma"))
        if not variant or not roles.get("agent") or not roles.get("object"):
            return None
        construction = conn.execute("SELECT * FROM semantic_constructions WHERE id=?", (variant["construction_id"],)).fetchone()
        implied = decode(construction["implied_semantics_json"], {}) if construction else {}
        frame = {
            "consumer": roles["agent"], "food": roles["object"],
            "purpose": roles.get("purpose") or {"lemma": implied.get("purpose"), "surface": implied.get("purpose")},
            "action_variant": {"lemma": action.get("lemma"), "construction_id": variant.get("construction_id")},
        }
        now = utcnow()
        projection_id = f"scene-projection-{scene_id}-{variant['action_concept_id']}"
        conn.execute(
            """INSERT INTO scene_concept_projections
            (id,scene_id,action_concept_id,semantic_frame_json,projection_confidence,projection_version,source_type,created_at,updated_at)
            VALUES(?,?,?,?,?,?, 'scene_parser', ?, ?)
            ON CONFLICT(scene_id,action_concept_id,projection_version) DO UPDATE SET
            semantic_frame_json=excluded.semantic_frame_json,projection_confidence=excluded.projection_confidence,
            updated_at=excluded.updated_at""",
            (projection_id, scene_id, variant["action_concept_id"], encode(frame), float(variant["weight"]), PROJECTION_VERSION, now, now),
        )
        return {"id": projection_id, "scene_id": scene_id, "action_concept_id": variant["action_concept_id"], "semantic_frame": frame, "projection_confidence": float(variant["weight"])}

    def rebuild(self, conn: Any) -> Dict[str, int]:
        self.ensure_seeded(conn)
        conn.execute("DELETE FROM scene_concept_projections WHERE projection_version=?", (PROJECTION_VERSION,))
        projected = 0
        for row in conn.execute("SELECT cloud_id FROM scenes ORDER BY cloud_id").fetchall():
            if self.project_scene(conn, int(row["cloud_id"])):
                projected += 1
        return {"action_concepts": 1, "action_variants": len(_CONSUME_CONSTRUCTIONS), "semantic_constructions": len(_CONSUME_CONSTRUCTIONS), "scene_concept_projections": projected}

    @staticmethod
    def variant(conn: Any, lemma: Optional[str]) -> Optional[Dict[str, Any]]:
        if not lemma:
            return None
        row = conn.execute("SELECT * FROM action_variants WHERE lemma=? ORDER BY weight DESC LIMIT 1", (lemma.casefold(),)).fetchone()
        return dict(row) if row else None

    def query_frame(self, conn: Any, frame: Dict[str, Any]) -> Dict[str, Any]:
        action = frame.get("roles", {}).get("action", {})
        variant = self.variant(conn, action.get("lemma"))
        if not variant:
            return {}
        purpose = (frame.get("semantic_constraints") or {}).get("purpose")
        requested = frame.get("requested_role")
        conceptual_constraints = {}
        for role, constraints in (frame.get("slot_constraints") or {}).items():
            canonical_role = "consumer" if role in {"agent", "subject", "consumer"} else role
            conceptual_constraints[canonical_role] = constraints
        return {
            "missing_role": "consumer" if requested == "agent" else requested,
            "action_concept_id": variant["action_concept_id"],
            "action_concept": "consume_as_food" if variant["action_concept_id"] == CONSUME_AS_FOOD_ID else variant["action_concept_id"],
            "food": frame.get("roles", {}).get("object"),
            "purpose": purpose or {"lemma": "пища", "source": "implied"},
            "slot_constraints": conceptual_constraints,
            "expansion_enabled": bool(purpose),
        }

    @staticmethod
    def scene_projection(conn: Any, scene_id: int, concept_id: str) -> Optional[Dict[str, Any]]:
        row = conn.execute(
            """SELECT p.*,c.status AS concept_status FROM scene_concept_projections p
            JOIN action_concepts c ON c.id=p.action_concept_id
            WHERE p.scene_id=? AND p.action_concept_id=? AND p.projection_version=?""",
            (scene_id, concept_id, PROJECTION_VERSION),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["semantic_frame"] = decode(result.pop("semantic_frame_json"), {})
        return result
