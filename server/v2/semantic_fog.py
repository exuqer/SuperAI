"""Evidence-led semantic concept fogs.

The service intentionally learns only from scenes in the model.  It has no
lexical aliases or external vocabulary: a relation exists only after a typed
observation has been persisted.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Tuple

from .repository import V2Repository, encode, utcnow


class SemanticFogService:
    EXTRACTOR_VERSION = 4
    DEFINITION_WEIGHT = .90
    CATEGORY_WEIGHT = .70
    CONTEXT_WEIGHT = .65

    def __init__(self, repository: V2Repository) -> None:
        self.repository = repository

    def backfill(self, conn: Any, global_space_id: int) -> Dict[str, int]:
        """Idempotently rebuild evidence and affected fog projections."""
        versions = conn.execute(
            "SELECT DISTINCT semantic_extractor_version FROM semantic_backfill_state"
        ).fetchall()
        registered_fogs = int(conn.execute("SELECT COUNT(*) FROM concept_fog_registry").fetchone()[0])
        if registered_fogs and (
            not versions
            or any(int(row["semantic_extractor_version"]) != self.EXTRACTOR_VERSION for row in versions)
        ):
            # Evidence extractors are semantic migrations.  Fogs produced by
            # an older extractor must not remain searchable after their
            # evidence has been rebuilt.
            concept_ids = [
                int(row["concept_cloud_id"])
                for row in conn.execute("SELECT concept_cloud_id FROM concept_fog_registry")
            ]
            marks = ",".join("?" for _ in concept_ids)
            conn.execute(f"DELETE FROM clouds WHERE id IN ({marks})", concept_ids)

        scenes = conn.execute("SELECT cloud_id FROM scenes ORDER BY cloud_id").fetchall()
        affected: set[Tuple[int, int]] = set()
        for scene in scenes:
            affected.update(self._evidence_for_scene(conn, int(scene["cloud_id"])))
        fogs = 0
        for left, right in sorted(affected):
            fogs += self._materialize_pair(conn, left, right, global_space_id)
        # Definitions with a shared right-hand term form a category fog.  The
        # category itself is observed in the scene and is never guessed.
        category_rows = conn.execute(
            """SELECT
                CAST(json_extract(evidence_json,'$.definition_lexeme_id') AS INTEGER) AS category_id,
                GROUP_CONCAT(DISTINCT CAST(json_extract(evidence_json,'$.defined_lexeme_id') AS INTEGER)) AS members
            FROM semantic_evidence
            WHERE evidence_type='definition'
              AND json_extract(evidence_json,'$.definition_lexeme_id') IS NOT NULL
            GROUP BY category_id
            HAVING COUNT(DISTINCT json_extract(evidence_json,'$.defined_lexeme_id')) >= 2"""
        ).fetchall()
        for row in category_rows:
            members = [int(item) for item in str(row["members"]).split(",")]
            category_id = int(row["category_id"])
            for member in members:
                source = conn.execute(
                    """SELECT source_scene_cloud_id FROM semantic_evidence
                    WHERE evidence_type='definition'
                      AND CAST(json_extract(evidence_json,'$.defined_lexeme_id') AS INTEGER)=?
                      AND CAST(json_extract(evidence_json,'$.definition_lexeme_id') AS INTEGER)=?
                    ORDER BY source_scene_cloud_id LIMIT 1""", (member, category_id),
                ).fetchone()
                if source:
                    self._record(conn, int(source["source_scene_cloud_id"]), member, category_id, "shared_category", self.CATEGORY_WEIGHT, details={"category_lexeme_id": category_id})
            fogs += self._materialize_fog(conn, "shared_category", members + [category_id], global_space_id, .45, len(members))
        return {"scenes": len(scenes), "fogs": fogs}

    def _evidence_for_scene(self, conn: Any, scene_id: int) -> set[Tuple[int, int]]:
        rows = conn.execute(
            """SELECT sc.lexeme_cloud_id, sc.grammatical_role, sc.token_index, wf.normalized_form
            FROM scene_components sc JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id
            WHERE sc.scene_cloud_id=? AND sc.lexeme_cloud_id IS NOT NULL ORDER BY sc.token_index""",
            (scene_id,),
        ).fetchall()
        words = [dict(row) for row in rows]
        input_fingerprint = hashlib.sha256(encode(words).encode("utf-8")).hexdigest()
        state = conn.execute("SELECT semantic_extractor_version,input_fingerprint FROM semantic_backfill_state WHERE source_scene_cloud_id=?", (scene_id,)).fetchone()
        if state and int(state["semantic_extractor_version"]) == self.EXTRACTOR_VERSION and str(state["input_fingerprint"]) == input_fingerprint:
            existing = conn.execute("SELECT left_lexeme_cloud_id,right_lexeme_cloud_id FROM semantic_evidence WHERE source_scene_cloud_id=?", (scene_id,)).fetchall()
            return {tuple(sorted((int(row["left_lexeme_cloud_id"]), int(row["right_lexeme_cloud_id"])))) for row in existing}
        if state:
            conn.execute("DELETE FROM semantic_evidence WHERE source_scene_cloud_id=?", (scene_id,))
        affected: set[Tuple[int, int]] = set()
        definition_mark = next((item for item in words if item["normalized_form"] == "это"), None)
        if definition_mark:
            left = next((item for item in words if item["token_index"] < definition_mark["token_index"] and item["grammatical_role"] == "subject"), None)
            right = next((item for item in words if item["token_index"] > definition_mark["token_index"] and item["grammatical_role"] == "definition"), None)
            if left and right:
                defined_id = int(left["lexeme_cloud_id"])
                definition_id = int(right["lexeme_cloud_id"])
                self._record(
                    conn,
                    scene_id,
                    defined_id,
                    definition_id,
                    "definition",
                    self.DEFINITION_WEIGHT,
                    details={
                        "defined_lexeme_id": defined_id,
                        "definition_lexeme_id": definition_id,
                    },
                )
                affected.add(tuple(sorted((int(left["lexeme_cloud_id"]), int(right["lexeme_cloud_id"])))) )

        affected.update(self._contextual_evidence(conn, scene_id, words))
        evidence = conn.execute("SELECT evidence_key FROM semantic_evidence WHERE source_scene_cloud_id=? ORDER BY evidence_key", (scene_id,)).fetchall()
        result_fingerprint = hashlib.sha256("|".join(str(row["evidence_key"] or "") for row in evidence).encode("utf-8")).hexdigest()
        conn.execute(
            """INSERT INTO semantic_backfill_state(source_scene_cloud_id,semantic_extractor_version,input_fingerprint,result_fingerprint,processed_at)
            VALUES(?,?,?,?,?) ON CONFLICT(source_scene_cloud_id) DO UPDATE SET
            semantic_extractor_version=excluded.semantic_extractor_version,input_fingerprint=excluded.input_fingerprint,
            result_fingerprint=excluded.result_fingerprint,processed_at=excluded.processed_at""",
            (scene_id, self.EXTRACTOR_VERSION, input_fingerprint, result_fingerprint, utcnow()),
        )
        return affected

    def _contextual_evidence(
        self, conn: Any, scene_id: int, words: List[Dict[str, Any]],
    ) -> set[Tuple[int, int]]:
        """Learn distributional similarity from equal roles in equal contexts.

        Co-occurrence inside one sentence is deliberately insufficient: in
        ``кот ест рыбу`` the cat and fish are not synonyms.  Two lexemes gain
        one contextual observation only when they occupy the same role in two
        scenes that share a predicate and at least one other role anchor.
        """
        predicate = next(
            (item for item in words if item["grammatical_role"] == "predicate"), None
        )
        if not predicate:
            return set()
        comparable_roles = {
            "subject", "object", "location", "destination", "source", "instrument",
        }
        current = {
            str(item["grammatical_role"]): int(item["lexeme_cloud_id"])
            for item in words
            if item["grammatical_role"] in comparable_roles
        }
        if not current:
            return set()
        prior_rows = conn.execute(
            """SELECT sc.scene_cloud_id, sc.grammatical_role, sc.lexeme_cloud_id
            FROM scene_components sc
            WHERE sc.scene_cloud_id < ? AND sc.lexeme_cloud_id IS NOT NULL
              AND sc.scene_cloud_id IN (
                SELECT scene_cloud_id FROM scene_components
                WHERE grammatical_role='predicate' AND lexeme_cloud_id=?
              )
            ORDER BY sc.scene_cloud_id, sc.token_index""",
            (scene_id, int(predicate["lexeme_cloud_id"])),
        ).fetchall()
        prior_scenes: Dict[int, Dict[str, int]] = {}
        for row in prior_rows:
            role = str(row["grammatical_role"])
            if role in comparable_roles:
                prior_scenes.setdefault(int(row["scene_cloud_id"]), {})[role] = int(
                    row["lexeme_cloud_id"]
                )

        affected: set[Tuple[int, int]] = set()
        for prior_scene_id, prior in prior_scenes.items():
            for role, current_lexeme in current.items():
                prior_lexeme = prior.get(role)
                if not prior_lexeme or prior_lexeme == current_lexeme:
                    continue
                shared_roles = sorted(
                    anchor_role
                    for anchor_role, anchor_lexeme in current.items()
                    if anchor_role != role and prior.get(anchor_role) == anchor_lexeme
                )
                if not shared_roles:
                    continue
                self._record(
                    conn,
                    scene_id,
                    current_lexeme,
                    prior_lexeme,
                    "contextual_similarity",
                    .55,
                    independence=.8,
                    details={
                        "predicate_lexeme_id": int(predicate["lexeme_cloud_id"]),
                        "compared_scene_cloud_id": prior_scene_id,
                        "role": role,
                        "shared_roles": shared_roles,
                    },
                )
                affected.add(tuple(sorted((current_lexeme, prior_lexeme))))
        return affected

    def _record(self, conn: Any, scene_id: int | None, left: int, right: int, kind: str, evidence_weight: float, independence: float = 1.0, details: Dict[str, Any] | None = None) -> None:
        left, right = sorted((left, right))
        now = utcnow()
        source_key = str(scene_id) if scene_id is not None else str((details or {}).get("category_lexeme_id", "derived"))
        evidence_key = hashlib.sha256(f"{kind}:{source_key}:{left}:{right}:{encode(details or {})}".encode("utf-8")).hexdigest()
        conn.execute(
            """INSERT INTO semantic_evidence
            (source_scene_cloud_id,left_lexeme_cloud_id,right_lexeme_cloud_id,evidence_type,weight,evidence_weight,independence,evidence_key,evidence_json,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(source_scene_cloud_id,left_lexeme_cloud_id,right_lexeme_cloud_id,evidence_type)
            DO UPDATE SET weight=excluded.weight,evidence_weight=excluded.evidence_weight,independence=excluded.independence,
            evidence_key=excluded.evidence_key,evidence_json=excluded.evidence_json,updated_at=excluded.updated_at""",
            (scene_id, left, right, kind, evidence_weight, evidence_weight, independence, evidence_key, encode(details or {}), now, now),
        )

    def _materialize_pair(self, conn: Any, left: int, right: int, global_space_id: int) -> int:
        rows = conn.execute(
            """SELECT evidence_type, evidence_weight, independence, COUNT(*) AS count
            FROM semantic_evidence WHERE left_lexeme_cloud_id=? AND right_lexeme_cloud_id=?
            GROUP BY evidence_type,evidence_weight,independence""", (left, right)
        ).fetchall()
        for row in rows:
            kind, count = str(row["evidence_type"]), int(row["count"])
            if kind == "definition":
                return self._materialize_fog(conn, "definition", [left, right], global_space_id, .85, count)
            if kind == "contextual_similarity":
                context_rows = conn.execute("SELECT evidence_weight,independence FROM semantic_evidence WHERE left_lexeme_cloud_id=? AND right_lexeme_cloud_id=? AND evidence_type='contextual_similarity'", (left, right)).fetchall()
                combined_evidence = 1.0
                for evidence in context_rows:
                    combined_evidence *= 1.0 - float(evidence["evidence_weight"]) * float(evidence["independence"])
                combined_evidence = 1.0 - combined_evidence
                if len(context_rows) >= 2 and combined_evidence >= self.CONTEXT_WEIGHT:
                    return self._materialize_fog(conn, "contextual_similarity", [left, right], global_space_id, .65, len(context_rows))
        return 0

    def _materialize_fog(self, conn: Any, kind: str, members: Iterable[int], global_space_id: int, match_weight: float, evidence_count: int) -> int:
        member_ids = sorted(set(int(item) for item in members))
        labels = conn.execute(
            f"SELECT cloud_id, lemma FROM lexemes WHERE cloud_id IN ({','.join('?' for _ in member_ids)})", member_ids,
        ).fetchall()
        name = f"{kind}:{'|'.join(sorted(str(row['lemma']) for row in labels))}"
        concept, _ = self.repository.get_or_create_cloud(conn, "concept", name, mass=1.2, density=.8, stability=min(1.0, match_weight))
        space, _ = self.repository.get_or_create_space(conn, "concept_space", int(concept["id"]), global_space_id, seed=int(concept["id"]) * 7919)
        self.repository.ensure_global_placement(conn, concept, global_space_id)
        now = utcnow()
        conn.execute(
            """INSERT INTO concept_fog_registry(concept_cloud_id,concept_space_id,evidence_type,stability,evidence_count,metadata_json,updated_at)
            VALUES(?,?,?,?,?,?,?) ON CONFLICT(concept_cloud_id) DO UPDATE SET
            stability=excluded.stability,evidence_count=excluded.evidence_count,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at""",
            (concept["id"], space["id"], kind, match_weight, evidence_count, encode({"members": member_ids}), now),
        )
        for index, lexeme_id in enumerate(member_ids):
            conn.execute(
                """INSERT INTO semantic_memberships(lexeme_cloud_id,concept_cloud_id,weight,confidence,evidence_count,updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(lexeme_cloud_id,concept_cloud_id) DO UPDATE SET
                weight=excluded.weight,confidence=excluded.confidence,evidence_count=excluded.evidence_count,updated_at=excluded.updated_at""",
                (lexeme_id, concept["id"], match_weight, 1.0, evidence_count, now),
            )
            placement = conn.execute("SELECT 1 FROM cloud_placements WHERE cloud_id=? AND space_id=?", (lexeme_id, space["id"])).fetchone()
            if not placement:
                x, y = self.repository.stable_position(f"concept:{concept['id']}", index)
                self.repository.create_placement(conn, lexeme_id, int(space["id"]), x, y, metadata={"placement_kind": "concept_member", "concept_cloud_id": concept["id"]})
            candidate = conn.execute(
                """SELECT c.id FROM clouds c JOIN lexemes l ON l.lemma=c.canonical_name
                WHERE c.cloud_type='concept_candidate' AND l.cloud_id=? LIMIT 1""", (lexeme_id,)
            ).fetchone()
            if candidate and not conn.execute("SELECT 1 FROM cloud_placements WHERE cloud_id=? AND space_id=?", (candidate["id"], space["id"])).fetchone():
                x, y = self.repository.stable_position(f"concept:{concept['id']}:candidate", index)
                self.repository.create_placement(conn, int(candidate["id"]), int(space["id"]), x, y, metadata={"placement_kind": "concept_candidate", "concept_cloud_id": concept["id"]})
            if candidate:
                conn.execute(
                    """INSERT INTO concept_candidate_registry(concept_candidate_cloud_id,status,stability_score,is_search_eligible,metadata_json,updated_at)
                    VALUES(?,'candidate',?,0,?,?) ON CONFLICT(concept_candidate_cloud_id) DO UPDATE SET
                    stability_score=excluded.stability_score,is_search_eligible=0,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at""",
                    (candidate["id"], min(.64, match_weight), encode({"concept_cloud_id": concept["id"], "status": "candidate"}), now),
                )
        # Deterministic constrained relaxation is limited to this fog; global
        # and hive placements are never touched.
        for index, lexeme_id in enumerate(member_ids):
            x, y = self.repository.stable_position(f"concept:{concept['id']}:relaxed", index)
            conn.execute("UPDATE cloud_placements SET x=?,y=?,updated_at=? WHERE cloud_id=? AND space_id=?", (x, y, now, lexeme_id, space["id"]))
        conn.execute("UPDATE lexemes SET semantic_state='stable' WHERE cloud_id IN (%s)" % ",".join("?" for _ in member_ids), member_ids)
        return 1
