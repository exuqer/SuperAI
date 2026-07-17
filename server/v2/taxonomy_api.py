from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .concept_relations import ConceptRelationTrainer
from .repository import V2Repository
from .taxonomy_resolver import TaxonomyResolver


router = APIRouter(prefix="/api", tags=["taxonomy"])


def _lexeme_id(conn, lemma: str) -> int:
    row = conn.execute("SELECT cloud_id FROM lexemes WHERE lemma=? ORDER BY cloud_id LIMIT 1", (lemma.casefold(),)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"lexeme not found: {lemma}")
    return int(row["cloud_id"])


@router.post("/concept-relations/rebuild")
async def rebuild_concept_relations() -> dict:
    with V2Repository().transaction() as conn:
        return ConceptRelationTrainer().rebuild(conn)


@router.get("/concept-relations")
async def list_concept_relations(relation_type: str | None = None) -> dict:
    with V2Repository().transaction() as conn:
        where, params = ("WHERE cr.relation_type=?", (relation_type,)) if relation_type else ("", ())
        rows = conn.execute(
            f"""SELECT cr.*, s.lemma AS subject, o.lemma AS object
                FROM concept_relations cr
                JOIN lexemes s ON s.cloud_id=cr.subject_lexeme_cloud_id
                JOIN lexemes o ON o.cloud_id=cr.object_lexeme_cloud_id {where}
                ORDER BY cr.relation_type, s.lemma, o.lemma""", params
        ).fetchall()
        return {"items": [dict(row) for row in rows]}


@router.get("/concept-relations/{relation_id}")
async def concept_relation(relation_id: str) -> dict:
    with V2Repository().transaction() as conn:
        row = conn.execute("SELECT * FROM concept_relations WHERE id=?", (relation_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="concept relation not found")
        result = dict(row)
        evidence = conn.execute(
            "SELECT * FROM concept_relation_evidence WHERE concept_relation_id=? ORDER BY created_at", (relation_id,)
        ).fetchall()
        result["evidence"] = [dict(item) for item in evidence]
        return result


@router.get("/taxonomy/path")
async def taxonomy_path(subject: str = Query(...), type: str = Query(...), max_depth: int = Query(3, ge=1, le=3)) -> dict:
    with V2Repository().transaction() as conn:
        return TaxonomyResolver(conn).resolve_is_a(_lexeme_id(conn, subject), _lexeme_id(conn, type), max_depth)


@router.get("/taxonomy/children")
async def taxonomy_children(parent: str = Query(...)) -> dict:
    with V2Repository().transaction() as conn:
        parent_id = _lexeme_id(conn, parent)
        rows = conn.execute(
            """SELECT cr.id, cr.confidence, l.lemma FROM concept_relations cr
               JOIN lexemes l ON l.cloud_id=cr.subject_lexeme_cloud_id
               WHERE cr.relation_type='IS_A' AND cr.object_lexeme_cloud_id=? ORDER BY l.lemma""", (parent_id,)
        ).fetchall()
        return {"parent": parent, "children": [dict(row) for row in rows]}


@router.get("/taxonomy/parents")
async def taxonomy_parents(subject: str = Query(...)) -> dict:
    with V2Repository().transaction() as conn:
        subject_id = _lexeme_id(conn, subject)
        rows = conn.execute(
            """SELECT cr.id, cr.confidence, l.lemma FROM concept_relations cr
               JOIN lexemes l ON l.cloud_id=cr.object_lexeme_cloud_id
               WHERE cr.relation_type='IS_A' AND cr.subject_lexeme_cloud_id=? ORDER BY l.lemma""", (subject_id,)
        ).fetchall()
        return {"subject": subject, "parents": [dict(row) for row in rows]}
