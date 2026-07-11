"""A small, provenance-preserving Cosmos of concepts and claims."""

from __future__ import annotations

import re
import hashlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .contracts import (
    AccessScope,
    Claim,
    ClaimScores,
    Concept,
    ImportReport,
    RetrievalResult,
    RetrievedClaim,
    TaskContract,
    new_id,
    utcnow,
)
from .database import SqliteDatabase, json_dumps, json_loads
from .hive import topic_terms
from .storage import ArtifactNotFound, ObjectStore


class Cosmos:
    """Canonical claims, not a vector index or a second copy of source text."""

    def __init__(self, database: SqliteDatabase, store: ObjectStore) -> None:
        self.database = database
        self.store = store

    def import_text(
        self,
        *,
        title: str,
        text: str,
        tenant_id: str,
        access_scope: Optional[AccessScope] = None,
        sectors: Optional[Sequence[str]] = None,
        trusted: bool = True,
    ) -> ImportReport:
        if not text.strip():
            raise ValueError("source text cannot be empty")
        scope = access_scope or AccessScope(tenant_id=tenant_id)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        duplicate = self.database.one(
            "SELECT * FROM sources WHERE tenant_id = ? AND content_hash = ? AND access_json = ? AND deleted_at IS NULL",
            (tenant_id, content_hash, json_dumps(scope)),
        )
        if duplicate:
            return ImportReport(
                source_id=duplicate["source_id"],
                artifact=self.store.get_metadata(duplicate["artifact_id"], tenant_id, project_id=scope.project_id),
                status=duplicate["status"],
                duplicate=True,
            )
        artifact = self.store.put_bytes(
            text.encode("utf-8"),
            tenant_id=tenant_id,
            media_type="text/markdown" if title.lower().endswith((".md", ".markdown")) else "text/plain",
            schema_name="SourceDocument",
            access_scope=scope,
        )
        source_id = new_id("src")
        now = utcnow()
        sentences = _sentences(text)
        concept_count = 0
        claim_count = 0
        sectors_value = list(dict.fromkeys(sectors or ["Unclassified"]))
        # Source archive and quarantine record precede interpretation. A source
        # which does not pass the local validator remains quarantined and is not
        # visible to normal retrieval.
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO sources(source_id, tenant_id, artifact_id, content_hash, title, access_json, status, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    tenant_id,
                    artifact.artifact_id,
                    artifact.content_hash,
                    title,
                    json_dumps(scope),
                    "quarantined",
                    now.isoformat(),
                ),
            )
            if trusted:
                seen_claims: set[tuple[str, str]] = set()
                for sentence in sentences:
                    terms = topic_terms(sentence)
                    if not terms:
                        continue
                    # The first content term is an explicit, inspectable
                    # subject candidate; no hidden semantic relation is claimed.
                    subject = self._upsert_concept(connection, tenant_id, terms[0], now)
                    concept_count += int(subject[1])
                    key = (subject[0], sentence.casefold())
                    if key in seen_claims:
                        continue
                    seen_claims.add(key)
                    claim = Claim(
                        tenant_id=tenant_id,
                        subject_id=subject[0],
                        predicate="states",
                        object_value=sentence,
                        source_id=source_id,
                        source_artifact_id=artifact.artifact_id,
                        source_fragment=sentence,
                        sectors=sectors_value,
                        access_scope=scope,
                        verification_status="reviewed",
                        scores=ClaimScores(confidence=0.6, relevance=0.5, freshness=1.0),
                    )
                    connection.execute(
                        "INSERT INTO claims(claim_id, tenant_id, subject_id, predicate, object_value, source_id, source_artifact_id, source_fragment, sector_json, access_json, verification_status, scores_json, valid_from, valid_to, created_at, deleted_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                        (
                            claim.claim_id,
                            tenant_id,
                            claim.subject_id,
                            claim.predicate,
                            claim.object_value,
                            source_id,
                            artifact.artifact_id,
                            claim.source_fragment,
                            json_dumps(claim.sectors),
                            json_dumps(scope),
                            claim.verification_status,
                            json_dumps(claim.scores),
                            None,
                            None,
                            claim.created_at.isoformat(),
                        ),
                    )
                    claim_count += 1
                connection.execute("UPDATE sources SET status = ? WHERE source_id = ?", ("integrated", source_id))
        return ImportReport(
            source_id=source_id,
            artifact=artifact,
            status="integrated" if trusted else "quarantined",
            imported_claims=claim_count,
            imported_concepts=concept_count,
        )

    def delete_source(self, source_id: str, tenant_id: str) -> int:
        row = self.database.one("SELECT * FROM sources WHERE source_id = ? AND tenant_id = ?", (source_id, tenant_id))
        if row is None:
            raise KeyError("source not found")
        now = utcnow().isoformat()
        with self.database.transaction() as connection:
            return self._revoke_source(connection, source_id, tenant_id, now, set())

    def _revoke_source(self, connection: Any, source_id: str, tenant_id: str, now: str, visited: set[str]) -> int:
        """Revoke claims, source bytes and all explicitly derived compost paths."""
        if source_id in visited:
            return 0
        visited.add(source_id)
        source = connection.execute(
            "SELECT * FROM sources WHERE source_id = ? AND tenant_id = ?", (source_id, tenant_id)
        ).fetchone()
        if source is None or source["deleted_at"] is not None:
            return 0
        connection.execute("UPDATE sources SET status = ?, deleted_at = ? WHERE source_id = ?", ("deleted", now, source_id))
        count = connection.execute(
            "UPDATE claims SET deleted_at = ? WHERE source_id = ? AND deleted_at IS NULL", (now, source_id)
        ).rowcount
        # Logical deletion immediately blocks exact archive reads; bytes remain
        # until the storage grace-period sweep can reclaim them.
        connection.execute(
            "UPDATE artifacts SET deleted_at = ? WHERE artifact_id = ? AND tenant_id = ?",
            (now, source["artifact_id"], tenant_id),
        )
        derived = connection.execute(
            "SELECT c.compost_id, c.artifact_id FROM composts c "
            "JOIN compost_dependencies d ON d.compost_id = c.compost_id "
            "WHERE c.tenant_id = ? AND d.source_artifact_id = ? AND c.status != 'deleted'",
            (tenant_id, source["artifact_id"]),
        ).fetchall()
        for dependent in derived:
            connection.execute("UPDATE composts SET status = 'deleted' WHERE compost_id = ?", (dependent["compost_id"],))
            connection.execute(
                "UPDATE artifacts SET deleted_at = ? WHERE artifact_id = ? AND tenant_id = ?",
                (now, dependent["artifact_id"], tenant_id),
            )
            integrations = connection.execute(
                "SELECT source_id FROM compost_integrations WHERE compost_id = ?", (dependent["compost_id"],)
            ).fetchall()
            for integration in integrations:
                count += self._revoke_source(connection, integration["source_id"], tenant_id, now, visited)
        return count

    def retrieve(
        self,
        contract: TaskContract,
        *,
        limit: int = 12,
        sector: Optional[str] = None,
        include_unverified: bool = False,
    ) -> RetrievalResult:
        limit = max(1, min(limit, contract.budget.event_limit, 100))
        terms = topic_terms(contract.goal)
        rows = self.database.all(
            "SELECT claims.*, concepts.label AS subject_label FROM claims "
            "JOIN concepts ON concepts.concept_id = claims.subject_id "
            "WHERE claims.deleted_at IS NULL AND claims.verification_status != 'rejected'"
        )
        candidates: list[tuple[float, Dict[str, Any], list[str]]] = []
        for row in rows:
            if not include_unverified and row["verification_status"] not in ("reviewed", "verified"):
                continue
            scope = AccessScope.model_validate(json_loads(row["access_json"]))
            if not self._allows(scope, contract.tenant_id, contract.project_id):
                continue
            sectors = json_loads(row["sector_json"], [])
            if sector and sector not in sectors:
                continue
            haystack = (row["subject_label"] + " " + row["object_value"]).lower()
            matches = [term for term in terms if term in haystack]
            if terms and not matches:
                continue
            scores = ClaimScores.model_validate(json_loads(row["scores_json"]))
            lexical = len(matches) / max(1, len(terms))
            score = lexical * 0.70 + scores.relevance * 0.15 + scores.confidence * 0.10 + scores.freshness * 0.05
            reasons = (["lexical:" + ",".join(matches)] if matches else ["empty-query"]) + ["status:" + row["verification_status"]]
            candidates.append((score, row, reasons))
        candidates.sort(key=lambda item: (-item[0], item[1]["created_at"]))
        # Diversity: avoid returning a long run of duplicate source fragments.
        selected: list[tuple[float, Dict[str, Any], list[str]]] = []
        sources: set[str] = set()
        for candidate in candidates:
            if len(selected) >= limit:
                break
            _, row, _ = candidate
            if row["source_id"] in sources and len(selected) >= max(2, limit // 2):
                continue
            selected.append(candidate)
            sources.add(row["source_id"])
        retrieved: list[RetrievedClaim] = []
        for score, row, reasons in selected:
            claim = self._row_to_claim(row)
            try:
                source = self.store.get_metadata(
                    claim.source_artifact_id,
                    contract.tenant_id,
                    project_id=contract.project_id,
                )
            except ArtifactNotFound:
                # A claim with a missing exact source is deliberately excluded.
                continue
            contradictions = self._contradictions(claim, contract.tenant_id, contract.project_id)
            retrieved.append(
                RetrievedClaim(
                    claim=claim,
                    subject_label=row["subject_label"],
                    source=source,
                    score=round(score, 4),
                    reasons=reasons,
                    contradictory_claim_ids=contradictions,
                )
            )
        gaps: list[str] = []
        if not retrieved:
            gaps.append("Нет разрешённых проверенных утверждений, соответствующих запросу.")
        return RetrievalResult(claims=retrieved, budget_used=len(retrieved), gaps=gaps, query_terms=terms)

    def list_concepts(
        self,
        *,
        tenant_id: str,
        project_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> list[Concept]:
        rows = self.database.all("SELECT * FROM concepts ORDER BY normalized_label LIMIT ?", (max(1, min(limit, 500)),))
        needle = _normalise(query or "")
        concepts: list[Concept] = []
        for row in rows:
            if row["tenant_id"] != tenant_id:
                # Cross-tenant global concepts are possible only when a global
                # source makes a claim visible; concepts themselves have no
                # authority without the claim, so keep the boundary strict.
                continue
            if needle and needle not in row["normalized_label"]:
                continue
            claims = self.database.all(
                "SELECT access_json FROM claims WHERE subject_id = ? AND deleted_at IS NULL", (row["concept_id"],)
            )
            if not any(
                self._allows(AccessScope.model_validate(json_loads(claim["access_json"])), tenant_id, project_id)
                for claim in claims
            ):
                continue
            concepts.append(
                Concept(
                    concept_id=row["concept_id"],
                    label=row["label"],
                    concept_type=row["concept_type"],
                    aliases=json_loads(row["aliases_json"], []),
                    tenant_id=row["tenant_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return concepts

    def list_claims(
        self,
        *,
        tenant_id: str,
        project_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[Claim]:
        result: list[Claim] = []
        for row in self.database.all("SELECT * FROM claims WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)):
            scope = AccessScope.model_validate(json_loads(row["access_json"]))
            if self._allows(scope, tenant_id, project_id):
                result.append(self._row_to_claim(row))
        return result

    def _upsert_concept(self, connection: Any, tenant_id: str, label: str, now: datetime) -> tuple[str, bool]:
        normalized = _normalise(label)
        row = connection.execute(
            "SELECT concept_id FROM concepts WHERE tenant_id = ? AND normalized_label = ?", (tenant_id, normalized)
        ).fetchone()
        if row:
            return row["concept_id"], False
        concept = Concept(label=label, tenant_id=tenant_id)
        connection.execute(
            "INSERT INTO concepts(concept_id, tenant_id, label, normalized_label, concept_type, aliases_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (concept.concept_id, tenant_id, label, normalized, concept.concept_type, json_dumps(concept.aliases), now.isoformat()),
        )
        return concept.concept_id, True

    def _contradictions(self, claim: Claim, tenant_id: str, project_id: Optional[str]) -> list[str]:
        rows = self.database.all(
            "SELECT claim_id, object_value, access_json FROM claims WHERE subject_id = ? AND predicate = ? AND deleted_at IS NULL AND claim_id != ?",
            (claim.subject_id, claim.predicate, claim.claim_id),
        )
        return [
            row["claim_id"]
            for row in rows
            if row["object_value"] != claim.object_value
            and self._allows(AccessScope.model_validate(json_loads(row["access_json"])), tenant_id, project_id)
        ]

    @staticmethod
    def _allows(scope: AccessScope, tenant_id: str, project_id: Optional[str]) -> bool:
        if scope.visibility == "global":
            return True
        if scope.tenant_id != tenant_id:
            return False
        return scope.visibility != "project" or scope.project_id == project_id

    @staticmethod
    def _row_to_claim(row: Dict[str, Any]) -> Claim:
        return Claim(
            claim_id=row["claim_id"],
            tenant_id=row["tenant_id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_value=row["object_value"],
            source_id=row["source_id"],
            source_artifact_id=row["source_artifact_id"],
            source_fragment=row["source_fragment"],
            sectors=json_loads(row["sector_json"], []),
            access_scope=AccessScope.model_validate(json_loads(row["access_json"])),
            verification_status=row["verification_status"],
            scores=ClaimScores.model_validate(json_loads(row["scores_json"])),
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )


def _normalise(value: str) -> str:
    return " ".join(topic_terms(value)).strip()


def _sentences(text: str) -> list[str]:
    # Markdown headings are retained as source structure but claims refer to
    # bounded sentences, so exact provenance stays inspectable in the archive.
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [re.sub(r"^\s*#+\s*", "", chunk).strip()[:2_000] for chunk in chunks if len(chunk.strip()) >= 12]
