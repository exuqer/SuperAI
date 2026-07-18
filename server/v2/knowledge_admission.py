"""Staging, admission, dependency tracking and reversible knowledge batches."""

from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from server.tokenizer import normalize_text

from .dialogue_state import DialogueStateService
from .language.models import (
    Actuality,
    ClauseMode,
    LanguageAnalysis,
)
from .repository import decode, encode, utcnow


def _stable_id(prefix: str, *parts: object) -> str:
    key = "|".join(str(part) for part in parts)
    return f"{prefix}-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:20]}"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class AdmissionDecision:
    id: str
    staging_id: str
    decision: str
    structural_valid: bool
    factuality_valid: bool
    source_valid: bool
    independent_source_count: int
    reasons: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    materialized_objects: List[Dict[str, Any]] = field(default_factory=list)
    parser_version: str = "dialogue-v2.5"
    created_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "staging_id": self.staging_id,
            "decision": self.decision,
            "structural_valid": self.structural_valid,
            "factuality_valid": self.factuality_valid,
            "source_valid": self.source_valid,
            "independent_source_count": self.independent_source_count,
            "reasons": list(self.reasons),
            "evidence": list(self.evidence),
            "materialized_objects": list(self.materialized_objects),
            "parser_version": self.parser_version,
            "created_at": self.created_at,
        }


class KnowledgeAdmissionService:
    def __init__(
        self,
        repository: Any,
        analyzer: Any = None,
    ) -> None:
        self.repository = repository
        self.analyzer = analyzer

    def _analysis(
        self,
        text: str,
        *,
        conversation_id: str = "",
        speaker_role: str = "source",
        source_type: str = "knowledge_staging",
    ) -> LanguageAnalysis:
        if self.analyzer is None:
            from .language.analyzer import UniversalLanguageAnalyzer
            from .training import RussianMorphology
            self.analyzer = UniversalLanguageAnalyzer(RussianMorphology())
        return self.analyzer.analyze(
            text,
            conversation_id=conversation_id,
            speaker_role=speaker_role,
            source_type=source_type,
        )

    @staticmethod
    def _source_key(
        source_type: str,
        source_key: str,
        conversation_id: str,
        speaker_role: str,
    ) -> str:
        explicit = str(source_key or "").strip()
        if explicit:
            return explicit
        if conversation_id:
            return f"dialogue:{conversation_id}:{speaker_role or 'unknown'}"
        return f"{source_type}:anonymous"

    def stage(
        self,
        text: str,
        *,
        source_type: str = "training",
        source_key: str = "",
        conversation_id: str = "",
        speaker_role: str = "",
        analysis: Optional[LanguageAnalysis] = None,
        supersedes_staging_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized = normalize_text(text).casefold()
        source_hash = _hash(normalized)
        independent_source_key = self._source_key(
            source_type,
            source_key,
            conversation_id,
            speaker_role,
        )
        analysis = analysis or self._analysis(
            text,
            conversation_id=conversation_id or independent_source_key,
            speaker_role=speaker_role or "source",
            source_type=source_type,
        )
        now = utcnow()
        staging_id = _stable_id(
            "knowledge-stage",
            source_hash,
            independent_source_key,
            analysis.interpretation_version,
            supersedes_staging_id or "",
            now if supersedes_staging_id else "",
        )
        with self.repository.transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM knowledge_staging WHERE id=?",
                (staging_id,),
            ).fetchone()
            created = existing is None
            if not existing:
                conn.execute(
                    """INSERT INTO knowledge_staging
                       (id,source_type,source_key,raw_text,normalized_text,
                        source_hash,independent_source_key,conversation_id,
                        speaker_role,parser_version,interpretation_status,
                        interpretation_json,validation_json,status,
                        supersedes_staging_id,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'STAGED',?,?,?)""",
                    (
                        staging_id,
                        source_type,
                        source_key or None,
                        text,
                        normalized,
                        source_hash,
                        independent_source_key,
                        conversation_id or None,
                        speaker_role or None,
                        analysis.interpretation_version,
                        analysis.interpretation_status.value,
                        encode(analysis.as_dict()),
                        "{}",
                        supersedes_staging_id,
                        now,
                        now,
                    ),
                )
            DialogueStateService(
                self.repository
            ).persist_interpretation(conn, analysis)
            row = conn.execute(
                "SELECT * FROM knowledge_staging WHERE id=?",
                (staging_id,),
            ).fetchone()
            if created:
                self._record_language_patterns(
                    conn,
                    analysis,
                    independent_source_key,
                )
            return self._row(row)

    @staticmethod
    def _record_language_patterns(
        conn: Any,
        analysis: LanguageAnalysis,
        independent_source_key: str,
    ) -> None:
        patterns: List[tuple[str, str]] = []
        for clause in analysis.clauses:
            predicates = ",".join(
                str(item.get("lemma") or "")
                for item in clause.predicate_hypotheses
            )
            patterns.append((
                "clause_construction",
                "|".join([
                    clause.mode.value,
                    clause.actuality.value,
                    clause.polarity.value,
                    str(clause.modality.value if clause.modality else ""),
                    predicates,
                ]),
            ))
        for act in analysis.dialogue_acts:
            patterns.append((
                "dialogue_act",
                f"{act.act_type.value}:{act.token_end - act.token_start + 1}",
            ))
        now = utcnow()
        for pattern_type, signature in patterns:
            pattern_id = _stable_id(
                "language-pattern",
                pattern_type,
                signature,
                analysis.interpretation_version,
            )
            row = conn.execute(
                "SELECT * FROM language_patterns WHERE id=?",
                (pattern_id,),
            ).fetchone()
            evidence = (
                decode(row["evidence_json"], []) if row else []
            )
            observation = {
                "source": independent_source_key,
                "utterance_id": (
                    analysis.utterance.id if analysis.utterance else None
                ),
                "parser_version": analysis.interpretation_version,
            }
            observation_key = (
                observation["source"],
                observation["utterance_id"],
                observation["parser_version"],
            )
            if observation_key not in {
                (
                    item.get("source"),
                    item.get("utterance_id"),
                    item.get("parser_version"),
                )
                for item in evidence
            }:
                evidence.append(observation)
            independent_count = len({
                item.get("source") for item in evidence if item.get("source")
            })
            observation_count = len(evidence)
            status = (
                "STABLE"
                if independent_count >= 5
                else "PROBABLE"
                if independent_count >= 3
                else "CANDIDATE"
                if independent_count >= 2
                else "OBSERVED"
            )
            conn.execute(
                """INSERT INTO language_patterns
                   (id,pattern_type,signature,status,observation_count,
                    independent_source_count,evidence_json,parser_version,
                    created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     status=excluded.status,
                     observation_count=excluded.observation_count,
                     independent_source_count=excluded.independent_source_count,
                     evidence_json=excluded.evidence_json,
                     updated_at=excluded.updated_at""",
                (
                    pattern_id,
                    pattern_type,
                    signature,
                    status,
                    observation_count,
                    independent_count,
                    encode(evidence),
                    analysis.interpretation_version,
                    now,
                    now,
                ),
            )

    @staticmethod
    def _row(row: Any) -> Dict[str, Any]:
        item = dict(row)
        for key in list(item):
            if key.endswith("_json"):
                item[key[:-5]] = decode(item.pop(key), {})
        return item

    @staticmethod
    def _decision_from_row(row: Any) -> AdmissionDecision:
        item = dict(row)
        return AdmissionDecision(
            id=str(item["id"]),
            staging_id=str(item["staging_id"]),
            decision=str(item["decision"]),
            structural_valid=bool(item["structural_valid"]),
            factuality_valid=bool(item["factuality_valid"]),
            source_valid=bool(item["source_valid"]),
            independent_source_count=int(
                item["independent_source_count"]
            ),
            reasons=decode(item["reasons_json"], []),
            evidence=decode(item["evidence_json"], []),
            materialized_objects=decode(
                item["materialized_objects_json"],
                [],
            ),
            parser_version=str(item["parser_version"]),
            created_at=str(item["created_at"]),
        )

    def _retract_scene_derivations(self, conn: Any, scene_id: int) -> None:
        concept_ids = [
            str(row["concept_id"])
            for row in conn.execute(
                """SELECT DISTINCT concept_id FROM concept_evidence
                   WHERE source_scene_id=?""",
                (scene_id,),
            ).fetchall()
        ]
        relation_ids = [
            str(row["concept_relation_id"])
            for row in conn.execute(
                """SELECT DISTINCT concept_relation_id
                   FROM concept_relation_evidence
                   WHERE source_scene_id=?
                     AND concept_relation_id IS NOT NULL""",
                (scene_id,),
            ).fetchall()
        ]
        conn.execute(
            """UPDATE concept_evidence SET status='RETRACTED'
               WHERE source_scene_id=?""",
            (scene_id,),
        )
        conn.execute(
            """UPDATE concept_relation_evidence SET status='RETRACTED'
               WHERE source_scene_id=?""",
            (scene_id,),
        )
        conn.execute(
            "DELETE FROM scene_concept_projections WHERE scene_id=?",
            (scene_id,),
        )
        now = utcnow()
        for concept_id in concept_ids:
            evidence = conn.execute(
                """SELECT evidence_type,confidence
                   FROM concept_evidence
                   WHERE concept_id=? AND status='ACTIVE'""",
                (concept_id,),
            ).fetchall()
            count = len(evidence)
            conflicted = any(
                row["evidence_type"] == "EXPLICIT_OPPOSITION"
                for row in evidence
            )
            status = (
                "DEPRECATED"
                if not count
                else "CONFLICTED"
                if conflicted
                else "STABLE"
                if count >= 3
                else "PROBABLE"
            )
            confidence = max(
                (float(row["confidence"]) for row in evidence),
                default=0.0,
            )
            conn.execute(
                """UPDATE concepts
                   SET status=?,confidence=?,evidence_count=?,updated_at=?
                   WHERE id=?""",
                (status, confidence, count, now, concept_id),
            )
            conn.execute(
                """UPDATE action_concepts
                   SET status=?,confidence=?,evidence_count=?,updated_at=?
                   WHERE id=?""",
                (
                    "DEPRECATED" if not count else status,
                    confidence,
                    count,
                    now,
                    concept_id,
                ),
            )
            conn.execute(
                """UPDATE action_variants
                   SET weight=?,evidence_count=?,updated_at=?
                   WHERE action_concept_id=?""",
                (confidence, count, now, concept_id),
            )
        for relation_id in relation_ids:
            active_count = int(conn.execute(
                """SELECT COUNT(*) FROM concept_relation_evidence
                   WHERE concept_relation_id=? AND status<>'RETRACTED'""",
                (relation_id,),
            ).fetchone()[0])
            conn.execute(
                """UPDATE concept_relations
                   SET status=?,evidence_count=?,updated_at=?
                   WHERE id=?""",
                (
                    "STABLE" if active_count else "DEPRECATED",
                    active_count,
                    now,
                    relation_id,
                ),
            )
        global_space = conn.execute(
            """SELECT id FROM spaces
               WHERE space_type='global_field' LIMIT 1"""
        ).fetchone()
        if global_space:
            from .semantic_fog import SemanticFogService

            SemanticFogService(self.repository).backfill(
                conn,
                int(global_space["id"]),
            )

    @staticmethod
    def _structural_validation(
        interpretation: Mapping[str, Any],
    ) -> tuple[bool, List[str]]:
        clauses = interpretation.get("clauses") or []
        acts = interpretation.get("dialogue_acts") or []
        reasons: List[str] = []
        if not clauses and acts:
            non_content = {
                "GREETING",
                "SMALL_TALK",
                "CONFIRMATION",
                "DENIAL",
            }
            if any(act.get("act_type") not in non_content for act in acts):
                reasons.append("content act has no clause")
        for clause in clauses:
            if clause.get("mode") in {
                "ASSERTION",
                "DEFINITION",
                "REPORTED_SPEECH",
            } and not clause.get("predicate_hypotheses"):
                reasons.append(
                    f"clause {clause.get('id')} has no predicate hypothesis"
                )
        status = interpretation.get("interpretation_status")
        if status in {"AMBIGUOUS", "INCOMPLETE", "CONFLICTED"}:
            reasons.append(f"interpretation status is {status}")
        return not reasons, reasons

    @staticmethod
    def _factuality_validation(
        interpretation: Mapping[str, Any],
    ) -> tuple[bool, List[str]]:
        reasons: List[str] = []
        factual_clauses = 0
        for clause in interpretation.get("clauses") or []:
            mode = clause.get("mode")
            actuality = clause.get("actuality")
            evidence_status = clause.get("evidence_status")
            if mode in {"QUESTION", "REQUEST", "COMMAND"}:
                continue
            if actuality != Actuality.ACTUAL.value:
                reasons.append(
                    f"clause {clause.get('id')} is {actuality}"
                )
                continue
            if mode in {
                ClauseMode.HYPOTHESIS.value,
                ClauseMode.ASSUMPTION.value,
                ClauseMode.CONDITION.value,
                ClauseMode.COUNTERFACTUAL.value,
                ClauseMode.DESIRE.value,
                ClauseMode.PLAN.value,
                ClauseMode.QUOTE.value,
            }:
                reasons.append(
                    f"clause {clause.get('id')} mode {mode} is not world fact"
                )
                continue
            if evidence_status in {"DISPUTED", "REJECTED"}:
                reasons.append(
                    f"clause {clause.get('id')} evidence is {evidence_status}"
                )
                continue
            factual_clauses += 1
        if not factual_clauses:
            reasons.append("no admissible actual clause")
        return factual_clauses > 0 and not reasons, reasons

    @staticmethod
    def _source_validation(
        row: Mapping[str, Any],
    ) -> tuple[bool, List[str]]:
        reasons: List[str] = []
        if not row.get("raw_text"):
            reasons.append("raw source is empty")
        if not row.get("independent_source_key"):
            reasons.append("independent source key is missing")
        if row.get("source_type") in {"assistant_answer", "derived_answer"}:
            reasons.append("derived assistant answer cannot become independent fact")
        return not reasons, reasons

    def decide(
        self,
        staging_id: str,
        *,
        manual_validation: bool = False,
    ) -> AdmissionDecision:
        with self.repository.transaction() as conn:
            row_value = conn.execute(
                "SELECT * FROM knowledge_staging WHERE id=?",
                (staging_id,),
            ).fetchone()
            if not row_value:
                raise KeyError(staging_id)
            row = self._row(row_value)
            interpretation = row.get("interpretation") or {}
            structural, structural_reasons = self._structural_validation(
                interpretation
            )
            factuality, factuality_reasons = self._factuality_validation(
                interpretation
            )
            source, source_reasons = self._source_validation(row)
            duplicate_rows = conn.execute(
                """SELECT DISTINCT independent_source_key
                   FROM knowledge_staging
                   WHERE source_hash=? AND status IN
                     ('STAGED','ADMITTED','COMMITTED')""",
                (row["source_hash"],),
            ).fetchall()
            independent_count = len({
                str(item["independent_source_key"]) for item in duplicate_rows
            })
            reasons = [
                *structural_reasons,
                *factuality_reasons,
                *source_reasons,
            ]
            if not structural:
                decision_value = "QUARANTINE"
            elif not factuality:
                decision_value = "QUARANTINE"
            elif not source:
                decision_value = "REJECT"
            else:
                decision_value = "ADMIT"
            decision = AdmissionDecision(
                id=_stable_id(
                    "admission",
                    staging_id,
                    decision_value,
                    row["parser_version"],
                ),
                staging_id=staging_id,
                decision=decision_value,
                structural_valid=structural,
                factuality_valid=factuality,
                source_valid=source,
                independent_source_count=independent_count,
                reasons=reasons,
                evidence=[
                    {
                        "group": "manual_validation"
                        if manual_validation else "source",
                        "support": 1.0 if manual_validation else 0.7,
                    }
                ],
                parser_version=row["parser_version"],
                created_at=utcnow(),
            )
            conn.execute(
                """INSERT INTO knowledge_admission_decisions
                   (id,staging_id,decision,structural_valid,factuality_valid,
                    source_valid,independent_source_count,reasons_json,
                    evidence_json,materialized_objects_json,parser_version,
                    created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     structural_valid=excluded.structural_valid,
                     factuality_valid=excluded.factuality_valid,
                     source_valid=excluded.source_valid,
                     independent_source_count=excluded.independent_source_count,
                     reasons_json=excluded.reasons_json,
                     evidence_json=excluded.evidence_json,
                     parser_version=excluded.parser_version""",
                (
                    decision.id,
                    staging_id,
                    decision.decision,
                    int(decision.structural_valid),
                    int(decision.factuality_valid),
                    int(decision.source_valid),
                    decision.independent_source_count,
                    encode(decision.reasons),
                    encode(decision.evidence),
                    "[]",
                    decision.parser_version,
                    decision.created_at,
                ),
            )
            conn.execute(
                """UPDATE knowledge_staging
                   SET validation_json=?,status=?,updated_at=? WHERE id=?""",
                (
                    encode(decision.as_dict()),
                    "ADMITTED"
                    if decision_value == "ADMIT"
                    else "QUARANTINED"
                    if decision_value == "QUARANTINE"
                    else "REJECTED",
                    utcnow(),
                    staging_id,
                ),
            )
            return decision

    def commit(
        self,
        staging_id: str,
        *,
        materializer: Optional[
            Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
        ] = None,
        manual_validation: bool = False,
    ) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            staging = conn.execute(
                "SELECT status FROM knowledge_staging WHERE id=?",
                (staging_id,),
            ).fetchone()
            if not staging:
                raise KeyError(staging_id)
            if staging["status"] in {"RETRACTED", "SUPERSEDED"}:
                raise ValueError(
                    f"cannot commit staging item in "
                    f"{staging['status']} status"
                )
            if staging["status"] == "COMMITTED":
                existing = conn.execute(
                    """SELECT * FROM knowledge_admission_decisions
                       WHERE staging_id=? AND decision='ADMIT'
                       ORDER BY created_at DESC,id DESC LIMIT 1""",
                    (staging_id,),
                ).fetchone()
                if not existing:
                    raise RuntimeError(
                        "committed staging item has no admission decision"
                    )
                result = self._decision_from_row(existing).as_dict()
                result["status"] = "COMMITTED"
                return result
        decision = self.decide(
            staging_id,
            manual_validation=manual_validation,
        )
        if decision.decision != "ADMIT":
            return decision.as_dict()
        with self.repository.transaction() as conn:
            row_value = conn.execute(
                "SELECT * FROM knowledge_staging WHERE id=?",
                (staging_id,),
            ).fetchone()
            row = self._row(row_value)
        materialized = [
            deepcopy(dict(item))
            for item in (materializer(row) if materializer else [])
        ]
        with self.repository.transaction() as conn:
            conn.execute(
                """UPDATE knowledge_admission_decisions
                   SET materialized_objects_json=? WHERE id=?""",
                (encode(materialized), decision.id),
            )
            conn.execute(
                """UPDATE knowledge_staging SET status='COMMITTED',updated_at=?
                   WHERE id=?""",
                (utcnow(), staging_id),
            )
            for item in materialized:
                dependent_type = str(item.get("type") or "object")
                dependent_id = str(item.get("id") or item.get("scene_cloud_id") or "")
                if not dependent_id:
                    continue
                dependency_id = _stable_id(
                    "dependency",
                    "knowledge_staging",
                    staging_id,
                    dependent_type,
                    dependent_id,
                )
                conn.execute(
                    """INSERT OR REPLACE INTO knowledge_dependencies
                       (id,source_type,source_id,dependent_type,dependent_id,
                        dependency_type,status,metadata_json,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,'ACTIVE',?,?,?)""",
                    (
                        dependency_id,
                        "knowledge_staging",
                        staging_id,
                        dependent_type,
                        dependent_id,
                        "MATERIALIZES",
                        encode({
                            "admission_decision_id": decision.id,
                            "created": bool(item.get("created")),
                        }),
                        utcnow(),
                        utcnow(),
                    ),
                )
                if dependent_type == "scene" and dependent_id.isdigit():
                    source_interpretation_id = (
                        item.get("source_interpretation_id")
                        or next(
                            (
                                hypothesis.get("id")
                                for hypothesis in (
                                    row.get("interpretation") or {}
                                ).get("interpretation_hypotheses", [])
                                if hypothesis.get("selected")
                                and hypothesis.get("hypothesis_type")
                                == "predicate"
                            ),
                            None,
                        )
                    )
                    conn.execute(
                        """UPDATE scenes
                           SET admission_decision_id=?,
                               source_interpretation_id=COALESCE(
                                 ?,source_interpretation_id),
                               knowledge_status='CONFIRMED'
                           WHERE cloud_id=?""",
                        (
                            decision.id,
                            source_interpretation_id,
                            int(dependent_id),
                        ),
                    )
                    conn.execute(
                        """UPDATE events SET admission_decision_id=?
                           WHERE source_scene_id=?""",
                        (decision.id, int(dependent_id)),
                    )
            result = decision.as_dict()
            result["materialized_objects"] = materialized
            result["status"] = "COMMITTED"
            return result

    def retract(
        self,
        target_id: str,
        *,
        target_type: str = "knowledge_staging",
        reason: str = "",
        operation: str = "RETRACT_EVIDENCE",
    ) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            dependencies = [
                dict(row) for row in conn.execute(
                    """SELECT * FROM knowledge_dependencies
                       WHERE source_type=? AND source_id=? AND status='ACTIVE'""",
                    (target_type, target_id),
                )
            ]
            previous_status = None
            if target_type == "knowledge_staging":
                row = conn.execute(
                    "SELECT status FROM knowledge_staging WHERE id=?",
                    (target_id,),
                ).fetchone()
                if not row:
                    raise KeyError(target_id)
                previous_status = str(row["status"])
                conn.execute(
                    """UPDATE knowledge_staging
                       SET status='RETRACTED',updated_at=? WHERE id=?""",
                    (utcnow(), target_id),
                )
            conn.execute(
                """UPDATE knowledge_dependencies SET status='RETRACTED',
                   updated_at=? WHERE source_type=? AND source_id=?""",
                (utcnow(), target_type, target_id),
            )
            for dependency in dependencies:
                if (
                    dependency["dependent_type"] == "scene"
                    and str(dependency["dependent_id"]).isdigit()
                ):
                    scene_id = int(dependency["dependent_id"])
                    active = conn.execute(
                        """SELECT metadata_json FROM knowledge_dependencies
                           WHERE dependent_type='scene' AND dependent_id=?
                             AND status='ACTIVE'
                           ORDER BY updated_at DESC LIMIT 1""",
                        (str(scene_id),),
                    ).fetchone()
                    if active:
                        active_metadata = decode(
                            active["metadata_json"],
                            {},
                        )
                        conn.execute(
                            """UPDATE scenes
                               SET admission_decision_id=?,
                                   knowledge_status='CONFIRMED'
                               WHERE cloud_id=?""",
                            (
                                active_metadata.get(
                                    "admission_decision_id"
                                ),
                                scene_id,
                            ),
                        )
                    else:
                        all_sources = conn.execute(
                            """SELECT metadata_json
                               FROM knowledge_dependencies
                               WHERE dependent_type='scene'
                                 AND dependent_id=?""",
                            (str(scene_id),),
                        ).fetchall()
                        created_by_admission = any(
                            bool(
                                decode(
                                    item["metadata_json"],
                                    {},
                                ).get("created")
                            )
                            for item in all_sources
                        )
                        knowledge_status = (
                            "RETRACTED"
                            if created_by_admission
                            else "LEGACY_CONFIRMED"
                        )
                        conn.execute(
                            """UPDATE scenes
                               SET admission_decision_id=NULL,
                                   knowledge_status=?
                               WHERE cloud_id=?""",
                            (knowledge_status, scene_id),
                        )
                        if knowledge_status == "RETRACTED":
                            self._retract_scene_derivations(
                                conn,
                                scene_id,
                            )
                            placement_ids = [
                                int(item["hive_placement_id"])
                                for item in conn.execute(
                                    """SELECT hive_placement_id
                                       FROM hive_cells
                                       WHERE source_scene_cloud_id=?""",
                                    (scene_id,),
                                ).fetchall()
                            ]
                            conn.execute(
                                """DELETE FROM hive_cells
                                   WHERE source_scene_cloud_id=?""",
                                (scene_id,),
                            )
                            for placement_id in placement_ids:
                                conn.execute(
                                    """DELETE FROM cloud_placements
                                       WHERE id=?""",
                                    (placement_id,),
                                )
                elif (
                    dependency["dependent_type"]
                    == "clause_event_projection"
                ):
                    conn.execute(
                        """UPDATE clause_event_projections
                           SET status='RETRACTED',updated_at=?
                           WHERE id=?""",
                        (
                            utcnow(),
                            str(dependency["dependent_id"]),
                        ),
                    )
            retraction_id = _stable_id(
                "retraction",
                target_type,
                target_id,
                operation,
                utcnow(),
            )
            payload = {
                "dependencies": dependencies,
                "recalculated_dependents": [
                    {
                        "type": item["dependent_type"],
                        "id": item["dependent_id"],
                        "status": "REQUIRES_RECALCULATION",
                    }
                    for item in dependencies
                ],
            }
            conn.execute(
                """INSERT INTO knowledge_retractions
                   (id,target_type,target_id,operation,reason,previous_status,
                    new_status,payload_json,created_at)
                   VALUES(?,?,?,?,?,?, 'RETRACTED',?,?)""",
                (
                    retraction_id,
                    target_type,
                    target_id,
                    operation,
                    reason or operation,
                    previous_status,
                    encode(payload),
                    utcnow(),
                ),
            )
            return {
                "id": retraction_id,
                "target_type": target_type,
                "target_id": target_id,
                "operation": operation,
                "previous_status": previous_status,
                "new_status": "RETRACTED",
                **payload,
            }

    def reprocess(
        self,
        staging_id: str,
        *,
        analyzer: Any = None,
    ) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            row_value = conn.execute(
                "SELECT * FROM knowledge_staging WHERE id=?",
                (staging_id,),
            ).fetchone()
            if not row_value:
                raise KeyError(staging_id)
            row = self._row(row_value)
        if analyzer is not None:
            self.analyzer = analyzer
        analysis = self._analysis(
            row["raw_text"],
            conversation_id=(
                row.get("conversation_id")
                or row.get("independent_source_key")
                or ""
            ),
            speaker_role=row.get("speaker_role") or "source",
            source_type=row.get("source_type") or "knowledge_staging",
        )
        new_stage = self.stage(
            row["raw_text"],
            source_type=row["source_type"],
            source_key=row.get("source_key") or "",
            conversation_id=row.get("conversation_id") or "",
            speaker_role=row.get("speaker_role") or "",
            analysis=analysis,
            supersedes_staging_id=staging_id,
        )
        with self.repository.transaction() as conn:
            conn.execute(
                """UPDATE knowledge_staging
                   SET status='SUPERSEDED',updated_at=? WHERE id=? AND id<>?""",
                (utcnow(), staging_id, new_stage["id"]),
            )
        return {
            "previous_staging_id": staging_id,
            "staging": new_stage,
            "history_preserved": True,
        }

    def preview_batch(
        self,
        sources: Sequence[Mapping[str, Any]],
        *,
        config: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        batch_id = _stable_id(
            "knowledge-batch",
            json.dumps(list(sources), ensure_ascii=False, sort_keys=True),
            utcnow(),
        )
        staged = [
            self.stage(
                str(source.get("text") or ""),
                source_type=str(source.get("source_type") or "batch"),
                source_key=str(source.get("source_key") or ""),
                conversation_id=str(source.get("conversation_id") or ""),
                speaker_role=str(source.get("speaker_role") or ""),
            )
            for source in sources
        ]
        metrics = {
            "staged": len(staged),
            "stable": sum(
                item.get("interpretation_status") == "STABLE"
                for item in staged
            ),
            "quarantine_candidates": sum(
                item.get("interpretation_status") != "STABLE"
                for item in staged
            ),
            "duplicate_sources": len(staged)
            - len({item["source_hash"] for item in staged}),
        }
        preview = {
            "batch_id": batch_id,
            "status": "PREVIEW",
            "staging_ids": [item["id"] for item in staged],
            "metrics": metrics,
        }
        with self.repository.transaction() as conn:
            conn.execute(
                """INSERT INTO knowledge_batches
                   (id,status,config_json,preview_json,metrics_before_json,
                    metrics_after_json,created_at,updated_at)
                   VALUES(?,'PREVIEW',?,?,?,'{}',?,?)""",
                (
                    batch_id,
                    encode(dict(config or {})),
                    encode(preview),
                    encode(self.metrics(conn)),
                    utcnow(),
                    utcnow(),
                ),
            )
        return preview

    def commit_batch(
        self,
        batch_id: str,
        *,
        materializer: Optional[
            Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
        ] = None,
    ) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_batches WHERE id=?",
                (batch_id,),
            ).fetchone()
            if not row:
                raise KeyError(batch_id)
            preview = decode(row["preview_json"], {})
        decisions = [
            self.commit(staging_id, materializer=materializer)
            for staging_id in preview.get("staging_ids", [])
        ]
        batch_status = (
            "COMMITTED"
            if all(
                item.get("status") == "COMMITTED"
                for item in decisions
            )
            else "PARTIAL"
        )
        with self.repository.transaction() as conn:
            metrics_after = self.metrics(conn)
            conn.execute(
                """UPDATE knowledge_batches SET status=?,
                   metrics_after_json=?,updated_at=? WHERE id=?""",
                (
                    batch_status,
                    encode(metrics_after),
                    utcnow(),
                    batch_id,
                ),
            )
        return {
            "batch_id": batch_id,
            "status": batch_status,
            "decisions": decisions,
            "metrics": metrics_after,
        }

    def rollback_batch(self, batch_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            row = conn.execute(
                "SELECT preview_json FROM knowledge_batches WHERE id=?",
                (batch_id,),
            ).fetchone()
            if not row:
                raise KeyError(batch_id)
            staging_ids = decode(row["preview_json"], {}).get(
                "staging_ids",
                [],
            )
        retractions = [
            self.retract(
                staging_id,
                reason=f"rollback batch {batch_id}",
                operation="BATCH_ROLLBACK",
            )
            for staging_id in staging_ids
        ]
        with self.repository.transaction() as conn:
            conn.execute(
                """UPDATE knowledge_batches SET status='ROLLED_BACK',
                   metrics_after_json=?,updated_at=? WHERE id=?""",
                (encode(self.metrics(conn)), utcnow(), batch_id),
            )
        return {
            "batch_id": batch_id,
            "status": "ROLLED_BACK",
            "retractions": retractions,
        }

    @staticmethod
    def metrics(conn: Any) -> Dict[str, Any]:
        counts = {
            str(row["status"]): int(row["count"])
            for row in conn.execute(
                """SELECT status,COUNT(*) AS count
                   FROM knowledge_staging GROUP BY status"""
            )
        }
        independent_sources = int(conn.execute(
            """SELECT COUNT(DISTINCT independent_source_key)
               FROM knowledge_staging WHERE status='COMMITTED'"""
        ).fetchone()[0])
        committed = counts.get("COMMITTED", 0)
        contaminated = int(conn.execute(
            """SELECT COUNT(*) FROM knowledge_staging
               WHERE status='COMMITTED'
               AND interpretation_status<>'STABLE'"""
        ).fetchone()[0])
        return {
            "staging_by_status": counts,
            "independent_sources": independent_sources,
            "memory_contamination_rate": (
                round(contaminated / committed, 6) if committed else 0.0
            ),
            "quarantine_count": counts.get("QUARANTINED", 0),
            "retraction_count": int(conn.execute(
                "SELECT COUNT(*) FROM knowledge_retractions"
            ).fetchone()[0]),
        }
