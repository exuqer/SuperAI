"""Universal entity, event, construction and concept learning."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .language import (
    EntityMentionParser,
    MentionDraft,
    ParsedToken,
    UniversalLanguageAnalyzer,
)
from .repository import decode, encode, utcnow
from .semantics import RoleCompatibilityGraph, RoleHypothesisResolver


PARSER_VERSION = "universal-event-v2"
NOUN_POS = {"NOUN", "NPRO"}
ADJECTIVE_POS = {"ADJF", "ADJS", "PRTF", "PRTS", "NUMR"}
PREDICATE_POS = {"VERB", "INFN", "PRTS", "GRND"}
ROLE_VALUES = {
    "agent", "patient", "theme", "object", "experiencer", "recipient", "source",
    "destination", "location", "instrument", "material", "cause", "result", "purpose",
    "time", "attribute", "quantity", "owner", "possessed", "manner",
}
RELATION_VALUES = {
    "IS_A", "INSTANCE_OF", "PART_OF", "HAS_PART", "HAS_PROPERTY", "LOCATED_IN",
    "LOCATED_ON", "LOCATED_NEAR", "OWNS", "USES", "PRODUCES", "REQUIRES", "CAUSES",
    "RESULTS_IN", "BEFORE", "AFTER", "SIMILAR_TO", "OPPOSITE_TO", "ALIAS_OF",
}


def stable_id(prefix: str, *values: object, size: int = 20) -> str:
    payload = ":".join(str(value) for value in values)
    return f"{prefix}-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:size]


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class UniversalRoleResolver(RoleHypothesisResolver):
    """Compatibility name for the former event-core-local resolver."""


class UniversalEventPipeline:
    def __init__(self, repository: Any, morphology: Any) -> None:
        self.repository = repository
        self.morphology = morphology
        self.language = UniversalLanguageAnalyzer(morphology)
        self.mentions = EntityMentionParser()
        self.roles = UniversalRoleResolver()

    def _canonical_display_name(
        self,
        lemma: str,
        observed_surface: str,
        features: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Produce a nominative display form independently of its mention.

        An entity can first be seen in any case.  Its identity must retain a
        canonical lexical form, while event participants continue to preserve
        the observed surface needed for answer realization.
        """
        target_features: Dict[str, str] = {"case": "nomn"}
        observed_features = dict(features or {})
        if not observed_features and observed_surface:
            observed_features = dict(
                self.morphology.parse(observed_surface).features
            )
        number = observed_features.get("number")
        if number:
            target_features["number"] = str(number)
        display = self.morphology.inflect(lemma, target_features) or lemma
        if observed_surface[:1].isupper():
            display = display[:1].upper() + display[1:]
        return display

    def _scene_tokens(self, conn: Any, scene_id: int) -> tuple[str, List[ParsedToken]]:
        scene = conn.execute(
            "SELECT sentence_text FROM scenes WHERE cloud_id=?", (scene_id,)
        ).fetchone()
        if not scene:
            raise KeyError(scene_id)
        source_text = str(scene["sentence_text"])
        rows = conn.execute(
            """SELECT sc.token_index,sc.grammatical_role,sc.morphology_json,
                      sc.lexeme_cloud_id,sc.word_form_cloud_id,wf.normalized_form,
                      l.lemma,l.pos_tag
               FROM scene_components sc
               JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id
               LEFT JOIN lexemes l ON l.cloud_id=sc.lexeme_cloud_id
               WHERE sc.scene_cloud_id=? ORDER BY sc.token_index""",
            (scene_id,),
        ).fetchall()
        metadata: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            index = int(row["token_index"])
            metadata[index] = {
                "lexeme_cloud_id": (
                    int(row["lexeme_cloud_id"])
                    if row["lexeme_cloud_id"]
                    else None
                ),
                "word_form_cloud_id": (
                    int(row["word_form_cloud_id"])
                    if row["word_form_cloud_id"]
                    else None
                ),
                "grammatical_role": str(row["grammatical_role"]),
            }
        analysis = self.language.analyze(
            source_text,
            token_metadata=metadata,
            detect_question=False,
        )
        return source_text, analysis.tokens

    def _ensure_entity(
        self,
        conn: Any,
        lemma: str,
        display_name: str,
        *,
        source_scene_id: int,
        lexeme_cloud_id: Optional[int],
        alias: str,
        entity_kind: str = "entity",
    ) -> int:
        # The entity record stores a canonical lexical label.  The alias and
        # entity mention keep the exact observed inflected surface.
        display_name = self._canonical_display_name(lemma, display_name)
        canonical = f"{entity_kind}:{lemma.casefold()}"
        cloud, _ = self.repository.get_or_create_cloud(
            conn,
            "entity",
            canonical,
            mass=.8,
            stability=.2,
            metadata={"lemma": lemma.casefold(), "display_name": display_name},
        )
        now = utcnow()
        conn.execute(
            """INSERT INTO entities
               (cloud_id,canonical_lemma,display_name,entity_kind,status,confidence,
                metadata_json,created_at,updated_at)
               VALUES(?,?,?,?, 'OBSERVED',.65,'{}',?,?)
               ON CONFLICT(cloud_id) DO UPDATE SET
                 display_name=CASE WHEN length(excluded.display_name)>length(entities.display_name)
                                   THEN excluded.display_name ELSE entities.display_name END,
                 updated_at=excluded.updated_at""",
            (int(cloud["id"]), lemma.casefold(), display_name, entity_kind, now, now),
        )
        conn.execute(
            """INSERT INTO entity_aliases
               (entity_id,alias,normalized_alias,lexeme_cloud_id,source_scene_id,
                confidence,source_type,created_at)
               VALUES(?,?,?,?,?,1,'observation',?)
               ON CONFLICT(entity_id,normalized_alias) DO UPDATE SET
                 lexeme_cloud_id=COALESCE(excluded.lexeme_cloud_id,entity_aliases.lexeme_cloud_id),
                 confidence=MAX(entity_aliases.confidence,excluded.confidence)""",
            (
                int(cloud["id"]),
                alias,
                alias.casefold(),
                lexeme_cloud_id,
                source_scene_id,
                now,
            ),
        )
        return int(cloud["id"])

    def _ensure_relation(
        self,
        conn: Any,
        subject_id: int,
        relation_type: str,
        object_id: int,
        scene_id: int,
        *,
        confidence: float = .9,
        source_type: str = "EVENT_DERIVATION",
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        if relation_type not in RELATION_VALUES or subject_id == object_id:
            return ""
        relation_id = stable_id("relation", relation_type, subject_id, object_id)
        now = utcnow()
        conn.execute(
            """INSERT INTO concept_relations
               (id,relation_type,subject_lexeme_cloud_id,object_lexeme_cloud_id,
                confidence,status,direct,depth,evidence_count,source_type,created_at,updated_at)
               VALUES(?,?,?,?,?,'STABLE',1,1,0,?,?,?)
               ON CONFLICT(relation_type,subject_lexeme_cloud_id,object_lexeme_cloud_id)
               DO UPDATE SET confidence=MAX(concept_relations.confidence,excluded.confidence),
                             updated_at=excluded.updated_at""",
            (
                relation_id,
                relation_type,
                subject_id,
                object_id,
                confidence,
                source_type,
                now,
                now,
            ),
        )
        actual = conn.execute(
            """SELECT id FROM concept_relations
               WHERE relation_type=? AND subject_lexeme_cloud_id=?
                 AND object_lexeme_cloud_id=?""",
            (relation_type, subject_id, object_id),
        ).fetchone()
        relation_id = str(actual["id"])
        evidence_id = stable_id("relation-evidence", relation_id, scene_id)
        conn.execute(
            """INSERT OR IGNORE INTO concept_relation_evidence
               (id,concept_relation_id,source_scene_id,evidence_type,relation_type,
                weight,confidence,status,evidence_json,created_at)
               VALUES(?,?,?,'EVENT_DERIVATION',?,?,?,'STABLE',?,?)""",
            (
                evidence_id,
                relation_id,
                scene_id,
                relation_type,
                confidence,
                confidence,
                encode(payload or {}),
                now,
            ),
        )
        count = int(conn.execute(
            "SELECT COUNT(*) FROM concept_relation_evidence WHERE concept_relation_id=?",
            (relation_id,),
        ).fetchone()[0])
        conn.execute(
            "UPDATE concept_relations SET evidence_count=?,updated_at=? WHERE id=?",
            (count, now, relation_id),
        )
        return relation_id

    @staticmethod
    def _pattern(
        tokens: Sequence[ParsedToken],
        predicate_index: int,
        mentions: Sequence[MentionDraft] = (),
    ) -> str:
        mention_by_start = {mention.start: mention for mention in mentions}
        covered = {
            index
            for mention in mentions
            for index in mention.token_indices
        }
        relation_tokens = {
            index
            for mention in mentions
            if mention.preposition
            for index in range(
                max(0, mention.start - len(mention.preposition.split())),
                mention.start,
            )
        }
        values = []
        for token in tokens:
            if token.pos in {"PNCT"}:
                continue
            if token.index == predicate_index:
                values.append("VERB")
                continue
            mention = mention_by_start.get(token.index)
            if mention:
                relation = (
                    f"REL:{mention.relation_type}"
                    if mention.relation_type
                    else f"PREP:{mention.preposition.casefold()}"
                    if mention.preposition
                    else ""
                )
                noun_phrase = (
                    f"NP:{mention.features.get('case') or '*'}"
                    f":{mention.mention_type}"
                )
                values.extend(item for item in (relation, noun_phrase) if item)
                continue
            if token.index in covered:
                continue
            if token.index in relation_tokens:
                continue
            if token.pos in NOUN_POS:
                values.append(f"NOUN:{token.features.get('case') or '*'}")
            elif token.pos in ADJECTIVE_POS:
                values.append(f"ADJ:{token.features.get('case') or '*'}")
            else:
                values.append(token.pos)
        return " ".join(values)

    def _construction(
        self,
        conn: Any,
        scene_id: int,
        predicate_lemma: str,
        pattern: str,
        participants: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        construction_id = stable_id("construction", predicate_lemma, pattern)
        now = utcnow()
        conn.execute(
            """INSERT INTO construction_templates
               (id,predicate_lemma,surface_pattern,status,confidence,evidence_count,
                source_type,created_at,updated_at)
               VALUES(?,?,?,'OBSERVED',.55,0,'learned',?,?)
               ON CONFLICT(predicate_lemma,surface_pattern) DO UPDATE SET
                 updated_at=excluded.updated_at""",
            (construction_id, predicate_lemma, pattern, now, now),
        )
        conn.execute(
            """INSERT OR IGNORE INTO construction_evidence
               (id,construction_id,source_scene_id,evidence_type,weight,payload_json,created_at)
               VALUES(?,?,?,'SCENE_PATTERN',.7,?,?)""",
            (
                stable_id("construction-evidence", construction_id, scene_id),
                construction_id,
                scene_id,
                encode({"pattern": pattern}),
                now,
            ),
        )
        for index, participant in enumerate(participants):
            for hypothesis in participant["hypotheses"]:
                conn.execute(
                    """INSERT INTO construction_arguments
                       (id,construction_id,argument_index,grammatical_slot,
                        morphological_constraints_json,semantic_role,confidence)
                       VALUES(?,?,?,?,?,?,?)
                       ON CONFLICT(construction_id,argument_index,semantic_role)
                       DO UPDATE SET confidence=MAX(construction_arguments.confidence,
                                                    excluded.confidence)""",
                    (
                        stable_id(
                            "construction-argument",
                            construction_id,
                            index,
                            hypothesis["role"],
                        ),
                        construction_id,
                        index,
                        participant["slot"],
                        encode(participant["features"]),
                        hypothesis["role"],
                        hypothesis["confidence"],
                    ),
                )
        count = int(conn.execute(
            "SELECT COUNT(*) FROM construction_evidence WHERE construction_id=?",
            (construction_id,),
        ).fetchone()[0])
        status = "OBSERVED" if count == 1 else "CANDIDATE" if count == 2 else "PROBABLE" if count < 5 else "STABLE"
        confidence = min(.95, .5 + .1 * count)
        conn.execute(
            """UPDATE construction_templates
               SET status=?,confidence=?,evidence_count=?,updated_at=? WHERE id=?""",
            (status, confidence, count, now, construction_id),
        )
        return {
            "id": construction_id,
            "predicate_lemma": predicate_lemma,
            "surface_pattern": pattern,
            "status": status,
            "confidence": confidence,
            "evidence_count": count,
        }

    @staticmethod
    def classify_observation(text: str) -> str:
        normalized = " ".join(text.casefold().replace("—", "-").split())
        if "противополож" in normalized or "антоним" in normalized:
            return "SEMANTIC_OPPOSITION"
        if "близк" in normalized or "синоним" in normalized or " здесь значит " in normalized:
            return "SEMANTIC_SIMILARITY"
        if any(marker in normalized for marker in ("потому что", "приводит к", "вызывает")):
            return "CAUSE_EFFECT"
        if any(marker in normalized for marker in ("после", "затем", "до того")):
            return "TEMPORAL_SEQUENCE"
        if any(marker in normalized for marker in (" это ", " является ", " относится к ", " - ")):
            return "CLASSIFICATION"
        return "WORLD_EVENT"

    def _learn_action_concept(
        self,
        conn: Any,
        scene_id: int,
        text: str,
        tokens: Sequence[ParsedToken],
    ) -> Optional[Dict[str, Any]]:
        kind = self.classify_observation(text)
        if kind not in {
            "SEMANTIC_SIMILARITY",
            "SEMANTIC_OPPOSITION",
            "CLASSIFICATION",
        }:
            return None
        lemmas = []
        for token in tokens:
            if token.pos not in PREDICATE_POS:
                continue
            if token.lemma not in lemmas:
                lemmas.append(token.lemma)
        nominalization_links: List[Dict[str, Any]] = []
        verb_rows = conn.execute(
            """SELECT lemma FROM lexemes
               WHERE pos_tag IN ('VERB','INFN')
               ORDER BY lemma"""
        ).fetchall()
        for token in tokens:
            if token.pos not in NOUN_POS:
                continue
            best: Optional[tuple[int, str]] = None
            for row in verb_rows:
                verb_lemma = str(row["lemma"])
                common = 0
                for left, right in zip(token.lemma, verb_lemma):
                    if left != right:
                        break
                    common += 1
                if (
                    common >= 5
                    and common / max(1, min(len(token.lemma), len(verb_lemma))) >= .6
                    and (best is None or common > best[0])
                ):
                    best = (common, verb_lemma)
            if best and best[1] not in lemmas:
                lemmas.append(best[1])
                nominalization_links.append({
                    "nominalization": token.lemma,
                    "predicate": best[1],
                    "common_prefix_length": best[0],
                })
        if len(lemmas) < 2:
            return None
        members = sorted(lemmas[:8])
        concept_id = stable_id("concept-action", *members)
        canonical = "action:" + hashlib.sha256("|".join(members).encode("utf-8")).hexdigest()[:16]
        display = " / ".join(members)
        now = utcnow()
        status = "CONFLICTED" if kind == "SEMANTIC_OPPOSITION" else "PROBABLE"
        confidence = .72 if status == "PROBABLE" else .65
        cloud, _ = self.repository.get_or_create_cloud(
            conn,
            "concept",
            canonical,
            mass=.7,
            stability=confidence,
            metadata={"concept_kind": "action", "members": members},
        )
        conn.execute(
            """INSERT INTO concepts
               (id,cloud_id,concept_kind,canonical_name,display_name,status,confidence,
                evidence_count,source_type,created_at,updated_at)
               VALUES(?,?, 'action',?,?,?,?,0,'learned',?,?)
               ON CONFLICT(id) DO UPDATE SET
                 status=CASE WHEN concepts.status='CONFLICTED' THEN concepts.status
                             ELSE excluded.status END,
                 confidence=MAX(concepts.confidence,excluded.confidence),
                 updated_at=excluded.updated_at""",
            (
                concept_id,
                int(cloud["id"]),
                canonical,
                display,
                status,
                confidence,
                now,
                now,
            ),
        )
        evidence_type = (
            "EXPLICIT_OPPOSITION"
            if status == "CONFLICTED"
            else "EXPLICIT_DEFINITION"
            if kind == "CLASSIFICATION"
            else "EXPLICIT_SIMILARITY"
        )
        conn.execute(
            """INSERT OR IGNORE INTO concept_evidence
               (id,concept_id,source_scene_id,evidence_type,weight,confidence,
                independence_key,status,payload_json,created_at)
               VALUES(?,?,?,?,?,?,?,'ACTIVE',?,?)""",
            (
                stable_id("concept-evidence", concept_id, scene_id, evidence_type),
                concept_id,
                scene_id,
                evidence_type,
                confidence,
                confidence,
                f"scene:{scene_id}:{evidence_type}",
                encode({"text": text, "members": members}),
                now,
            ),
        )
        for link in nominalization_links:
            conn.execute(
                """INSERT OR IGNORE INTO concept_evidence
                   (id,concept_id,source_scene_id,evidence_type,weight,confidence,
                    independence_key,status,payload_json,created_at)
                   VALUES(?,?,?,'MORPHOLOGICAL_LINK',.68,.68,?,'ACTIVE',?,?)""",
                (
                    stable_id(
                        "concept-evidence",
                        concept_id,
                        scene_id,
                        link["nominalization"],
                        link["predicate"],
                    ),
                    concept_id,
                    scene_id,
                    (
                        f"scene:{scene_id}:nominalization:"
                        f"{link['nominalization']}:{link['predicate']}"
                    ),
                    encode(link),
                    now,
                ),
            )
        for lemma in members:
            lexeme = conn.execute(
                "SELECT cloud_id FROM lexemes WHERE lemma=? ORDER BY cloud_id LIMIT 1",
                (lemma,),
            ).fetchone()
            member_cloud_id = int(lexeme["cloud_id"]) if lexeme else None
            conn.execute(
                """INSERT INTO concept_members
                   (id,concept_id,member_cloud_id,member_lemma,member_role,weight,
                    confidence,evidence_count,created_at,updated_at)
                   VALUES(?,?,?,?, 'variant',?,?,1,?,?)
                   ON CONFLICT(concept_id,member_lemma,member_role) DO UPDATE SET
                     member_cloud_id=COALESCE(excluded.member_cloud_id,concept_members.member_cloud_id),
                     confidence=MAX(concept_members.confidence,excluded.confidence),
                     updated_at=excluded.updated_at""",
                (
                    stable_id("concept-member", concept_id, lemma),
                    concept_id,
                    member_cloud_id,
                    lemma,
                    confidence,
                    confidence,
                    now,
                    now,
                ),
            )
        relation_type = (
            "OPPOSITE_TO"
            if kind == "SEMANTIC_OPPOSITION"
            else "SIMILAR_TO"
        )
        member_ids = {
            str(row["member_lemma"]): int(row["member_cloud_id"])
            for row in conn.execute(
                """SELECT member_lemma,member_cloud_id FROM concept_members
                   WHERE concept_id=? AND member_cloud_id IS NOT NULL""",
                (concept_id,),
            ).fetchall()
        }
        for left_index, left in enumerate(members):
            for right in members[left_index + 1:]:
                if left not in member_ids or right not in member_ids:
                    continue
                self._ensure_relation(
                    conn,
                    member_ids[left],
                    relation_type,
                    member_ids[right],
                    scene_id,
                    confidence=confidence,
                    source_type=evidence_type,
                    payload={"concept_id": concept_id},
                )
                self._ensure_relation(
                    conn,
                    member_ids[right],
                    relation_type,
                    member_ids[left],
                    scene_id,
                    confidence=confidence,
                    source_type=evidence_type,
                    payload={"concept_id": concept_id},
                )
        evidence_count = int(conn.execute(
            "SELECT COUNT(*) FROM concept_evidence WHERE concept_id=? AND status='ACTIVE'",
            (concept_id,),
        ).fetchone()[0])
        if status != "CONFLICTED":
            status = "STABLE" if evidence_count >= 3 else "PROBABLE"
        conn.execute(
            """UPDATE concepts
               SET evidence_count=?,status=?,updated_at=? WHERE id=?""",
            (evidence_count, status, now, concept_id),
        )
        if status != "CONFLICTED":
            compatibility_status = "STABLE" if evidence_count >= 3 else "PROBABLE"
            conn.execute(
                """INSERT INTO action_concepts
                   (id,canonical_name,display_name,space_id,status,confidence,mass,
                    evidence_count,created_at,updated_at)
                   VALUES(?,?,?,NULL,?,?,.7,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     status=excluded.status,confidence=MAX(action_concepts.confidence,
                     excluded.confidence),evidence_count=excluded.evidence_count,
                     updated_at=excluded.updated_at""",
                (
                    concept_id,
                    canonical,
                    display,
                    compatibility_status,
                    confidence,
                    evidence_count,
                    now,
                    now,
                ),
            )
            for lemma in members:
                lexeme = conn.execute(
                    "SELECT cloud_id FROM lexemes WHERE lemma=? ORDER BY cloud_id LIMIT 1",
                    (lemma,),
                ).fetchone()
                conn.execute(
                    """INSERT INTO action_variants
                       (id,action_concept_id,lexeme_cloud_id,lemma,construction_id,
                        weight,evidence_count,source_type,created_at,updated_at)
                       VALUES(?,?,?,?,NULL,?,?,'explicit_similarity',?,?)
                       ON CONFLICT(action_concept_id,lemma) DO UPDATE SET
                         lexeme_cloud_id=COALESCE(excluded.lexeme_cloud_id,
                                                  action_variants.lexeme_cloud_id),
                         weight=MAX(action_variants.weight,excluded.weight),
                         evidence_count=MAX(action_variants.evidence_count,
                                            excluded.evidence_count),
                         source_type='explicit_similarity',updated_at=excluded.updated_at""",
                    (
                        stable_id("action-variant", concept_id, lemma),
                        concept_id,
                        int(lexeme["cloud_id"]) if lexeme else None,
                        lemma,
                        confidence,
                        evidence_count,
                        now,
                        now,
                    ),
                )
        return {
            "id": concept_id,
            "canonical_name": canonical,
            "display_name": display,
            "status": status,
            "confidence": confidence,
            "members": members,
            "evidence_count": evidence_count,
        }

    def _project_action_concept(
        self,
        conn: Any,
        scene_id: int,
        predicate_lemma: str,
        participants: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        variant = conn.execute(
            """SELECT av.*,ac.status AS concept_status
               FROM action_variants av
               JOIN action_concepts ac ON ac.id=av.action_concept_id
               WHERE av.lemma=? AND av.source_type<>'manual_seed'
                 AND av.weight>0
                 AND ac.status IN ('PROBABLE','STABLE')
                 AND EXISTS (
                    SELECT 1 FROM scenes
                    WHERE cloud_id=? AND knowledge_status<>'RETRACTED'
                 )
               ORDER BY av.weight DESC LIMIT 1""",
            (predicate_lemma, scene_id),
        ).fetchone()
        if not variant:
            return None
        semantic_frame = {
            participant["role"]: {
                "entity_id": participant["entity_id"],
                "lemma": participant["lemma"],
                "surface": participant["surface"],
                "grammatical_slot": participant["slot"],
            }
            for participant in participants
        }
        now = utcnow()
        projection_id = stable_id(
            "scene-projection",
            scene_id,
            variant["action_concept_id"],
            PARSER_VERSION,
        )
        conn.execute(
            """INSERT INTO scene_concept_projections
               (id,scene_id,action_concept_id,semantic_frame_json,
                projection_confidence,projection_version,source_type,created_at,updated_at)
               VALUES(?,?,?,?,?,2,'learned_evidence',?,?)
               ON CONFLICT(scene_id,action_concept_id,projection_version) DO UPDATE SET
                 semantic_frame_json=excluded.semantic_frame_json,
                 projection_confidence=excluded.projection_confidence,
                 source_type=excluded.source_type,updated_at=excluded.updated_at""",
            (
                projection_id,
                scene_id,
                variant["action_concept_id"],
                encode(semantic_frame),
                float(variant["weight"]),
                now,
                now,
            ),
        )
        return {
            "id": projection_id,
            "action_concept_id": str(variant["action_concept_id"]),
            "semantic_frame": semantic_frame,
            "confidence": float(variant["weight"]),
        }

    def materialize_scene(
        self,
        conn: Any,
        scene_id: int,
        clause_interpretation: Optional[Mapping[str, Any]] = None,
        admission_decision_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_status = conn.execute(
            "SELECT knowledge_status FROM scenes WHERE cloud_id=?",
            (scene_id,),
        ).fetchone()
        if (
            not source_status
            or str(source_status["knowledge_status"]) == "RETRACTED"
        ):
            return {
                "scene_cloud_id": scene_id,
                "status": "SKIPPED_RETRACTED",
            }
        source_text, tokens = self._scene_tokens(conn, scene_id)
        previous_event = conn.execute(
            "SELECT construction_id FROM events WHERE source_scene_id=?",
            (scene_id,),
        ).fetchone()
        previous_construction_id = (
            str(previous_event["construction_id"])
            if previous_event and previous_event["construction_id"]
            else None
        )
        relation_phrases = self.language.relation_phrases.parse(tokens)
        drafts = self.mentions.parse(tokens, relation_phrases)
        predicate = next(
            (
                token for token in tokens
                if token.grammatical_role == "predicate" or token.pos in PREDICATE_POS
            ),
            None,
        )
        if not predicate:
            predicate = ParsedToken(
                index=-1,
                surface="",
                normalized="",
                lemma="state",
                pos="VERB",
                features={},
                lexeme_cloud_id=None,
                word_form_cloud_id=None,
                grammatical_role="predicate",
            )
        mention_rows: List[Dict[str, Any]] = []
        conn.execute("DELETE FROM entity_mentions WHERE source_scene_id=?", (scene_id,))
        for draft in drafts:
            head = tokens[draft.head]
            canonical_display_name = self._canonical_display_name(
                draft.lemma,
                head.surface,
                draft.features,
            )
            entity_id = self._ensure_entity(
                conn,
                draft.lemma,
                canonical_display_name,
                source_scene_id=scene_id,
                lexeme_cloud_id=head.lexeme_cloud_id,
                alias=head.surface,
            )
            if draft.normalized_surface != head.normalized:
                conn.execute(
                    """INSERT INTO entity_aliases
                       (entity_id,alias,normalized_alias,lexeme_cloud_id,
                        source_scene_id,confidence,source_type,created_at)
                       VALUES(?,?,?,?,?,.92,'phrase_observation',?)
                       ON CONFLICT(entity_id,normalized_alias) DO UPDATE SET
                         confidence=MAX(entity_aliases.confidence,
                                        excluded.confidence)""",
                    (
                        entity_id,
                        draft.surface,
                        draft.normalized_surface,
                        head.lexeme_cloud_id,
                        scene_id,
                        utcnow(),
                    ),
                )
            entity_type_id = None
            if draft.type_token is not None:
                type_head = tokens[draft.type_token]
                entity_type_id = self._ensure_entity(
                    conn,
                    type_head.lemma,
                    type_head.surface,
                    source_scene_id=scene_id,
                    lexeme_cloud_id=type_head.lexeme_cloud_id,
                    alias=type_head.surface,
                    entity_kind="type",
                )
                self._ensure_relation(
                    conn,
                    entity_id,
                    "IS_A",
                    entity_type_id,
                    scene_id,
                    confidence=.96,
                    source_type="APPOSITION",
                    payload={"surface": draft.surface},
                )
            mention_id = stable_id("mention", scene_id, draft.start, draft.end)
            now = utcnow()
            conn.execute(
                """INSERT INTO entity_mentions
                   (id,source_scene_id,entity_id,token_start,token_end,head_token_index,
                    surface,normalized_surface,mention_type,entity_type_id,preposition,
                    grammatical_features_json,attributes_json,confidence,parser_version,
                    created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, ?,?)""",
                (
                    mention_id,
                    scene_id,
                    entity_id,
                    draft.start,
                    draft.end,
                    draft.head,
                    draft.surface,
                    draft.normalized_surface,
                    "apposition" if draft.type_token is not None else "noun_phrase",
                    entity_type_id,
                    draft.preposition,
                    encode(draft.features),
                    encode(draft.attributes),
                    .96 if draft.type_token is not None else .82,
                    PARSER_VERSION,
                    now,
                    now,
                ),
            )
            owner_entity_id = None
            if draft.owner_token is not None:
                owner = tokens[draft.owner_token]
                owner_entity_id = self._ensure_entity(
                    conn,
                    owner.lemma,
                    owner.surface,
                    source_scene_id=scene_id,
                    lexeme_cloud_id=owner.lexeme_cloud_id,
                    alias=owner.surface,
                )
                self._ensure_relation(
                    conn,
                    owner_entity_id,
                    "OWNS",
                    entity_id,
                    scene_id,
                    confidence=.72,
                    source_type="GENITIVE_DEPENDENCY",
                    payload={"surface": draft.surface},
                )
            for attribute in draft.attributes:
                attribute_id = self._ensure_entity(
                    conn,
                    attribute,
                    attribute,
                    source_scene_id=scene_id,
                    lexeme_cloud_id=None,
                    alias=attribute,
                    entity_kind="attribute",
                )
                self._ensure_relation(
                    conn,
                    entity_id,
                    "HAS_PROPERTY",
                    attribute_id,
                    scene_id,
                    confidence=.78,
                    source_type="ATTRIBUTE_DEPENDENCY",
                )
            mention_rows.append({
                "id": mention_id,
                "entity_id": entity_id,
                "entity_type_id": entity_type_id,
                "owner_entity_id": owner_entity_id,
                "start": draft.start,
                "end": draft.end,
                "head": draft.head,
                "surface": draft.surface,
                "lemma": draft.lemma,
                "features": draft.features,
                "preposition": draft.preposition,
                "relation_type": draft.relation_type,
                "relation_function": draft.relation_function,
                "attributes": draft.attributes,
                "mention_type": draft.mention_type,
                "entity_value_surface": head.surface,
                "entity_type_surface": (
                    tokens[draft.type_token].surface
                    if draft.type_token is not None
                    else None
                ),
            })
        has_preverbal_subject = any(
            mention["start"] < predicate.index
            and mention["features"].get("case") in {"nomn", None}
            for mention in mention_rows
        )
        predicate_profile = self.roles.spatial.predicate_profile(
            conn, predicate.lemma
        )
        participant_drafts: List[Dict[str, Any]] = []
        for mention in mention_rows:
            if mention["owner_entity_id"] and mention["head"] != mention["start"]:
                pass
            slot = self.roles.grammatical_slot(
                MentionDraft(
                    start=mention["start"],
                    end=mention["end"],
                    head=mention["head"],
                    token_indices=list(range(mention["start"], mention["end"] + 1)),
                    surface=mention["surface"],
                    normalized_surface=mention["surface"].casefold(),
                    lemma=mention["lemma"],
                    features=mention["features"],
                    preposition=mention["preposition"],
                    attributes=mention["attributes"],
                    relation_type=mention.get("relation_type"),
                    relation_function=mention.get("relation_function"),
                ),
                predicate.index,
                has_preverbal_subject,
                predicate_profile=predicate_profile,
            )
            participant_drafts.append({
                **mention,
                "slot": slot,
            })
        has_direct_object = any(
            participant["slot"] == "direct_object" for participant in participant_drafts
        )
        for participant in participant_drafts:
            participant["hypotheses"] = self.roles.hypotheses(
                participant["slot"],
                has_direct_object,
                predicate_lemma=predicate.lemma,
                conn=conn,
            )
            participant["role"] = participant["hypotheses"][0]["role"]
            participant["confidence"] = participant["hypotheses"][0]["confidence"]
        pattern = self._pattern(tokens, predicate.index, drafts)
        construction = self._construction(
            conn,
            scene_id,
            predicate.lemma,
            pattern,
            participant_drafts,
        )
        event_id = stable_id("event", scene_id)
        now = utcnow()
        clause = dict(clause_interpretation or {})
        event_polarity = str(
            clause.get("polarity")
            or (
                "NEGATIVE"
                if any(token.normalized == "не" for token in tokens)
                else "POSITIVE"
            )
        ).casefold()
        event_modality = str(clause.get("modality") or "fact").casefold()
        conn.execute(
            """INSERT INTO events
               (id,source_scene_id,predicate_lemma,predicate_surface,
                predicate_lexeme_cloud_id,construction_id,polarity,modality,
                confidence,parser_version,created_at,updated_at,source_clause_id,
                actuality,evidence_status,negation_scope_json,
                completion_status,temporal_anchor_json,attribution_json,
                admission_decision_id)
               VALUES(?,?,?,?,?,?,?,?,.82,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(source_scene_id) DO UPDATE SET
                 predicate_lemma=excluded.predicate_lemma,
                 predicate_surface=excluded.predicate_surface,
                 predicate_lexeme_cloud_id=excluded.predicate_lexeme_cloud_id,
                 construction_id=excluded.construction_id,
                 polarity=excluded.polarity,modality=excluded.modality,
                 confidence=excluded.confidence,
                 parser_version=excluded.parser_version,
                 source_clause_id=COALESCE(
                   excluded.source_clause_id,events.source_clause_id),
                 actuality=excluded.actuality,
                 evidence_status=excluded.evidence_status,
                 negation_scope_json=excluded.negation_scope_json,
                 completion_status=excluded.completion_status,
                 temporal_anchor_json=excluded.temporal_anchor_json,
                 attribution_json=excluded.attribution_json,
                 admission_decision_id=COALESCE(
                   excluded.admission_decision_id,
                   events.admission_decision_id),
                 updated_at=excluded.updated_at""",
            (
                event_id,
                scene_id,
                predicate.lemma,
                predicate.surface,
                predicate.lexeme_cloud_id,
                construction["id"],
                event_polarity,
                event_modality,
                PARSER_VERSION,
                now,
                now,
                clause.get("id"),
                clause.get("actuality") or "ACTUAL",
                clause.get("evidence_status") or "OBSERVED",
                encode(clause.get("negation_scope"))
                if clause.get("negation_scope") else None,
                clause.get("completion_status") or "UNKNOWN",
                encode(clause.get("temporal_anchor"))
                if clause.get("temporal_anchor") else None,
                encode({
                    "speaker": clause.get("speaker"),
                    "quoted_speaker": clause.get("quoted_speaker"),
                }) if clause else None,
                admission_decision_id,
            ),
        )
        stored_event = conn.execute(
            "SELECT id FROM events WHERE source_scene_id=?",
            (scene_id,),
        ).fetchone()
        event_id = str(stored_event["id"])
        if (
            previous_construction_id
            and previous_construction_id != construction["id"]
        ):
            conn.execute(
                """DELETE FROM construction_evidence
                   WHERE construction_id=? AND source_scene_id=?""",
                (previous_construction_id, scene_id),
            )
            previous_count = int(conn.execute(
                """SELECT COUNT(*) FROM construction_evidence
                   WHERE construction_id=?""",
                (previous_construction_id,),
            ).fetchone()[0])
            if previous_count == 0:
                conn.execute(
                    "DELETE FROM construction_templates WHERE id=?",
                    (previous_construction_id,),
                )
            else:
                previous_status = (
                    "OBSERVED"
                    if previous_count == 1
                    else "CANDIDATE"
                    if previous_count == 2
                    else "PROBABLE"
                    if previous_count < 5
                    else "STABLE"
                )
                conn.execute(
                    """UPDATE construction_templates
                       SET evidence_count=?,status=?,confidence=?,updated_at=?
                       WHERE id=?""",
                    (
                        previous_count,
                        previous_status,
                        min(.95, .5 + .1 * previous_count),
                        now,
                        previous_construction_id,
                    ),
                )
        conn.execute("DELETE FROM event_modifiers WHERE event_id=?", (event_id,))
        conn.execute("DELETE FROM event_participants WHERE event_id=?", (event_id,))
        participants: List[Dict[str, Any]] = []
        for index, participant in enumerate(participant_drafts):
            participant_id = stable_id("participant", event_id, index)
            conn.execute(
                """INSERT INTO event_participants
                   (id,event_id,entity_id,mention_id,semantic_role,grammatical_slot,
                    participant_index,confidence,preposition,surface,lemma,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    participant_id,
                    event_id,
                    participant["entity_id"],
                    participant["id"],
                    participant["role"],
                    participant["slot"],
                    index,
                    participant["confidence"],
                    participant["preposition"],
                    participant["surface"],
                    participant["lemma"],
                    now,
                ),
            )
            for hypothesis in participant["hypotheses"]:
                conn.execute(
                    """INSERT INTO event_role_hypotheses
                       (id,participant_id,semantic_role,confidence,source_type,
                        evidence_json,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        stable_id("role-hypothesis", participant_id, hypothesis["role"]),
                        participant_id,
                        hypothesis["role"],
                        hypothesis["confidence"],
                        hypothesis["source_type"],
                        encode({"evidence": hypothesis.get("evidence", [])}),
                        now,
                    ),
                )
            participants.append({
                "id": participant_id,
                "entity_id": participant["entity_id"],
                "mention_id": participant["id"],
                "role": participant["role"],
                "slot": participant["slot"],
                "lemma": participant["lemma"],
                "surface": participant["surface"],
                "preposition": participant["preposition"],
                "attributes": participant["attributes"],
                "owner_entity_id": participant["owner_entity_id"],
                "mention_type": participant["mention_type"],
                "mention_surface": participant["surface"],
                "entity_value_surface": participant["entity_value_surface"],
                "entity_type_surface": participant["entity_type_surface"],
                "confidence": participant["confidence"],
                "hypotheses": participant["hypotheses"],
            })
        modifiers: List[Dict[str, Any]] = []
        for participant in participants:
            for attribute in participant["attributes"]:
                modifier = {
                    "id": stable_id(
                        "event-modifier",
                        event_id,
                        participant["id"],
                        "attribute",
                        attribute,
                    ),
                    "event_id": event_id,
                    "target_participant_id": participant["id"],
                    "role": "attribute",
                    "value_entity_id": None,
                    "value_text": attribute,
                    "attributes": [],
                    "confidence": .78,
                    "source_mention_id": participant["mention_id"],
                }
                conn.execute(
                    """INSERT INTO event_modifiers
                       (id,event_id,target_participant_id,role,value_entity_id,
                        value_text,attributes_json,confidence,source_mention_id,
                        created_at)
                       VALUES(?,?,?,?,?,?,?, ?,?,?)""",
                    (
                        modifier["id"],
                        event_id,
                        participant["id"],
                        modifier["role"],
                        None,
                        attribute,
                        "[]",
                        modifier["confidence"],
                        participant["mention_id"],
                        now,
                    ),
                )
                modifiers.append(modifier)
            if participant["owner_entity_id"]:
                modifier = {
                    "id": stable_id(
                        "event-modifier",
                        event_id,
                        participant["id"],
                        "owner",
                        participant["owner_entity_id"],
                    ),
                    "event_id": event_id,
                    "target_participant_id": participant["id"],
                    "role": "owner",
                    "value_entity_id": participant["owner_entity_id"],
                    "value_text": "",
                    "attributes": [],
                    "confidence": .72,
                    "source_mention_id": participant["mention_id"],
                }
                conn.execute(
                    """INSERT INTO event_modifiers
                       (id,event_id,target_participant_id,role,value_entity_id,
                        value_text,attributes_json,confidence,source_mention_id,
                        created_at)
                       VALUES(?,?,?,?,?,'','[]',?,?,?)""",
                    (
                        modifier["id"],
                        event_id,
                        participant["id"],
                        modifier["role"],
                        participant["owner_entity_id"],
                        modifier["confidence"],
                        participant["mention_id"],
                        now,
                    ),
                )
                modifiers.append(modifier)
        relation_token_indices = {
            index
            for relation in relation_phrases
            for index in range(relation.token_start, relation.token_end + 1)
        }
        for token in tokens:
            if token.pos != "ADVB" or token.index in relation_token_indices:
                continue
            modifier = {
                "id": stable_id(
                    "event-modifier",
                    event_id,
                    "manner",
                    token.index,
                    token.lemma,
                ),
                "event_id": event_id,
                "target_participant_id": None,
                "role": "manner",
                "value_entity_id": None,
                "value_text": token.surface,
                "attributes": [],
                "confidence": .55,
                "source_mention_id": None,
            }
            conn.execute(
                """INSERT INTO event_modifiers
                   (id,event_id,target_participant_id,role,value_entity_id,
                    value_text,attributes_json,confidence,source_mention_id,
                    created_at)
                   VALUES(?,?,NULL,'manner',NULL,?,'[]',.55,NULL,?)""",
                (modifier["id"], event_id, token.surface, now),
            )
            modifiers.append(modifier)
        observation_type = self.classify_observation(source_text)
        relation_participants = [
            participant
            for participant in participant_drafts
            if participant.get("entity_id")
        ]
        morphological_links: List[Dict[str, Any]] = []
        if observation_type in {"CAUSE_EFFECT", "TEMPORAL_SEQUENCE"}:
            predicate_rows = conn.execute(
                """SELECT DISTINCT predicate_lemma,predicate_lexeme_cloud_id
                   FROM events
                   WHERE predicate_lexeme_cloud_id IS NOT NULL
                   ORDER BY predicate_lemma"""
            ).fetchall()
            for participant in relation_participants:
                nominalization = str(participant.get("lemma") or "")
                best: Optional[tuple[int, str, int]] = None
                for row in predicate_rows:
                    predicate_lemma = str(row["predicate_lemma"])
                    common = 0
                    for left, right in zip(nominalization, predicate_lemma):
                        if left != right:
                            break
                        common += 1
                    if (
                        common >= 5
                        and common / max(
                            1,
                            min(len(nominalization), len(predicate_lemma)),
                        ) >= .6
                        and (best is None or common > best[0])
                    ):
                        best = (
                            common,
                            predicate_lemma,
                            int(row["predicate_lexeme_cloud_id"]),
                        )
                if not best:
                    continue
                link = {
                    "nominalization": nominalization,
                    "predicate": best[1],
                    "common_prefix_length": best[0],
                    "source_scene_id": scene_id,
                }
                self._ensure_relation(
                    conn,
                    int(participant["entity_id"]),
                    "ALIAS_OF",
                    best[2],
                    scene_id,
                    confidence=.68,
                    source_type="MORPHOLOGICAL_LINK",
                    payload=link,
                )
                morphological_links.append(link)
        subject = next(
            (
                participant for participant in relation_participants
                if participant.get("slot") == "subject"
            ),
            relation_participants[0] if relation_participants else None,
        )
        if subject:
            for participant in relation_participants:
                if participant is subject:
                    continue
                if participant.get("slot") == "location_oblique":
                    preposition = str(participant.get("preposition") or "")
                    relation_type = (
                        participant.get("relation_type")
                        if participant.get("relation_type") in {
                            "LOCATED_IN", "LOCATED_ON", "LOCATED_NEAR"
                        }
                        else
                        "LOCATED_ON"
                        if preposition.casefold() in {"на", "над", "под"}
                        else "LOCATED_NEAR"
                        if preposition.casefold() in {"около", "возле", "рядом"}
                        else "LOCATED_IN"
                    )
                    self._ensure_relation(
                        conn,
                        int(subject["entity_id"]),
                        relation_type,
                        int(participant["entity_id"]),
                        scene_id,
                        confidence=.82,
                        source_type="SPATIAL_DEPENDENCY",
                        payload={
                            "preposition": preposition,
                            "grammatical_slot": participant["slot"],
                        },
                    )
                if participant.get("slot") == "instrumental":
                    self._ensure_relation(
                        conn,
                        int(subject["entity_id"]),
                        "USES",
                        int(participant["entity_id"]),
                        scene_id,
                        confidence=.76,
                        source_type="INSTRUMENT_DEPENDENCY",
                        payload={"grammatical_slot": participant["slot"]},
                    )
        if len(relation_participants) >= 2:
            left = relation_participants[0]
            right = relation_participants[1]
            if observation_type == "CAUSE_EFFECT":
                for relation_type in ("CAUSES", "RESULTS_IN"):
                    self._ensure_relation(
                        conn,
                        int(left["entity_id"]),
                        relation_type,
                        int(right["entity_id"]),
                        scene_id,
                        confidence=.9,
                        source_type="EXPLICIT_CAUSAL_MARKER",
                        payload={"surface": source_text},
                    )
            elif observation_type == "TEMPORAL_SEQUENCE":
                normalized = source_text.casefold()
                if "после" in normalized:
                    relation_pairs = (
                        (left, "AFTER", right),
                        (right, "BEFORE", left),
                    )
                else:
                    relation_pairs = (
                        (left, "BEFORE", right),
                        (right, "AFTER", left),
                    )
                for subject, relation_type, object_ in relation_pairs:
                    self._ensure_relation(
                        conn,
                        int(subject["entity_id"]),
                        relation_type,
                        int(object_["entity_id"]),
                        scene_id,
                        confidence=.86,
                        source_type="EXPLICIT_TEMPORAL_MARKER",
                        payload={"surface": source_text},
                    )
        concept = self._learn_action_concept(conn, scene_id, source_text, tokens)
        projection = self._project_action_concept(
            conn,
            scene_id,
            predicate.lemma,
            participants,
        )
        if concept and concept["status"] != "CONFLICTED":
            member_marks = ",".join("?" for _ in concept["members"])
            related_events = conn.execute(
                f"""SELECT source_scene_id FROM events
                    WHERE predicate_lemma IN ({member_marks})
                    ORDER BY source_scene_id""",
                concept["members"],
            ).fetchall()
            for related in related_events:
                related_scene_id = int(related["source_scene_id"])
                loaded = self.load_event(conn, related_scene_id)
                if not loaded:
                    continue
                related_projection = self._project_action_concept(
                    conn,
                    related_scene_id,
                    str(loaded["predicate"]["lemma"]),
                    [
                        {
                            **participant,
                            "slot": participant["grammatical_slot"],
                        }
                        for participant in loaded["participants"]
                    ],
                )
                if related_scene_id == scene_id and related_projection:
                    projection = related_projection
        return {
            "event": {
                "id": event_id,
                "source_scene_id": scene_id,
                "predicate": {
                    "lemma": predicate.lemma,
                    "surface": predicate.surface,
                    "lexeme_cloud_id": predicate.lexeme_cloud_id,
                },
                "participants": participants,
                "modifiers": modifiers,
                "source_clause_id": clause.get("id"),
                "polarity": event_polarity,
                "modality": event_modality,
                "actuality": clause.get("actuality") or "ACTUAL",
                "evidence_status": (
                    clause.get("evidence_status") or "OBSERVED"
                ),
                "negation_scope": deepcopy(
                    clause.get("negation_scope")
                ),
                "completion_status": (
                    clause.get("completion_status") or "UNKNOWN"
                ),
                "temporal_anchor": deepcopy(
                    clause.get("temporal_anchor")
                ),
                "attribution": (
                    {
                        "speaker": clause.get("speaker"),
                        "quoted_speaker": clause.get("quoted_speaker"),
                    }
                    if clause else None
                ),
                "construction_id": construction["id"],
            },
            "entity_mentions": mention_rows,
            "phrase_graph": {
                "phrases": [
                    {
                        "id": f"phrase-np-{index}",
                        "type": "noun_phrase",
                        "token_start": participant["start"],
                        "token_end": participant["end"],
                        "head_token_index": participant["head"],
                        "tokens": list(range(
                            participant["start"], participant["end"] + 1
                        )),
                        "surface": participant["surface"],
                        "mention_type": participant["mention_type"],
                        "preposition": participant["preposition"],
                        "grammatical_slot": participant["slot"],
                        "role_hypotheses": deepcopy(
                            participant["hypotheses"]
                        ),
                    }
                    for index, participant in enumerate(participant_drafts)
                ] + ([{
                    "id": "phrase-predicate",
                    "type": "verb_phrase",
                    "token_start": predicate.index,
                    "token_end": predicate.index,
                    "head_token_index": predicate.index,
                    "tokens": [predicate.index],
                    "surface": predicate.surface,
                    "lemma": predicate.lemma,
                }] if predicate.index >= 0 else []),
                "dependencies": [
                    {
                        "source": "phrase-predicate",
                        "relation": participant["slot"],
                        "target": f"phrase-np-{index}",
                    }
                    for index, participant in enumerate(participant_drafts)
                    if predicate.index >= 0
                ],
            },
            "role_hypotheses": [
                {
                    "participant_id": participant["id"],
                    "entity_id": participant["entity_id"],
                    "grammatical_slot": participant["slot"],
                    "selected_role": participant["role"],
                    "hypotheses": participant["hypotheses"],
                }
                for participant in participants
            ],
            "construction": construction,
            "concept_update": concept,
            "concept_evidence": [
                {
                    **dict(row),
                    "payload": decode(row["payload_json"], {}),
                }
                for row in conn.execute(
                    """SELECT ce.* FROM concept_evidence ce
                       WHERE ce.source_scene_id=? ORDER BY ce.created_at,ce.id""",
                    (scene_id,),
                ).fetchall()
            ],
            "scene_concept_projection": projection,
            "observation_type": observation_type,
            "morphological_evidence": morphological_links,
        }

    def materialize_clause_events(
        self,
        conn: Any,
        clauses: Sequence[Mapping[str, Any]],
        *,
        scene_id: Optional[int] = None,
        admission_decision_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        projections: List[Dict[str, Any]] = []
        confirmed_event_id: Optional[str] = None
        confirmed_clause_id: Optional[str] = None
        if scene_id is not None and clauses:
            factual = next(
                (
                    clause for clause in clauses
                    if clause.get("actuality") == "ACTUAL"
                    and clause.get("mode") in {
                        "ASSERTION",
                        "DEFINITION",
                        "REPORTED_SPEECH",
                    }
                ),
                None,
            )
            if factual:
                confirmed_clause_id = str(factual.get("id") or "") or None
                materialized = self.materialize_scene(
                    conn,
                    scene_id,
                    factual,
                    admission_decision_id,
                )
                confirmed_event_id = str(
                    materialized.get("event", {}).get("id") or ""
                ) or None
        now = utcnow()
        for clause in clauses:
            predicates = clause.get("predicate_hypotheses") or []
            selected = next(
                (item for item in predicates if item.get("selected")),
                predicates[0] if predicates else {},
            )
            projection = {
                "id": stable_id(
                    "clause-event",
                    clause.get("id"),
                    selected.get("lemma"),
                ),
                "source_clause_id": clause.get("id"),
                "source_scene_id": scene_id,
                "predicate": deepcopy(selected),
                "participants": deepcopy(clause.get("participants") or []),
                "mode": clause.get("mode"),
                "actuality": clause.get("actuality"),
                "evidence_status": clause.get("evidence_status"),
                "polarity": clause.get("polarity"),
                "negation_scope": deepcopy(clause.get("negation_scope")),
                "modality": clause.get("modality"),
                "completion_status": clause.get("completion_status"),
                "temporal_anchor": deepcopy(clause.get("temporal_anchor")),
                "attribution": {
                    "speaker": clause.get("speaker"),
                    "quoted_speaker": clause.get("quoted_speaker"),
                },
                "confirmed_event_id": (
                    confirmed_event_id
                    if clause.get("id") == confirmed_clause_id
                    else None
                ),
                "status": (
                    "CONFIRMED"
                    if clause.get("id") == confirmed_clause_id
                    and confirmed_event_id
                    else "PROVISIONAL"
                ),
            }
            conn.execute(
                """INSERT OR REPLACE INTO clause_event_projections
                   (id,source_clause_id,source_scene_id,confirmed_event_id,
                    event_json,status,parser_version,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    projection["id"],
                    projection["source_clause_id"],
                    scene_id,
                    projection["confirmed_event_id"],
                    encode(projection),
                    projection["status"],
                    PARSER_VERSION,
                    now,
                    now,
                ),
            )
            projections.append(projection)
        return projections

    @staticmethod
    def load_event(conn: Any, scene_id: int) -> Optional[Dict[str, Any]]:
        event = conn.execute(
            "SELECT * FROM events WHERE source_scene_id=?",
            (scene_id,),
        ).fetchone()
        if not event:
            return None
        participants = []
        for row in conn.execute(
            """SELECT ep.*,e.display_name,e.canonical_lemma,
                      em.surface AS mention_surface,em.mention_type,
                      em.entity_type_id,
                      et.display_name AS entity_type_surface,
                      et.canonical_lemma AS entity_type_lemma,
                      em.grammatical_features_json,em.attributes_json,
                      (SELECT ea.lexeme_cloud_id
                         FROM entity_aliases ea
                        WHERE ea.entity_id=ep.entity_id
                          AND ea.lexeme_cloud_id IS NOT NULL
                        ORDER BY ea.confidence DESC,ea.id
                        LIMIT 1) AS lexeme_cloud_id
               FROM event_participants ep
               JOIN entities e ON e.cloud_id=ep.entity_id
               LEFT JOIN entity_mentions em ON em.id=ep.mention_id
               LEFT JOIN entities et ON et.cloud_id=em.entity_type_id
               WHERE ep.event_id=? ORDER BY ep.participant_index""",
            (event["id"],),
        ).fetchall():
            hypotheses = [
                {
                    "role": hypothesis["semantic_role"],
                    "confidence": float(hypothesis["confidence"]),
                    "source_type": hypothesis["source_type"],
                    "evidence": decode(hypothesis["evidence_json"], {}),
                }
                for hypothesis in conn.execute(
                    """SELECT * FROM event_role_hypotheses
                       WHERE participant_id=? ORDER BY confidence DESC,semantic_role""",
                    (row["id"],),
                ).fetchall()
            ]
            participants.append({
                "id": row["id"],
                "entity_id": int(row["entity_id"]),
                "mention_id": row["mention_id"],
                "role": row["semantic_role"],
                "grammatical_slot": row["grammatical_slot"],
                "surface": (
                    row["display_name"]
                    if row["mention_type"] == "apposition"
                    else (
                        str(row["surface"])
                        if decode(
                            row["grammatical_features_json"], {}
                        ).get("proper_name")
                        else str(row["surface"]).casefold()
                    )
                ),
                "mention_surface": row["mention_surface"] or row["surface"],
                "full_surface": row["mention_surface"] or row["surface"],
                "mention_type": row["mention_type"] or "noun_phrase",
                "entity_value": row["display_name"],
                "entity_type": (
                    {
                        "entity_id": int(row["entity_type_id"]),
                        "surface": row["entity_type_surface"],
                        "lemma": row["entity_type_lemma"],
                    }
                    if row["entity_type_id"] is not None
                    else None
                ),
                "answer_surfaces": {
                    "short_name": row["display_name"],
                    "full_mention": row["mention_surface"] or row["surface"],
                    "observed_surface": row["surface"],
                    "canonical_lemma": row["canonical_lemma"] or row["lemma"],
                },
                "lemma": row["canonical_lemma"] or row["lemma"],
                "canonical_lemma": row["canonical_lemma"] or row["lemma"],
                "observed_surface": row["surface"],
                "lexeme_cloud_id": (
                    int(row["lexeme_cloud_id"])
                    if row["lexeme_cloud_id"] is not None
                    else None
                ),
                "preposition": row["preposition"],
                "grammatical_features": decode(
                    row["grammatical_features_json"], {}
                ),
                "attributes": decode(row["attributes_json"], []),
                "confidence": float(row["confidence"]),
                "role_hypotheses": hypotheses,
            })
        modifiers = [
            {
                **dict(row),
                "value_entity_id": (
                    int(row["value_entity_id"])
                    if row["value_entity_id"] is not None
                    else None
                ),
                "attributes": decode(row["attributes_json"], []),
            }
            for row in conn.execute(
                """SELECT * FROM event_modifiers
                   WHERE event_id=? ORDER BY role,id""",
                (event["id"],),
            ).fetchall()
        ]
        return {
            "id": event["id"],
            "source_scene_id": int(event["source_scene_id"]),
            "predicate": {
                "lemma": event["predicate_lemma"],
                "surface": event["predicate_surface"],
                "lexeme_cloud_id": event["predicate_lexeme_cloud_id"],
            },
            "participants": participants,
            "modifiers": modifiers,
            "source_clause_id": event["source_clause_id"],
            "polarity": event["polarity"],
            "modality": event["modality"],
            "actuality": event["actuality"],
            "evidence_status": event["evidence_status"],
            "negation_scope": decode(event["negation_scope_json"], None),
            "completion_status": event["completion_status"],
            "temporal_anchor": decode(event["temporal_anchor_json"], None),
            "attribution": decode(event["attribution_json"], None),
            "admission_decision_id": event["admission_decision_id"],
            "confidence": float(event["confidence"]),
            "construction_id": event["construction_id"],
            "parser_version": event["parser_version"],
        }

    @staticmethod
    def load_mentions(conn: Any, scene_id: int) -> List[Dict[str, Any]]:
        return [
            {
                **dict(row),
                "grammatical_features": decode(row["grammatical_features_json"], {}),
                "attributes": decode(row["attributes_json"], []),
            }
            for row in conn.execute(
                """SELECT em.*,e.canonical_lemma,e.display_name
                   FROM entity_mentions em
                   JOIN entities e ON e.cloud_id=em.entity_id
                   WHERE em.source_scene_id=? ORDER BY em.token_start,em.token_end""",
                (scene_id,),
            ).fetchall()
        ]


def role_compatible(
    requested_role: Optional[str],
    requested_slot: Optional[str],
    participant: Dict[str, Any],
    requested_hypotheses: Iterable[Dict[str, Any]] = (),
) -> float:
    graph = RoleCompatibilityGraph()
    if requested_slot and participant.get("grammatical_slot") == requested_slot:
        return 1.0
    if requested_role and participant.get("role") == requested_role:
        return 1.0
    participant_roles = {
        item.get("role"): float(item.get("confidence", 0.0))
        for item in participant.get("role_hypotheses", [])
    }
    if requested_role in participant_roles:
        return participant_roles[requested_role]
    hypothesis_score = max(
        (
            min(
                float(hypothesis.get("confidence", 0.0)),
                participant_roles.get(hypothesis.get("role"), 0.0),
            )
            for hypothesis in requested_hypotheses
        ),
        default=0.0,
    )
    compatibility_score = max(
        (
            graph.score(str(requested_role or ""), str(role))
            * confidence
            for role, confidence in {
                participant.get("role"): 1.0,
                **participant_roles,
            }.items()
            if role
        ),
        default=0.0,
    )
    return max(hypothesis_score, compatibility_score)
