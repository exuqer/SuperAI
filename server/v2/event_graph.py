"""Materialization of utterances into role-free event graphs."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .graph_learning import (
    ConstructionLearner,
    ObservationBuilder,
    SemanticClusterLearner,
    SlotLearner,
)
from .graph_models import (
    EventNode,
    MentionComponent,
    MentionNode,
    ModelVersions,
    ObservationSignature,
    ParticipantNode,
    PredicateNode,
    SlotHypothesis,
    StructuralEdge,
)
from .graph_repository import (
    GraphRepository,
    content_hash,
    decode,
    encode,
    stable_id,
    utcnow,
)
from .language import UniversalLanguageAnalyzer


class EventGraphPipeline:
    """Persist facts as predicate-centered graphs with unnamed participants."""

    def __init__(
        self,
        repository: Optional[GraphRepository],
        morphology: Any,
    ) -> None:
        self.repository = repository or GraphRepository()
        self.morphology = morphology
        self.language = UniversalLanguageAnalyzer(morphology)
        self.observations = ObservationBuilder()
        self.slots = SlotLearner()
        self.constructions = ConstructionLearner()
        self.semantic_clusters = SemanticClusterLearner()

    @staticmethod
    def _source_status(analysis: Any) -> str:
        unsafe_modes = {
            "QUESTION",
            "REQUEST",
            "COMMAND",
            "CONDITION",
            "COUNTERFACTUAL",
            "HYPOTHESIS",
            "ASSUMPTION",
            "DESIRE",
            "PLAN",
            "QUOTE",
            "REPORTED_SPEECH",
        }
        modes = {
            str(getattr(clause.mode, "value", clause.mode))
            for clause in analysis.clauses
        }
        actuality = {
            str(getattr(clause.actuality, "value", clause.actuality))
            for clause in analysis.clauses
        }
        return (
            "STAGED"
            if modes & unsafe_modes or actuality & {
                "POSSIBLE",
                "HYPOTHETICAL",
                "COUNTERFACTUAL",
                "FICTIONAL",
                "UNKNOWN",
            }
            else "CONFIRMED"
        )

    def _persist_source(
        self,
        conn: Any,
        text: str,
        *,
        source_type: str,
        independent_key: str,
        status: str,
        metadata: Mapping[str, Any],
    ) -> tuple[str, bool, str]:
        digest = content_hash(text)
        normalized = " ".join(text.casefold().split())
        source_id = stable_id("graph-source", digest, independent_key)
        existing = conn.execute(
            "SELECT id FROM knowledge_sources WHERE id=?",
            (source_id,),
        ).fetchone()
        now = utcnow()
        conn.execute(
            """INSERT INTO knowledge_sources
               (id,raw_text,normalized_text,content_hash,source_type,status,
                confidence,independent_key,metadata_json,created_at,updated_at)
               VALUES(?,?,?,?,?,?,.82,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 status=CASE
                   WHEN knowledge_sources.status='RETRACTED'
                   THEN knowledge_sources.status
                   WHEN knowledge_sources.status='CONFIRMED'
                     AND excluded.status='STAGED'
                   THEN knowledge_sources.status
                   ELSE excluded.status
                 END,
                 updated_at=excluded.updated_at""",
            (
                source_id,
                text,
                normalized,
                digest,
                source_type,
                status,
                independent_key,
                encode(dict(metadata)),
                now,
                now,
            ),
        )
        stored = conn.execute(
            "SELECT status FROM knowledge_sources WHERE id=?",
            (source_id,),
        ).fetchone()
        return source_id, not bool(existing), str(stored["status"])

    def _persist_tokens(self, conn: Any, source_id: str, analysis: Any) -> None:
        for token in analysis.tokens:
            token_id = stable_id("graph-token", source_id, token.index)
            selected = next(
                (
                    hypothesis
                    for hypothesis in token.analyses
                    if hypothesis.selected
                ),
                None,
            )
            selected_id = (
                stable_id(
                    "morph-hypothesis",
                    token_id,
                    selected.lemma,
                    selected.pos,
                    sorted(selected.features.items()),
                )
                if selected else None
            )
            conn.execute(
                """INSERT OR IGNORE INTO graph_tokens
                   (id,source_id,token_index,sentence_index,surface,normalized,
                    selected_hypothesis_id,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    token_id,
                    source_id,
                    token.index,
                    int(token.features.get("sentence_index", 0)),
                    token.surface,
                    token.normalized,
                    selected_id,
                    utcnow(),
                ),
            )
            for hypothesis in token.analyses:
                hypothesis_id = stable_id(
                    "morph-hypothesis",
                    token_id,
                    hypothesis.lemma,
                    hypothesis.pos,
                    sorted(hypothesis.features.items()),
                )
                conn.execute(
                    """INSERT OR IGNORE INTO graph_morph_hypotheses
                       (id,token_id,lemma,part_of_speech,features_json,
                        morph_score,selected,evidence_json)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        hypothesis_id,
                        token_id,
                        hypothesis.lemma,
                        hypothesis.pos,
                        encode(hypothesis.features),
                        max(0.0, min(1.0, float(hypothesis.confidence))),
                        int(hypothesis.selected),
                        encode(hypothesis.evidence),
                    ),
                )

    @staticmethod
    def _clause_predicates(analysis: Any) -> List[tuple[Any, Any]]:
        result: List[tuple[Any, Any]] = []
        by_index = {token.index: token for token in analysis.tokens}
        for clause in analysis.clauses:
            candidates = [
                hypothesis
                for hypothesis in clause.predicate_hypotheses
                if not hypothesis.get("embedded")
            ] or list(clause.predicate_hypotheses)
            if len(candidates) > 1:
                lexical = [
                    hypothesis for hypothesis in candidates
                    if str(hypothesis.get("lemma") or "") != "быть"
                ]
                if lexical:
                    candidates = lexical
            for hypothesis in candidates:
                token = by_index.get(int(hypothesis["token_index"]))
                if token and all(token.index != item[1].index for item in result):
                    result.append((clause, token))
        if not result and analysis.predicate:
            result.append((None, analysis.predicate))
        return result

    @staticmethod
    def _mentions_for_clause(
        analysis: Any,
        clause: Any,
        predicate: Any,
    ) -> Sequence[Any]:
        def is_single_sentence(mention: Any) -> bool:
            indices = getattr(mention, "token_indices", ())
            return len({
                int(analysis.tokens[index].features.get("sentence_index", 0))
                for index in indices
            }) == 1

        if clause is None:
            return tuple(
                mention for mention in analysis.mentions
                if is_single_sentence(mention)
            )
        direct = [
            mention for mention in analysis.mentions
            if mention.start >= clause.token_start
            and mention.end <= clause.token_end
            and is_single_sentence(mention)
        ]
        start = clause.token_start if clause is not None else 0
        end = clause.token_end if clause is not None else len(analysis.tokens) - 1
        covered = {
            index
            for mention in direct
            for index in mention.token_indices
        }
        observable_values = [
            SimpleNamespace(
                start=token.index,
                end=token.index,
                head=token.index,
                token_indices=[token.index],
                surface=token.surface,
                normalized_surface=token.normalized,
                lemma=token.lemma,
                features=dict(token.features),
                preposition="",
                attributes=[],
                type_token=None,
                owner_token=None,
                relation_type=None,
                relation_signature=None,
                modifier_token_indices=[],
                confidence=0.68,
            )
            for token in analysis.tokens[start:end + 1]
            if (
                token.index not in covered
                and token.index != predicate.index
                and token.pos in {"ADVB", "PRED", "NUMR"}
                and token.lemma not in {"не", "ещё"}
            )
        ]
        if direct or observable_values:
            return tuple([*direct, *observable_values])
        # Elliptical coordinated clauses can inherit the nearest preceding
        # mention as an observation candidate.  It remains explicit in trace
        # and is not converted into a named semantic type.
        preceding = [
            mention for mention in analysis.mentions
            if mention.end < predicate.index
        ]
        return tuple(preceding[-1:]) if preceding else ()

    @staticmethod
    def _event_source_span(
        analysis: Any,
        clause: Any,
        predicate: Any,
    ) -> tuple[str, int, int, int]:
        """Return the exact token span that licensed one event.

        A source may contain many sentences and clauses.  Its complete text is
        useful provenance, but is never an event realization and must not be
        reused as an answer.
        """
        if clause is not None:
            start = int(clause.token_start)
            end = int(clause.token_end)
        else:
            sentence_index = int(
                predicate.features.get("sentence_index", 0)
            )
            indices = [
                token.index for token in analysis.tokens
                if int(token.features.get("sentence_index", 0))
                == sentence_index
            ]
            start = min(indices, default=predicate.index)
            end = max(indices, default=predicate.index)
        tokens = analysis.tokens[start:end + 1]
        sentence_index = int(
            predicate.features.get(
                "sentence_index",
                tokens[0].features.get("sentence_index", 0) if tokens else 0,
            )
        )
        surface = " ".join(token.surface for token in tokens).strip()
        return surface, start, end, sentence_index

    def _persist_mention(
        self,
        conn: Any,
        source_id: str,
        mention: MentionNode,
        confidence: float,
    ) -> None:
        now = utcnow()
        conn.execute(
            """INSERT INTO graph_entities
               (id,canonical_lemma,display_surface,confidence,metadata_json,
                created_at,updated_at)
               VALUES(?,?,?,?, '{}',?,?)
               ON CONFLICT(id) DO UPDATE SET
                 confidence=MAX(confidence,excluded.confidence),
                 updated_at=excluded.updated_at""",
            (
                mention.entity_id,
                mention.head_lemma,
                mention.head_surface,
                confidence,
                now,
                now,
            ),
        )
        conn.execute(
            """INSERT OR REPLACE INTO graph_mentions
               (id,source_id,entity_id,head_lemma,head_surface,surface,
                qualified_key,token_start,token_end,token_indices_json,
                features_json,components_json,preposition,confidence,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mention.id,
                source_id,
                mention.entity_id,
                mention.head_lemma,
                mention.head_surface,
                mention.surface,
                mention.qualified_key,
                mention.token_start,
                mention.token_end,
                encode(list(mention.token_indices)),
                encode(mention.features),
                encode([
                    component.as_dict() for component in mention.components
                ]),
                mention.preposition,
                confidence,
                now,
            ),
        )
        for component in mention.components:
            conn.execute(
                """INSERT OR IGNORE INTO graph_edges
                   (id,from_node_id,edge_type,to_node_id,evidence_json,
                    confidence,created_at)
                   VALUES(?,?,?,?,?,.9,?)""",
                (
                    stable_id(
                        "graph-edge",
                        mention.id,
                        StructuralEdge.MENTION_HAS_COMPONENT.value,
                        component.id,
                    ),
                    mention.id,
                    StructuralEdge.MENTION_HAS_COMPONENT.value,
                    component.id,
                    encode(component.attachment_signature.as_dict()),
                    now,
                ),
            )

    def materialize(
        self,
        text: str,
        *,
        source_type: str = "document",
        independent_key: str = "",
        domain_key: str = "",
        force_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("text must not be empty")
        analysis = self.language.analyze(
            normalized_text,
            detect_question=True,
            source_type=source_type,
            speaker_role="source",
        )
        inferred_status = self._source_status(analysis)
        if force_status == "STAGED":
            status = "STAGED"
        elif force_status == "CONFIRMED":
            # Manual admission can confirm an assertion, but must never turn a
            # question, command or hypothesis into an independent world fact.
            status = "CONFIRMED" if inferred_status == "CONFIRMED" else "STAGED"
        else:
            status = inferred_status
        independent_key = independent_key or content_hash(normalized_text)
        domain_key = domain_key or source_type
        with self.repository.transaction() as conn:
            source_id, created, status = self._persist_source(
                conn,
                normalized_text,
                source_type=source_type,
                independent_key=independent_key,
                status=status,
                metadata={
                    "interpretation_status": str(
                        getattr(
                            analysis.interpretation_status,
                            "value",
                            analysis.interpretation_status,
                        )
                    ),
                    "interpretation_version": analysis.interpretation_version,
                    "domain_key": domain_key,
                },
            )
            if status == "RETRACTED":
                return {
                    "source_id": source_id,
                    "created": False,
                    "status": status,
                    "events": [],
                    "language_analysis": analysis.as_dict(),
                }
            if not created:
                events = self.load_events_for_source(conn, source_id)
                if events or status != "CONFIRMED":
                    return {
                        "source_id": source_id,
                        "created": False,
                        "status": status,
                        "events": [event.as_dict() for event in events],
                        "language_analysis": analysis.as_dict(),
                    }
            self._persist_tokens(conn, source_id, analysis)
            if status != "CONFIRMED":
                return {
                    "source_id": source_id,
                    "created": True,
                    "status": status,
                    "events": [],
                    "language_analysis": analysis.as_dict(),
                }
            events: List[EventNode] = []
            learned_slots: Dict[str, Dict[str, Any]] = {}
            learned_sets: List[Dict[str, Any]] = []
            learned_prototypes: Dict[str, Dict[str, Any]] = {}
            for clause, predicate_token in self._clause_predicates(analysis):
                predicate_concept_id = stable_id(
                    "predicate-concept",
                    predicate_token.lemma.casefold(),
                )
                clause_key = getattr(clause, "id", "utterance")
                event_id = stable_id(
                    "event",
                    source_id,
                    clause_key,
                    predicate_token.index,
                    predicate_token.lemma,
                )
                base_construction_signature = self.observations.structural_signature(
                    analysis
                )
                construction = self.constructions.observe(
                    conn,
                    base_construction_signature,
                    source_id=source_id,
                    domain_key=domain_key,
                )
                predicate = PredicateNode(
                    lemma=predicate_token.lemma.casefold(),
                    surface=predicate_token.surface,
                    concept_id=predicate_concept_id,
                    token_index=predicate_token.index,
                    features=dict(predicate_token.features),
                )
                clause_mentions = self._mentions_for_clause(
                    analysis,
                    clause,
                    predicate_token,
                )
                participant_drafts: List[ParticipantNode] = []
                for mention_draft in clause_mentions:
                    mention = self.observations.mention_node(
                        analysis,
                        mention_draft,
                        event_id,
                    )
                    signature = self.observations.participant_signature(
                        analysis,
                        mention_draft,
                        predicate_token,
                        construction.id,
                    )
                    participant_drafts.append(ParticipantNode(
                        id=stable_id(
                            "participant",
                            event_id,
                            mention.id,
                        ),
                        mention=mention,
                        observation_signature=signature,
                        confidence=0.68,
                    ))
                polarity = (
                    str(getattr(clause.polarity, "value", clause.polarity))
                    if clause is not None else "POSITIVE"
                )
                actuality = (
                    str(getattr(clause.actuality, "value", clause.actuality))
                    if clause is not None else "ACTUAL"
                )
                properties = []
                if clause is not None:
                    properties = [{
                        "mode": str(
                            getattr(clause.mode, "value", clause.mode)
                        ),
                        "completion": str(
                            getattr(
                                clause.completion_status,
                                "value",
                                clause.completion_status,
                            )
                        ),
                        "modality": (
                            str(getattr(clause.modality, "value", clause.modality))
                            if clause.modality else None
                        ),
                        "negation_scope": clause.negation_scope,
                        "token_start": clause.token_start,
                        "token_end": clause.token_end,
                    }]
                (
                    source_surface,
                    source_token_start,
                    source_token_end,
                    sentence_index,
                ) = self._event_source_span(analysis, clause, predicate_token)
                now = utcnow()
                versions = ModelVersions()
                conn.execute(
                    """INSERT INTO graph_events
                       (id,source_id,predicate_lemma,predicate_concept_id,
                        predicate_surface,predicate_features_json,
                        predicate_token_index,construction_id,polarity,actuality,
                        confidence,properties_json,source_surface,token_start,
                        token_end,sentence_index,event_schema_version,
                        slot_model_version,construction_model_version,
                        semantic_cluster_version,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,.78,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        event_id,
                        source_id,
                        predicate.lemma,
                        predicate.concept_id,
                        predicate.surface,
                        encode(predicate.features),
                        predicate.token_index,
                        construction.id,
                        polarity,
                        actuality,
                        encode(properties),
                        source_surface,
                        source_token_start,
                        source_token_end,
                        sentence_index,
                        versions.event_schema,
                        versions.slot_model,
                        versions.construction_model,
                        versions.semantic_cluster,
                        now,
                        now,
                    ),
                )
                participants: List[ParticipantNode] = []
                selected_slot_ids: List[str] = []
                for ordinal, participant in enumerate(participant_drafts):
                    self._persist_mention(
                        conn,
                        source_id,
                        participant.mention,
                        participant.confidence,
                    )
                    context_signature = ObservationSignature({
                        **participant.observation_signature.as_dict(),
                        f"context:predicate:{predicate_concept_id}": 0.78,
                    })
                    semantic_memberships = self.semantic_clusters.observe(
                        conn,
                        participant.mention.entity_id or stable_id(
                            "entity",
                            participant.mention.head_lemma,
                        ),
                        context_signature,
                    )
                    participant = replace(
                        participant,
                        mention=replace(
                            participant.mention,
                            semantic_cluster_ids=tuple(
                                cluster_id
                                for cluster_id, _ in semantic_memberships
                            ),
                        ),
                        observation_signature=ObservationSignature({
                            **participant.observation_signature.as_dict(),
                            **{
                                f"entity_cluster:{cluster_id}": (
                                    0.68 * compatibility
                                )
                                for cluster_id, compatibility
                                in semantic_memberships
                            },
                        }),
                    )
                    assigned, current_slots = self.slots.assign(
                        conn,
                        predicate_concept_id,
                        participant,
                        domain_key=domain_key,
                        selected_in_event=selected_slot_ids,
                    )
                    if assigned.slot_hypotheses:
                        selected_slot_ids.append(
                            assigned.slot_hypotheses[0].local_slot_id
                        )
                    self._persist_mention(
                        conn,
                        source_id,
                        assigned.mention,
                        assigned.confidence,
                    )
                    conn.execute(
                        """INSERT INTO graph_participants
                           (id,event_id,mention_id,observation_signature_json,
                            confidence,ordinal_hint,created_at)
                           VALUES(?,?,?,?,?,?,?)""",
                        (
                            assigned.id,
                            event_id,
                            assigned.mention.id,
                            encode(assigned.observation_signature.as_dict()),
                            assigned.confidence,
                            ordinal,
                            now,
                        ),
                    )
                    conn.execute(
                        """INSERT OR IGNORE INTO graph_edges
                           (id,from_node_id,edge_type,to_node_id,evidence_json,
                            confidence,created_at)
                           VALUES(?,?,?,?,?,.95,?)""",
                        (
                            stable_id(
                                "graph-edge",
                                event_id,
                                StructuralEdge.EVENT_HAS_PARTICIPANT.value,
                                assigned.id,
                            ),
                            event_id,
                            StructuralEdge.EVENT_HAS_PARTICIPANT.value,
                            assigned.id,
                            encode({
                                "observation_signature": (
                                    assigned.observation_signature.as_dict()
                                ),
                            }),
                            now,
                        ),
                    )
                    self.slots.persist_participant_hypotheses(conn, assigned)
                    participants.append(assigned)
                    for slot in current_slots:
                        learned_slots[slot.id] = slot.as_dict()
                slot_set = self.slots.update_slot_set(
                    conn,
                    predicate_concept_id,
                    selected_slot_ids,
                )
                learned_sets.append(slot_set.as_dict())
                for slot in self.slots.load_local_slots(
                    conn,
                    predicate_concept_id,
                ):
                    prototype = self.slots.generalize(conn, slot)
                    if prototype:
                        learned_prototypes[prototype.id] = prototype.as_dict()
                event = EventNode(
                    id=event_id,
                    predicate=predicate,
                    participants=tuple(participants),
                    properties=tuple(properties),
                    construction_id=construction.id,
                    polarity=polarity,
                    actuality=actuality,
                    confidence=0.78,
                    raw_text=normalized_text,
                    source_surface=source_surface,
                    token_start=source_token_start,
                    token_end=source_token_end,
                    sentence_index=sentence_index,
                    versions=versions,
                )
                events.append(event)
            return {
                "source_id": source_id,
                "created": True,
                "status": status,
                "events": [event.as_dict() for event in events],
                "local_slots": list(learned_slots.values()),
                "slot_sets": learned_sets,
                "slot_prototypes": list(learned_prototypes.values()),
                "language_analysis": analysis.as_dict(),
            }

    @staticmethod
    def load_events_for_source(
        conn: Any,
        source_id: str,
    ) -> List[EventNode]:
        rows = conn.execute(
            """SELECT * FROM graph_events
               WHERE source_id=? ORDER BY predicate_token_index,id""",
            (source_id,),
        ).fetchall()
        return [
            EventGraphPipeline.load_event(conn, str(row["id"]))
            for row in rows
        ]

    @staticmethod
    def load_event(conn: Any, event_id: str) -> EventNode:
        row = conn.execute(
            """SELECT e.*,s.raw_text FROM graph_events e
               JOIN knowledge_sources s ON s.id=e.source_id
               WHERE e.id=?""",
            (event_id,),
        ).fetchone()
        if not row:
            raise KeyError(event_id)
        participant_rows = conn.execute(
            """SELECT p.*,m.*,
                      p.id AS participant_node_id,
                      m.id AS mention_node_id
               FROM graph_participants p
               JOIN graph_mentions m ON m.id=p.mention_id
               WHERE p.event_id=? ORDER BY p.ordinal_hint,p.id""",
            (event_id,),
        ).fetchall()
        participants: List[ParticipantNode] = []
        for participant_row in participant_rows:
            components = [
                # Components are already validated role-free JSON contracts.
                # Rehydrate only the data required by graph matching.
                item
                for item in decode(participant_row["components_json"], [])
            ]
            mention_components = tuple(
                MentionComponent(
                    id=str(item["component_id"]),
                    lemma=str(item["lemma"]),
                    surface=str(item["surface"]),
                    token_index=int(item["token_index"]),
                    attachment_signature=ObservationSignature(
                        item.get("attachment_signature") or {}
                    ),
                    required=bool(item.get("required", True)),
                )
                for item in components
            )
            mention = MentionNode(
                id=str(participant_row["mention_node_id"]),
                head_lemma=str(participant_row["head_lemma"]),
                head_surface=str(participant_row["head_surface"]),
                surface=str(participant_row["surface"]),
                token_start=int(participant_row["token_start"]),
                token_end=int(participant_row["token_end"]),
                token_indices=tuple(
                    decode(participant_row["token_indices_json"], [])
                ),
                features=decode(participant_row["features_json"], {}),
                components=mention_components,
                preposition=str(participant_row["preposition"]),
                entity_id=participant_row["entity_id"],
                semantic_cluster_ids=tuple(
                    str(item["semantic_cluster_id"])
                    for item in conn.execute(
                        """SELECT semantic_cluster_id
                           FROM semantic_cluster_members
                           WHERE entity_id=? ORDER BY compatibility DESC""",
                        (participant_row["entity_id"],),
                    ).fetchall()
                ),
            )
            hypothesis_rows = conn.execute(
                """SELECT local_slot_id,compatibility,evidence_json
                   FROM participant_slot_hypotheses
                   WHERE participant_id=?
                   ORDER BY selected DESC,compatibility DESC""",
                (participant_row["participant_node_id"],),
            ).fetchall()
            participants.append(ParticipantNode(
                id=str(participant_row["participant_node_id"]),
                mention=mention,
                observation_signature=ObservationSignature(
                    decode(
                        participant_row["observation_signature_json"],
                        {},
                    )
                ),
                slot_hypotheses=tuple(
                    SlotHypothesis(
                        local_slot_id=str(item["local_slot_id"]),
                        compatibility=float(item["compatibility"]),
                        evidence=tuple(decode(item["evidence_json"], [])),
                    )
                    for item in hypothesis_rows
                ),
                confidence=float(participant_row["confidence"]),
            ))
        predicate = PredicateNode(
            lemma=str(row["predicate_lemma"]),
            surface=str(row["predicate_surface"]),
            concept_id=str(row["predicate_concept_id"]),
            token_index=int(row["predicate_token_index"]),
            features=decode(row["predicate_features_json"], {}),
        )
        return EventNode(
            id=str(row["id"]),
            predicate=predicate,
            participants=tuple(participants),
            properties=tuple(decode(row["properties_json"], [])),
            construction_id=row["construction_id"],
            polarity=str(row["polarity"]),
            actuality=str(row["actuality"]),
            confidence=float(row["confidence"]),
            raw_text=str(row["raw_text"]),
            source_surface=str(row["source_surface"]),
            token_start=int(row["token_start"]),
            token_end=int(row["token_end"]),
            sentence_index=int(row["sentence_index"]),
            versions=ModelVersions(
                event_schema=str(row["event_schema_version"]),
                slot_model=str(row["slot_model_version"]),
                construction_model=str(row["construction_model_version"]),
                semantic_cluster=str(row["semantic_cluster_version"]),
            ),
        )

    def retract(self, source_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM knowledge_sources WHERE id=?",
                (source_id,),
            ).fetchone()
            if not row:
                raise KeyError(source_id)
            affected_slots = [
                str(item["local_slot_id"])
                for item in conn.execute(
                    """SELECT DISTINCT h.local_slot_id
                       FROM participant_slot_hypotheses h
                       JOIN graph_participants p ON p.id=h.participant_id
                       JOIN graph_events e ON e.id=p.event_id
                       WHERE e.source_id=?""",
                    (source_id,),
                ).fetchall()
            ]
            conn.execute(
                """UPDATE knowledge_sources
                   SET status='RETRACTED',updated_at=? WHERE id=?""",
                (utcnow(), source_id),
            )
            for local_slot_id in affected_slots:
                self.slots.weaken(conn, local_slot_id)
            return {
                "source_id": source_id,
                "status": "RETRACTED",
                "recalculated_slot_ids": affected_slots,
            }
