"""Observation extraction and unsupervised slot/construction learning."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .graph_models import (
    CONSTRUCTION_MODEL_VERSION,
    SEMANTIC_CLUSTER_VERSION,
    SLOT_MODEL_VERSION,
    ConstructionCluster,
    GapKind,
    LocalSlot,
    MentionComponent,
    MentionNode,
    ObservationSignature,
    ParticipantNode,
    SlotHypothesis,
    SlotPrototype,
    SlotSet,
    SlotStatus,
)
from .graph_repository import decode, encode, stable_id, utcnow
from .language.noun_phrase_parser import NOUN_POS


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def signature_similarity(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    """Cosine overlap with a dispersion penalty.

    A single shared case, position or entity cluster cannot produce a high
    score by itself.  The overlap must be supported by several independent
    feature namespaces.
    """
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    dot = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    cosine = dot / (left_norm * right_norm)
    shared_namespaces = {
        key.split(":", 1)[0]
        for key in set(left) & set(right)
        if left.get(key, 0.0) and right.get(key, 0.0)
    }
    diversity = min(1.0, len(shared_namespaces) / 4.0)
    key_overlap = len(set(left) & set(right)) / max(1, len(keys))
    return clamp(cosine * (0.52 + 0.30 * diversity + 0.18 * key_overlap))


def merge_centroid(
    centroid: Mapping[str, float],
    observation: Mapping[str, float],
    support_count: int,
) -> Dict[str, float]:
    old_weight = max(0, int(support_count))
    total = old_weight + 1
    keys = set(centroid) | set(observation)
    return {
        key: clamp(
            (
                old_weight * float(centroid.get(key, 0.0))
                + float(observation.get(key, 0.0))
            )
            / total
        )
        for key in keys
        if (
            old_weight * float(centroid.get(key, 0.0))
            + float(observation.get(key, 0.0))
        )
        / total
        >= 0.03
    }


class ObservationBuilder:
    """Build sparse signatures from directly observable language features."""

    @staticmethod
    def mention_node(analysis: Any, mention: Any, namespace: str) -> MentionNode:
        tokens = analysis.tokens
        components: List[MentionComponent] = []
        excluded_indices: set[int] = set()
        for token_index in mention.modifier_token_indices:
            token = tokens[token_index]
            if (
                token.lemma == "какой"
                or (
                    analysis.predicate is not None
                    and token.index == analysis.predicate.index
                )
            ):
                excluded_indices.add(token.index)
                continue
            attachment: Dict[str, float] = {}
            head = tokens[mention.head]
            for feature in ("case", "number", "gender"):
                left = token.features.get(feature)
                right = head.features.get(feature)
                if left and right and left == right:
                    attachment[f"agreement:{feature}"] = 1.0
            attachment[
                "position:before_head"
                if token.index < mention.head
                else "position:after_head"
            ] = 0.9
            components.append(MentionComponent(
                id=stable_id("component", namespace, token.index, token.lemma),
                lemma=token.lemma,
                surface=token.surface,
                token_index=token.index,
                attachment_signature=ObservationSignature(attachment),
            ))
        # A following genitive noun is not an adjective-like modifier.  It is
        # nevertheless a visible component of the maximal noun phrase and
        # must survive event persistence (``кусочки помидора``).
        if mention.owner_token is not None:
            owner = tokens[int(mention.owner_token)]
            components.append(MentionComponent(
                id=stable_id("component", namespace, owner.index, owner.lemma),
                lemma=owner.lemma.casefold(),
                surface=owner.surface,
                token_index=owner.index,
                attachment_signature=ObservationSignature({
                    "construction:noun_genitive_dependency": 0.96,
                    "position:after_head": 0.90,
                }),
                grammatical_features=dict(owner.features),
                evidence=(
                    "noun_genitive_dependency",
                    "adjacent_noun_phrase",
                ),
                confidence=float(getattr(mention, "confidence", 0.82)),
            ))
        head = tokens[mention.head]
        entity_id = stable_id("entity", head.lemma.casefold())
        retained_indices = tuple(
            index for index in mention.token_indices
            if index not in excluded_indices
        )
        surface = " ".join(
            tokens[index].surface for index in retained_indices
        ) or head.surface
        return MentionNode(
            id=stable_id(
                "mention",
                namespace,
                mention.start,
                mention.end,
                mention.lemma,
            ),
            head_lemma=head.lemma.casefold(),
            head_surface=head.surface,
            surface=surface,
            token_start=min(retained_indices, default=mention.head),
            token_end=max(retained_indices, default=mention.head),
            token_indices=retained_indices,
            features={
                **dict(mention.features),
                # Retain the local alternatives so an eventual rejected-event
                # diagnostic can show what morphology was discarded.
                "morphology_alternatives": [
                    {
                        "lemma": item.lemma.casefold(),
                        "case": item.features.get("case"),
                        "confidence": float(item.confidence),
                        "selected": bool(item.selected),
                    }
                    for item in head.analyses
                    if item.pos in NOUN_POS
                ],
                "preposition_support": (
                    mention.preposition.casefold()
                    if mention.preposition else ""
                ),
            },
            components=tuple(components),
            preposition=mention.preposition.casefold(),
            entity_id=entity_id,
        )

    @staticmethod
    def participant_signature(
        analysis: Any,
        mention: Any,
        predicate: Any,
        construction_id: str = "",
        semantic_cluster_ids: Sequence[str] = (),
    ) -> ObservationSignature:
        head = analysis.tokens[mention.head]
        values: Dict[str, float] = {}
        selected = next(
            (item for item in head.analyses if item.selected),
            head.analyses[0] if head.analyses else None,
        )
        morph_strength = clamp(
            float(getattr(selected, "confidence", 0.7)) if selected else 0.7
        )
        for feature in ("case", "number", "gender", "animacy"):
            value = head.features.get(feature)
            if value:
                values[f"morph:{feature}:{value}"] = morph_strength
        for hypothesis in head.analyses:
            hypothesis_strength = clamp(float(hypothesis.confidence))
            for feature in ("case", "number", "gender", "animacy"):
                value = hypothesis.features.get(feature)
                if value:
                    key = f"morph:{feature}:{value}"
                    values[key] = max(
                        values.get(key, 0.0),
                        0.72 * hypothesis_strength,
                    )
        if head.pos:
            values[f"morph:part_of_speech:{head.pos}"] = 0.92
        if mention.preposition:
            normalized = mention.preposition.casefold().replace(" ", "_")
            values[f"preposition:{normalized}"] = 0.95
        if mention.end < predicate.index:
            values["position:before_predicate"] = 0.82
        elif mention.start > predicate.index:
            values["position:after_predicate"] = 0.82
        else:
            values["position:overlaps_predicate"] = 0.7
        distance = min(
            abs(int(mention.head) - int(predicate.index)),
            8,
        )
        values[f"distance:predicate:{distance}"] = 0.68
        for feature in ("number", "gender"):
            participant_value = head.features.get(feature)
            predicate_value = predicate.features.get(feature)
            if participant_value and predicate_value:
                key = (
                    f"agreement:predicate_{feature}"
                    if participant_value == predicate_value
                    else f"disagreement:predicate_{feature}"
                )
                values[key] = 0.89 if participant_value == predicate_value else 0.72
        if construction_id:
            values[f"construction:{construction_id}"] = 0.74
        for cluster_id in semantic_cluster_ids:
            values[f"entity_cluster:{cluster_id}"] = 0.68
        if any(
            token.normalized.casefold() in {"не", "ни"}
            for token in analysis.tokens
        ):
            values["polarity:negative_surface"] = 0.92
        return ObservationSignature(values)

    @staticmethod
    def structural_signature(
        analysis: Any,
        *,
        gap_kind: Optional[GapKind] = None,
    ) -> ObservationSignature:
        predicate = analysis.predicate
        values: Dict[str, float] = {}
        content_tokens = [
            token for token in analysis.tokens
            if token.pos != "PNCT"
        ]
        values[f"shape:token_count:{min(len(content_tokens), 12)}"] = 1.0
        values[f"shape:mention_count:{min(len(analysis.mentions), 8)}"] = 1.0
        if predicate:
            values[
                f"shape:predicate_position:{min(predicate.index, 8)}"
            ] = 0.86
            values[f"shape:predicate_form:{predicate.pos}"] = 0.9
        question = analysis.question_operator
        if question:
            question_index = question.token_indices[0]
            question_token = analysis.tokens[question_index]
            values[
                f"question:position:{min(question_index, 8)}"
            ] = 0.9
            values[
                f"question:part_of_speech:{question_token.pos}"
            ] = 0.92
            for feature in ("case", "number", "gender"):
                value = question_token.features.get(feature)
                if value:
                    values[f"question:{feature}:{value}"] = 0.94
            values[
                "question:typed_modifier"
                if question.type_constraint_token_index is not None
                else "question:standalone"
            ] = 1.0
        if any(token.pos in {"PRTS", "PRTF"} for token in analysis.tokens):
            values["voice:participle_surface"] = 0.92
        if any(
            token.lemma == "быть" and token.pos in {"VERB", "INFN"}
            for token in analysis.tokens
        ):
            values["voice:auxiliary_surface"] = 0.82
        if gap_kind:
            values[f"gap_kind:{gap_kind.value}"] = 1.0
        return ObservationSignature(values)

    @staticmethod
    def question_signature(
        analysis: Any,
        question: Optional[Any] = None,
    ) -> ObservationSignature:
        question = question or analysis.question_operator
        if not question:
            return ObservationSignature({})
        token = analysis.tokens[question.token_indices[0]]
        predicate = analysis.predicate
        values: Dict[str, float] = {
            f"question:surface:{token.normalized.casefold()}": 0.7,
            f"question:part_of_speech:{token.pos}": 0.95,
        }
        ambiguous_cases = {
            str(hypothesis.features.get("case"))
            for hypothesis in token.analyses
            if hypothesis.pos == "NPRO"
            and hypothesis.features.get("case") in {"nomn", "accs"}
        }
        # For ``какой <noun>`` the modifier and head are chosen jointly, but
        # a masculine/neuter inanimate noun may still be surface-syncretic in
        # nominative/accusative.  Preserve those compatible pair alternatives
        # instead of pretending the selected modifier form resolved the role.
        typed_cases: set[str] = set()
        if question.type_constraint_token_index is not None:
            head = analysis.tokens[int(question.type_constraint_token_index)]
            for hypothesis in token.analyses:
                if hypothesis.pos != "ADJF":
                    continue
                if hypothesis.features.get("case") not in {"nomn", "accs"}:
                    continue
                if all(
                    not hypothesis.features.get(feature)
                    or not head.features.get(feature)
                    or hypothesis.features.get(feature) == head.features.get(feature)
                    for feature in ("gender", "number")
                ):
                    typed_cases.add(str(hypothesis.features["case"]))
        for feature in ("case", "number", "gender", "animacy"):
            if feature == "case" and (
                ambiguous_cases == {"nomn", "accs"}
                or typed_cases == {"nomn", "accs"}
            ):
                for case in sorted(typed_cases or ambiguous_cases):
                    values[f"morph:case:{case}"] = 0.93
                continue
            value = token.features.get(feature)
            if value:
                values[f"morph:{feature}:{value}"] = 0.93
        for hypothesis in token.analyses:
            if (
                question.type_constraint_token_index is not None
                and hypothesis.pos == "ADJF"
                and not all(
                    not hypothesis.features.get(feature)
                    or not head.features.get(feature)
                    or hypothesis.features.get(feature) == head.features.get(feature)
                    for feature in ("gender", "number")
                )
            ):
                # Rejected modifier/head pair; retain it in morphology
                # diagnostics but never turn it into a gap constraint.
                continue
            hypothesis_strength = clamp(float(hypothesis.confidence))
            for feature in ("case", "number", "gender", "animacy"):
                if feature == "case" and (
                    ambiguous_cases == {"nomn", "accs"}
                    or typed_cases == {"nomn", "accs"}
                ):
                    continue
                value = hypothesis.features.get(feature)
                if value:
                    key = f"morph:{feature}:{value}"
                    values[key] = max(
                        values.get(key, 0.0),
                        0.82 * hypothesis_strength,
                    )
        if ambiguous_cases == {"nomn", "accs"}:
            # Treat the two forms as actual alternatives, not as a selected
            # nominative plus an almost discarded accusative parser reading.
            # A gender mismatch in a past-tense predicate is evidence for an
            # omitted agreeing participant, so it can weaken nominative; it
            # never promotes nominative merely because the verb is inflected.
            case_weights = {"nomn": 0.72, "accs": 0.72}
            predicate_gender = (
                str(predicate.features.get("gender") or "")
                if predicate else ""
            )
            question_gender = str(token.features.get("gender") or "")
            if (
                predicate
                and str(predicate.features.get("tense") or "") == "past"
                and predicate_gender
                and question_gender
                and predicate_gender != question_gender
            ):
                case_weights = {"nomn": 0.30, "accs": 0.92}
            for case, weight in case_weights.items():
                values[f"morph:case:{case}"] = weight
        if predicate:
            if token.index < predicate.index:
                values["position:before_predicate"] = 0.76
            elif token.index > predicate.index:
                values["position:after_predicate"] = 0.76
            distance = min(abs(int(token.index) - int(predicate.index)), 8)
            values[f"distance:predicate:{distance}"] = 0.64
            # Do not encode verb/question agreement as if it described the
            # requested gap.  It may describe an unexpressed participant.
        previous = analysis.tokens[token.index - 1] if token.index else None
        if previous and previous.pos == "PREP":
            values[f"preposition:{previous.normalized.casefold()}"] = 0.94
        values[
            "question:typed_modifier"
            if question.type_constraint_token_index is not None
            else "question:standalone"
        ] = 1.0
        return ObservationSignature(values)


class SlotLearner:
    """Soft local-slot clustering with explicit anti-collapse constraints."""

    ASSIGN_THRESHOLD = 0.48
    ALTERNATIVE_MARGIN = 0.12
    PROTOTYPE_THRESHOLD = 0.64

    @staticmethod
    def _status(
        support_count: int,
        contradiction_count: int,
        domain_diversity: int,
        *,
        generalized: bool = False,
    ) -> SlotStatus:
        if support_count and contradiction_count / support_count >= 0.7:
            return SlotStatus.WEAKENED
        if generalized:
            return SlotStatus.GENERALIZED
        if support_count >= 5 and domain_diversity >= 2:
            return SlotStatus.STABLE
        if support_count >= 2:
            return SlotStatus.LOCAL
        return SlotStatus.CANDIDATE

    @staticmethod
    def _confidence(
        support_count: int,
        contradiction_count: int,
        domain_diversity: int,
        dispersion: float = 0.0,
    ) -> float:
        support = 1.0 - math.exp(-max(0, support_count) / 4.0)
        diversity = min(1.0, max(0, domain_diversity) / 3.0)
        contradictions = contradiction_count / max(
            1,
            support_count + contradiction_count,
        )
        return clamp(
            0.16 + 0.58 * support + 0.20 * diversity
            - 0.42 * contradictions - 0.28 * clamp(dispersion)
        )

    def load_local_slots(
        self,
        conn: Any,
        predicate_concept_id: str,
    ) -> List[LocalSlot]:
        rows = conn.execute(
            """SELECT * FROM local_slots
               WHERE predicate_concept_id=? AND status<>'DEPRECATED'
               ORDER BY confidence DESC,id""",
            (predicate_concept_id,),
        ).fetchall()
        return [
            LocalSlot(
                id=str(row["id"]),
                predicate_concept_id=str(row["predicate_concept_id"]),
                centroid_signature=ObservationSignature(
                    decode(row["centroid_signature_json"], {})
                ),
                support_count=int(row["support_count"]),
                contradiction_count=int(row["contradiction_count"]),
                domain_diversity=int(row["domain_diversity"]),
                confidence=float(row["confidence"]),
                status=SlotStatus(str(row["status"])),
                display_label=row["display_label"],
            )
            for row in rows
        ]

    @staticmethod
    def _new_slot_id(
        predicate_concept_id: str,
        signature: Mapping[str, float],
    ) -> str:
        # The id is based on a structural fingerprint, never participant order.
        structural = sorted(
            (
                key,
                round(float(value), 1),
            )
            for key, value in signature.items()
            if key.split(":", 1)[0] in {
                "morph",
                "preposition",
                "agreement",
                "disagreement",
            }
            and value >= 0.55
        )
        # Linear position remains an observation in the centroid, but it is
        # not an identity criterion for an unnamed local slot.  Russian free
        # word order can move the same attachment from either side of the
        # predicate without changing its function in the event.
        return stable_id("local-slot", predicate_concept_id, structural)

    def assign(
        self,
        conn: Any,
        predicate_concept_id: str,
        participant: ParticipantNode,
        *,
        domain_key: str,
        selected_in_event: Iterable[str] = (),
    ) -> Tuple[ParticipantNode, List[LocalSlot]]:
        existing = self.load_local_slots(conn, predicate_concept_id)
        already_selected = set(selected_in_event)
        scored: List[Tuple[float, LocalSlot]] = []
        for slot in existing:
            similarity = signature_similarity(
                participant.observation_signature.values,
                slot.centroid_signature.values,
            )
            # Multiple participants in the same event should not collapse into
            # one local cluster merely because they share case or entity type.
            if slot.id in already_selected:
                similarity *= 0.72
            scored.append((similarity, slot))
        scored.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
        available = [
            item for item in scored if item[1].id not in already_selected
        ]
        winner = (
            available[0]
            if available and available[0][0] >= self.ASSIGN_THRESHOLD
            else None
        )
        if winner is None:
            slot_id = self._new_slot_id(
                predicate_concept_id,
                participant.observation_signature.values,
            )
            if slot_id in already_selected:
                slot_id = stable_id(
                    "local-slot",
                    predicate_concept_id,
                    sorted(
                        participant.observation_signature.values.items()
                    ),
                    "cooccurs-with",
                    sorted(already_selected),
                )
            row = conn.execute(
                "SELECT * FROM local_slots WHERE id=?",
                (slot_id,),
            ).fetchone()
            if row:
                existing_slot = next(
                    slot for slot in existing if slot.id == slot_id
                )
                winner = (
                    signature_similarity(
                        participant.observation_signature.values,
                        existing_slot.centroid_signature.values,
                    ),
                    existing_slot,
                )
            else:
                now = utcnow()
                conn.execute(
                    """INSERT INTO local_slots
                       (id,predicate_concept_id,centroid_signature_json,
                        support_count,contradiction_count,domain_diversity,
                        confidence,status,display_label,slot_model_version,
                        created_at,updated_at)
                       VALUES(?,?,?,0,0,0,0,'CANDIDATE',NULL,?,?,?)""",
                    (
                        slot_id,
                        predicate_concept_id,
                        encode(participant.observation_signature.as_dict()),
                        SLOT_MODEL_VERSION,
                        now,
                        now,
                    ),
                )
                created = LocalSlot(
                    id=slot_id,
                    predicate_concept_id=predicate_concept_id,
                    centroid_signature=participant.observation_signature,
                    support_count=0,
                    contradiction_count=0,
                    domain_diversity=0,
                    confidence=0.0,
                    status=SlotStatus.CANDIDATE,
                )
                winner = (1.0, created)
                existing.append(created)
        assert winner is not None
        winner_score, winner_slot = winner
        alternatives = [
            (score, slot)
            for score, slot in scored
            if (
                slot.id != winner_slot.id
                and score >= self.ASSIGN_THRESHOLD
                and winner_score - score <= self.ALTERNATIVE_MARGIN
            )
        ]
        centroid = merge_centroid(
            winner_slot.centroid_signature.values,
            participant.observation_signature.values,
            winner_slot.support_count,
        )
        support_count = winner_slot.support_count + 1
        conn.execute(
            """INSERT INTO local_slot_domains
               (local_slot_id,domain_key,observation_count)
               VALUES(?,?,1)
               ON CONFLICT(local_slot_id,domain_key)
               DO UPDATE SET observation_count=observation_count+1""",
            (winner_slot.id, domain_key),
        )
        domain_diversity = int(conn.execute(
            "SELECT COUNT(*) FROM local_slot_domains WHERE local_slot_id=?",
            (winner_slot.id,),
        ).fetchone()[0])
        dispersion = 1.0 - signature_similarity(
            centroid,
            participant.observation_signature.values,
        )
        status = self._status(
            support_count,
            winner_slot.contradiction_count,
            domain_diversity,
        )
        confidence = self._confidence(
            support_count,
            winner_slot.contradiction_count,
            domain_diversity,
            dispersion,
        )
        conn.execute(
            """UPDATE local_slots
               SET centroid_signature_json=?,support_count=?,
                   domain_diversity=?,confidence=?,status=?,updated_at=?
               WHERE id=?""",
            (
                encode(centroid),
                support_count,
                domain_diversity,
                confidence,
                status.value,
                utcnow(),
                winner_slot.id,
            ),
        )
        hypotheses = [
            SlotHypothesis(
                local_slot_id=winner_slot.id,
                compatibility=max(winner_score, confidence),
                evidence=(
                    "multi_feature_signature_similarity",
                    "predicate_local_recurrence",
                ),
            ),
            *[
                SlotHypothesis(
                    local_slot_id=slot.id,
                    compatibility=score,
                    evidence=("preserved_competing_cluster",),
                )
                for score, slot in alternatives
            ],
        ]
        updated_participant = replace(
            participant,
            slot_hypotheses=tuple(hypotheses),
            confidence=max(participant.confidence, confidence),
        )
        updated_slots = self.load_local_slots(conn, predicate_concept_id)
        return updated_participant, updated_slots

    def persist_participant_hypotheses(
        self,
        conn: Any,
        participant: ParticipantNode,
    ) -> None:
        for index, hypothesis in enumerate(participant.slot_hypotheses):
            conn.execute(
                """INSERT INTO participant_slot_hypotheses
                   (participant_id,local_slot_id,compatibility,selected,
                    evidence_json,created_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(participant_id,local_slot_id)
                   DO UPDATE SET compatibility=excluded.compatibility,
                                 selected=excluded.selected,
                                 evidence_json=excluded.evidence_json""",
                (
                    participant.id,
                    hypothesis.local_slot_id,
                    hypothesis.compatibility,
                    int(index == 0),
                    encode(list(hypothesis.evidence)),
                    utcnow(),
                ),
            )

    def update_slot_set(
        self,
        conn: Any,
        predicate_concept_id: str,
        local_slot_ids: Sequence[str],
    ) -> SlotSet:
        unique_ids = tuple(sorted(set(local_slot_ids)))
        slot_set_id = stable_id("slot-set", predicate_concept_id, unique_ids)
        now = utcnow()
        conn.execute(
            """INSERT INTO slot_sets
               (id,predicate_concept_id,support_count,confidence,status,
                created_at,updated_at)
               VALUES(?,?,1,.24,'CANDIDATE',?,?)
               ON CONFLICT(id) DO UPDATE SET
                 support_count=slot_sets.support_count+1,
                 updated_at=excluded.updated_at""",
            (slot_set_id, predicate_concept_id, now, now),
        )
        for local_slot_id in unique_ids:
            conn.execute(
                """INSERT OR IGNORE INTO slot_set_members
                   (slot_set_id,local_slot_id) VALUES(?,?)""",
                (slot_set_id, local_slot_id),
            )
        row = conn.execute(
            "SELECT * FROM slot_sets WHERE id=?",
            (slot_set_id,),
        ).fetchone()
        support_count = int(row["support_count"])
        confidence = clamp(0.12 + 0.18 * math.log1p(support_count))
        status = self._status(support_count, 0, 1)
        conn.execute(
            "UPDATE slot_sets SET confidence=?,status=? WHERE id=?",
            (confidence, status.value, slot_set_id),
        )
        return SlotSet(
            id=slot_set_id,
            predicate_concept_id=predicate_concept_id,
            local_slot_ids=unique_ids,
            support_count=support_count,
            confidence=confidence,
            status=status,
        )

    def generalize(self, conn: Any, local_slot: LocalSlot) -> Optional[SlotPrototype]:
        if local_slot.status not in {
            SlotStatus.STABLE,
            SlotStatus.GENERALIZED,
        }:
            return None
        rows = conn.execute(
            """SELECT p.*,m.local_slot_id,m.compatibility,
                      s.predicate_concept_id
               FROM slot_prototypes p
               JOIN slot_prototype_members m ON m.prototype_id=p.id
               JOIN local_slots s ON s.id=m.local_slot_id
               WHERE s.predicate_concept_id<>?
               ORDER BY p.confidence DESC""",
            (local_slot.predicate_concept_id,),
        ).fetchall()
        candidates: Dict[str, Tuple[float, Any]] = {}
        for row in rows:
            prototype_id = str(row["id"])
            score = signature_similarity(
                local_slot.centroid_signature.values,
                decode(row["centroid_signature_json"], {}),
            )
            if (
                prototype_id not in candidates
                or score > candidates[prototype_id][0]
            ):
                candidates[prototype_id] = (score, row)
        winner = max(candidates.values(), default=None, key=lambda item: item[0])
        if winner and winner[0] >= self.PROTOTYPE_THRESHOLD:
            score, row = winner
            prototype_id = str(row["id"])
        else:
            score = 1.0
            prototype_id = stable_id(
                "slot-prototype",
                local_slot.id,
                local_slot.predicate_concept_id,
            )
        now = utcnow()
        conn.execute(
            """INSERT INTO slot_prototypes
               (id,centroid_signature_json,support_count,domain_diversity,
                confidence,display_label,slot_model_version,created_at,updated_at)
               VALUES(?,'{}',0,0,0,NULL,?,?,?)
               ON CONFLICT(id) DO NOTHING""",
            (
                prototype_id,
                SLOT_MODEL_VERSION,
                now,
                now,
            ),
        )
        conn.execute(
            """INSERT INTO slot_prototype_members
               (prototype_id,local_slot_id,compatibility)
               VALUES(?,?,?)
               ON CONFLICT(prototype_id,local_slot_id)
               DO UPDATE SET compatibility=MAX(compatibility,excluded.compatibility)""",
            (prototype_id, local_slot.id, score),
        )
        member_slots = conn.execute(
            """SELECT s.id,s.predicate_concept_id,
                      s.centroid_signature_json,s.support_count
               FROM slot_prototype_members m
               JOIN local_slots s ON s.id=m.local_slot_id
               WHERE m.prototype_id=? AND s.status<>'DEPRECATED'
               ORDER BY s.id""",
            (prototype_id,),
        ).fetchall()
        support_count = sum(
            max(0, int(item["support_count"])) for item in member_slots
        )
        weighted: Dict[str, float] = {}
        total_weight = 0
        for item in member_slots:
            weight = max(1, int(item["support_count"]))
            total_weight += weight
            signature = decode(item["centroid_signature_json"], {})
            for key, value in signature.items():
                weighted[key] = (
                    weighted.get(key, 0.0) + weight * float(value)
                )
        centroid = {
            key: clamp(value / max(1, total_weight))
            for key, value in weighted.items()
            if value / max(1, total_weight) >= 0.03
        }
        domain_diversity = int(conn.execute(
            """SELECT COUNT(DISTINCT d.domain_key)
               FROM slot_prototype_members m
               JOIN local_slot_domains d ON d.local_slot_id=m.local_slot_id
               WHERE m.prototype_id=?""",
            (prototype_id,),
        ).fetchone()[0])
        confidence = self._confidence(support_count, 0, domain_diversity)
        conn.execute(
            """UPDATE slot_prototypes
               SET centroid_signature_json=?,support_count=?,
                   domain_diversity=?,confidence=?,updated_at=? WHERE id=?""",
            (
                encode(centroid),
                support_count,
                domain_diversity,
                confidence,
                now,
                prototype_id,
            ),
        )
        predicate_diversity = len({
            str(item["predicate_concept_id"]) for item in member_slots
        })
        if predicate_diversity >= 2:
            conn.execute(
                """UPDATE local_slots SET status='GENERALIZED',updated_at=?
                   WHERE id IN (
                     SELECT local_slot_id FROM slot_prototype_members
                     WHERE prototype_id=?
                   ) AND status IN ('STABLE','GENERALIZED')""",
                (now, prototype_id),
            )
        return SlotPrototype(
            id=prototype_id,
            member_slot_ids=tuple(
                str(item["id"]) for item in member_slots
            ),
            centroid_signature=ObservationSignature(centroid),
            support_count=support_count,
            domain_diversity=domain_diversity,
            confidence=confidence,
        )

    def weaken(
        self,
        conn: Any,
        local_slot_id: str,
        *,
        amount: int = 1,
    ) -> LocalSlot:
        conn.execute(
            """UPDATE local_slots
               SET contradiction_count=contradiction_count+?,updated_at=?
               WHERE id=?""",
            (max(1, int(amount)), utcnow(), local_slot_id),
        )
        row = conn.execute(
            "SELECT * FROM local_slots WHERE id=?",
            (local_slot_id,),
        ).fetchone()
        if not row:
            raise KeyError(local_slot_id)
        status = self._status(
            int(row["support_count"]),
            int(row["contradiction_count"]),
            int(row["domain_diversity"]),
        )
        confidence = self._confidence(
            int(row["support_count"]),
            int(row["contradiction_count"]),
            int(row["domain_diversity"]),
        )
        conn.execute(
            "UPDATE local_slots SET status=?,confidence=? WHERE id=?",
            (status.value, confidence, local_slot_id),
        )
        return next(
            slot
            for slot in self.load_local_slots(
                conn,
                str(row["predicate_concept_id"]),
            )
            if slot.id == local_slot_id
        )

    def split_slot(
        self,
        conn: Any,
        local_slot_id: str,
        signatures: Sequence[Mapping[str, float]],
    ) -> Sequence[str]:
        if len(signatures) < 2:
            raise ValueError("slot split requires at least two observations")
        row = conn.execute(
            "SELECT predicate_concept_id FROM local_slots WHERE id=?",
            (local_slot_id,),
        ).fetchone()
        if not row:
            raise KeyError(local_slot_id)
        predicate_concept_id = str(row["predicate_concept_id"])
        created: List[str] = []
        for index, signature in enumerate(signatures):
            slot_id = stable_id(
                "local-slot",
                predicate_concept_id,
                "split",
                local_slot_id,
                index,
                sorted(signature.items()),
            )
            if slot_id in created:
                continue
            now = utcnow()
            conn.execute(
                """INSERT OR IGNORE INTO local_slots
                   (id,predicate_concept_id,centroid_signature_json,
                    support_count,contradiction_count,domain_diversity,
                    confidence,status,display_label,slot_model_version,
                    created_at,updated_at)
                   VALUES(?,?,?,1,0,1,.24,'CANDIDATE',NULL,?,?,?)""",
                (
                    slot_id,
                    predicate_concept_id,
                    encode(dict(signature)),
                    SLOT_MODEL_VERSION,
                    now,
                    now,
                ),
            )
            created.append(slot_id)
        if len(created) < 2:
            raise ValueError("split observations do not form distinct signatures")
        conn.execute(
            "UPDATE local_slots SET status='DEPRECATED',updated_at=? WHERE id=?",
            (utcnow(), local_slot_id),
        )
        return tuple(created)

    def merge_slots(
        self,
        conn: Any,
        local_slot_ids: Sequence[str],
    ) -> LocalSlot:
        unique_ids = tuple(sorted(set(local_slot_ids)))
        if len(unique_ids) < 2:
            raise ValueError("slot merge requires at least two local slots")
        marks = ",".join("?" for _ in unique_ids)
        rows = conn.execute(
            f"""SELECT * FROM local_slots
                WHERE id IN ({marks}) ORDER BY id""",
            unique_ids,
        ).fetchall()
        if len(rows) != len(unique_ids):
            raise KeyError("one or more local slots do not exist")
        predicates = {
            str(row["predicate_concept_id"]) for row in rows
        }
        if len(predicates) != 1:
            raise ValueError(
                "local slots from different predicates require a prototype"
            )
        predicate_concept_id = predicates.pop()
        total_support = sum(int(row["support_count"]) for row in rows)
        total_contradictions = sum(
            int(row["contradiction_count"]) for row in rows
        )
        centroid: Dict[str, float] = {}
        consumed = 0
        for row in rows:
            signature = decode(row["centroid_signature_json"], {})
            weight = max(1, int(row["support_count"]))
            for _ in range(weight):
                centroid = merge_centroid(centroid, signature, consumed)
                consumed += 1
        merged_id = stable_id(
            "local-slot",
            predicate_concept_id,
            "merged",
            unique_ids,
        )
        domains = conn.execute(
            f"""SELECT domain_key,SUM(observation_count) AS observations
                FROM local_slot_domains
                WHERE local_slot_id IN ({marks})
                GROUP BY domain_key""",
            unique_ids,
        ).fetchall()
        domain_diversity = len(domains)
        status = self._status(
            total_support,
            total_contradictions,
            domain_diversity,
        )
        confidence = self._confidence(
            total_support,
            total_contradictions,
            domain_diversity,
        )
        now = utcnow()
        conn.execute(
            """INSERT INTO local_slots
               (id,predicate_concept_id,centroid_signature_json,support_count,
                contradiction_count,domain_diversity,confidence,status,
                display_label,slot_model_version,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,NULL,?,?,?)""",
            (
                merged_id,
                predicate_concept_id,
                encode(centroid),
                total_support,
                total_contradictions,
                domain_diversity,
                confidence,
                status.value,
                SLOT_MODEL_VERSION,
                now,
                now,
            ),
        )
        for domain in domains:
            conn.execute(
                """INSERT INTO local_slot_domains
                   (local_slot_id,domain_key,observation_count)
                   VALUES(?,?,?)""",
                (
                    merged_id,
                    str(domain["domain_key"]),
                    int(domain["observations"]),
                ),
            )
        hypothesis_rows = conn.execute(
            f"""SELECT participant_id,MAX(compatibility) AS compatibility,
                       MAX(selected) AS selected
                FROM participant_slot_hypotheses
                WHERE local_slot_id IN ({marks})
                GROUP BY participant_id""",
            unique_ids,
        ).fetchall()
        for hypothesis in hypothesis_rows:
            conn.execute(
                """INSERT INTO participant_slot_hypotheses
                   (participant_id,local_slot_id,compatibility,selected,
                    evidence_json,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (
                    str(hypothesis["participant_id"]),
                    merged_id,
                    float(hypothesis["compatibility"]),
                    int(hypothesis["selected"]),
                    encode(["explicit_cluster_merge"]),
                    now,
                ),
            )
        slot_set_rows = conn.execute(
            f"""SELECT DISTINCT slot_set_id FROM slot_set_members
                WHERE local_slot_id IN ({marks})""",
            unique_ids,
        ).fetchall()
        for membership in slot_set_rows:
            conn.execute(
                """INSERT OR IGNORE INTO slot_set_members
                   (slot_set_id,local_slot_id) VALUES(?,?)""",
                (str(membership["slot_set_id"]), merged_id),
            )
        prototype_rows = conn.execute(
            f"""SELECT prototype_id,MAX(compatibility) AS compatibility
                FROM slot_prototype_members
                WHERE local_slot_id IN ({marks})
                GROUP BY prototype_id""",
            unique_ids,
        ).fetchall()
        for membership in prototype_rows:
            conn.execute(
                """INSERT OR IGNORE INTO slot_prototype_members
                   (prototype_id,local_slot_id,compatibility)
                   VALUES(?,?,?)""",
                (
                    str(membership["prototype_id"]),
                    merged_id,
                    float(membership["compatibility"]),
                ),
            )
        conn.execute(
            f"""DELETE FROM slot_set_members
                WHERE local_slot_id IN ({marks})""",
            unique_ids,
        )
        conn.execute(
            f"""DELETE FROM slot_prototype_members
                WHERE local_slot_id IN ({marks})""",
            unique_ids,
        )
        conn.execute(
            f"""UPDATE local_slots SET status='DEPRECATED',updated_at=?
                WHERE id IN ({marks})""",
            (now, *unique_ids),
        )
        return LocalSlot(
            id=merged_id,
            predicate_concept_id=predicate_concept_id,
            centroid_signature=ObservationSignature(centroid),
            support_count=total_support,
            contradiction_count=total_contradictions,
            domain_diversity=domain_diversity,
            confidence=confidence,
            status=status,
        )


class ConstructionLearner:
    """Clusters structural patterns and learns their gap compatibility."""

    MATCH_THRESHOLD = 0.70

    def observe(
        self,
        conn: Any,
        signature: ObservationSignature,
        *,
        source_id: str,
        domain_key: str,
        gap_kind: Optional[GapKind] = None,
        contradicted: bool = False,
    ) -> ConstructionCluster:
        rows = conn.execute(
            """SELECT * FROM construction_clusters
               WHERE status<>'DEPRECATED'
                 AND (gap_kind IS ? OR gap_kind=?)
               ORDER BY confidence DESC""",
            (
                gap_kind.value if gap_kind else None,
                gap_kind.value if gap_kind else None,
            ),
        ).fetchall()
        scored = [
            (
                signature_similarity(
                    signature.values,
                    decode(row["structural_signature_json"], {}),
                ),
                row,
            )
            for row in rows
        ]
        winner = max(scored, default=None, key=lambda item: item[0])
        if not winner or winner[0] < self.MATCH_THRESHOLD:
            construction_id = stable_id(
                "construction",
                sorted(signature.values.items()),
                gap_kind.value if gap_kind else "",
            )
            now = utcnow()
            conn.execute(
                """INSERT OR IGNORE INTO construction_clusters
                   (id,structural_signature_json,gap_kind,support_count,
                    contradiction_count,domain_diversity,confidence,status,
                    construction_model_version,created_at,updated_at)
                   VALUES(?,?,?,0,0,0,0,'CANDIDATE',?,?,?)""",
                (
                    construction_id,
                    encode(signature.as_dict()),
                    gap_kind.value if gap_kind else None,
                    CONSTRUCTION_MODEL_VERSION,
                    now,
                    now,
                ),
            )
        else:
            construction_id = str(winner[1]["id"])
        conn.execute(
            """INSERT OR IGNORE INTO construction_evidence
               (construction_id,source_id,structural_signature_json,domain_key,
                contradicted,created_at)
               VALUES(?,?,?,?,?,?)""",
            (
                construction_id,
                source_id,
                encode(signature.as_dict()),
                domain_key,
                int(contradicted),
                utcnow(),
            ),
        )
        evidence_rows = conn.execute(
            """SELECT structural_signature_json,domain_key,contradicted
               FROM construction_evidence WHERE construction_id=?""",
            (construction_id,),
        ).fetchall()
        centroid: Dict[str, float] = {}
        support_count = 0
        contradiction_count = 0
        domains = set()
        for evidence in evidence_rows:
            if int(evidence["contradicted"]):
                contradiction_count += 1
                continue
            centroid = merge_centroid(
                centroid,
                decode(evidence["structural_signature_json"], {}),
                support_count,
            )
            support_count += 1
            domains.add(str(evidence["domain_key"]))
        domain_diversity = len(domains)
        status = SlotLearner._status(
            support_count,
            contradiction_count,
            domain_diversity,
        )
        confidence = SlotLearner._confidence(
            support_count,
            contradiction_count,
            domain_diversity,
        )
        conn.execute(
            """UPDATE construction_clusters
               SET structural_signature_json=?,support_count=?,
                   contradiction_count=?,domain_diversity=?,confidence=?,
                   status=?,updated_at=? WHERE id=?""",
            (
                encode(centroid),
                support_count,
                contradiction_count,
                domain_diversity,
                confidence,
                status.value,
                utcnow(),
                construction_id,
            ),
        )
        compatibility_rows = conn.execute(
            """SELECT prototype_id,compatibility
               FROM construction_slot_compatibility
               WHERE construction_id=?""",
            (construction_id,),
        ).fetchall()
        return ConstructionCluster(
            id=construction_id,
            structural_signature=ObservationSignature(centroid),
            gap_kind=gap_kind,
            compatible_slot_prototypes={
                str(row["prototype_id"]): float(row["compatibility"])
                for row in compatibility_rows
            },
            support_count=support_count,
            contradiction_count=contradiction_count,
            domain_diversity=domain_diversity,
            confidence=confidence,
            status=status,
        )

    def best_match(
        self,
        conn: Any,
        signature: ObservationSignature,
        *,
        gap_kind: Optional[GapKind] = None,
    ) -> Optional[ConstructionCluster]:
        rows = conn.execute(
            """SELECT * FROM construction_clusters
               WHERE status NOT IN ('DEPRECATED','WEAKENED')
               ORDER BY confidence DESC LIMIT 128"""
        ).fetchall()
        winner: Optional[Tuple[float, Any]] = None
        for row in rows:
            row_gap = GapKind(str(row["gap_kind"])) if row["gap_kind"] else None
            if gap_kind and row_gap and row_gap != gap_kind:
                continue
            learned_signature = decode(
                row["structural_signature_json"],
                {},
            )
            current_predicate_forms = {
                key
                for key in signature.values
                if key.startswith("shape:predicate_form:")
            }
            learned_predicate_forms = {
                key
                for key in learned_signature
                if key.startswith("shape:predicate_form:")
            }
            current_has_voice = any(
                key.startswith("voice:") for key in signature.values
            )
            learned_has_voice = any(
                key.startswith("voice:") for key in learned_signature
            )
            # Predicate form and explicit voice markers are structural
            # controls, not answer semantics.  Treating an active finite-verb
            # construction as a passive participial one can transfer an
            # instrumental tool slot onto an instrumental passive participant.
            if (
                current_predicate_forms
                and learned_predicate_forms
                and current_predicate_forms.isdisjoint(
                    learned_predicate_forms
                )
            ):
                continue
            if current_has_voice != learned_has_voice:
                continue
            score = signature_similarity(
                signature.values,
                learned_signature,
            ) * float(row["confidence"])
            if winner is None or score > winner[0]:
                winner = (score, row)
        if not winner or winner[0] < 0.34:
            return None
        row = winner[1]
        compatibilities = conn.execute(
            """SELECT prototype_id,compatibility
               FROM construction_slot_compatibility
               WHERE construction_id=?""",
            (row["id"],),
        ).fetchall()
        return ConstructionCluster(
            id=str(row["id"]),
            structural_signature=ObservationSignature(
                decode(row["structural_signature_json"], {})
            ),
            gap_kind=GapKind(str(row["gap_kind"])) if row["gap_kind"] else None,
            compatible_slot_prototypes={
                str(item["prototype_id"]): float(item["compatibility"])
                for item in compatibilities
            },
            support_count=int(row["support_count"]),
            contradiction_count=int(row["contradiction_count"]),
            domain_diversity=int(row["domain_diversity"]),
            confidence=float(row["confidence"]),
            status=SlotStatus(str(row["status"])),
        )

    def reinforce_binding(
        self,
        conn: Any,
        construction_id: str,
        local_slot_id: str,
        *,
        accepted: bool,
    ) -> None:
        memberships = conn.execute(
            """SELECT prototype_id FROM slot_prototype_members
               WHERE local_slot_id=? ORDER BY compatibility DESC""",
            (local_slot_id,),
        ).fetchall()
        for membership in memberships:
            prototype_id = str(membership["prototype_id"])
            conn.execute(
                """INSERT INTO construction_slot_compatibility
                   (construction_id,prototype_id,compatibility,support_count,
                    contradiction_count)
                   VALUES(?,?,?, ?,?)
                   ON CONFLICT(construction_id,prototype_id) DO UPDATE SET
                     support_count=support_count+excluded.support_count,
                     contradiction_count=contradiction_count
                         +excluded.contradiction_count,
                     compatibility=MAX(
                       0.0,MIN(1.0,
                         (support_count+excluded.support_count+1.0)
                         /(support_count+excluded.support_count
                           +contradiction_count+excluded.contradiction_count+2.0)
                       )
                     )""",
                (
                    construction_id,
                    prototype_id,
                    0.67 if accepted else 0.33,
                    int(accepted),
                    int(not accepted),
                ),
            )


class SemanticClusterLearner:
    """Learn anonymous entity clusters from recurring context signatures."""

    MATCH_THRESHOLD = 0.63
    ALTERNATIVE_MARGIN = 0.08

    def observe(
        self,
        conn: Any,
        entity_id: str,
        context_signature: ObservationSignature,
    ) -> Sequence[tuple[str, float]]:
        rows = conn.execute(
            """SELECT * FROM semantic_clusters
               ORDER BY confidence DESC,id LIMIT 256"""
        ).fetchall()
        scored = sorted(
            [
                (
                signature_similarity(
                    context_signature.values,
                    decode(row["context_centroid_json"], {}),
                ),
                row,
                )
                for row in rows
            ],
            key=lambda item: item[0],
            reverse=True,
        )
        winner = (
            scored[0]
            if scored and scored[0][0] >= self.MATCH_THRESHOLD
            else None
        )
        if winner:
            cluster_id = str(winner[1]["id"])
            score = winner[0]
            support_count = int(winner[1]["support_count"])
            centroid = merge_centroid(
                decode(winner[1]["context_centroid_json"], {}),
                context_signature.values,
                support_count,
            )
        else:
            structural = sorted(
                (key, round(float(value), 1))
                for key, value in context_signature.values.items()
                if key.split(":", 1)[0] in {
                    "morph",
                    "agreement",
                    "preposition",
                    "context",
                }
                and value >= 0.35
            )
            cluster_id = stable_id("semantic-cluster", structural)
            score = 1.0
            support_count = 0
            centroid = context_signature.as_dict()
        now = utcnow()
        next_support = support_count + 1
        confidence = clamp(0.12 + 0.18 * math.log1p(next_support))
        conn.execute(
            """INSERT INTO semantic_clusters
               (id,seed_hints_json,context_centroid_json,support_count,
                confidence,display_label,version,created_at,updated_at)
               VALUES(?,'[]',?,?,?,NULL,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 context_centroid_json=excluded.context_centroid_json,
                 support_count=excluded.support_count,
                 confidence=excluded.confidence,
                 updated_at=excluded.updated_at""",
            (
                cluster_id,
                encode(centroid),
                next_support,
                confidence,
                SEMANTIC_CLUSTER_VERSION,
                now,
                now,
            ),
        )
        conn.execute(
            """INSERT INTO semantic_cluster_members
               (semantic_cluster_id,entity_id,compatibility,evidence_json)
               VALUES(?,?,?,?)
               ON CONFLICT(semantic_cluster_id,entity_id) DO UPDATE SET
                 compatibility=MAX(compatibility,excluded.compatibility),
                 evidence_json=excluded.evidence_json""",
            (
                cluster_id,
                entity_id,
                score,
                encode(["recurring_context_signature"]),
            ),
        )
        alternatives = [
            (str(row["id"]), alternative_score)
            for alternative_score, row in scored[1:]
            if (
                alternative_score >= self.MATCH_THRESHOLD
                and score - alternative_score <= self.ALTERNATIVE_MARGIN
            )
        ]
        return ((cluster_id, score), *alternatives)

    @staticmethod
    def split_cluster(
        conn: Any,
        cluster_id: str,
        partitions: Sequence[Sequence[str]],
    ) -> Sequence[str]:
        if len(partitions) < 2 or any(not partition for partition in partitions):
            raise ValueError("cluster split requires at least two non-empty partitions")
        row = conn.execute(
            "SELECT * FROM semantic_clusters WHERE id=?",
            (cluster_id,),
        ).fetchone()
        if not row:
            raise KeyError(cluster_id)
        created: List[str] = []
        for partition in partitions:
            new_id = stable_id(
                "semantic-cluster",
                cluster_id,
                sorted(partition),
            )
            now = utcnow()
            conn.execute(
                """INSERT INTO semantic_clusters
                   (id,seed_hints_json,context_centroid_json,support_count,
                    confidence,display_label,version,created_at,updated_at)
                   VALUES(?,'[]',?,?,.2,NULL,?,?,?)""",
                (
                    new_id,
                    row["context_centroid_json"],
                    len(partition),
                    SEMANTIC_CLUSTER_VERSION,
                    now,
                    now,
                ),
            )
            for entity_id in partition:
                conn.execute(
                    """INSERT OR IGNORE INTO semantic_cluster_members
                       (semantic_cluster_id,entity_id,compatibility,evidence_json)
                       VALUES(?,?,.7,'["cluster_split"]')""",
                    (new_id, entity_id),
                )
            created.append(new_id)
        conn.execute(
            "DELETE FROM semantic_clusters WHERE id=?",
            (cluster_id,),
        )
        return tuple(created)
