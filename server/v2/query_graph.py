"""QueryGraph construction, strict graph matching and gap binding."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .event_graph import EventGraphPipeline
from .graph_learning import (
    ConstructionLearner,
    ObservationBuilder,
    SlotLearner,
    signature_similarity,
)
from .graph_models import (
    AnswerStatus,
    BindingStatus,
    CandidateBinding,
    GapKind,
    GapNode,
    GraphStatus,
    MentionNode,
    ModelVersions,
    ObservationSignature,
    PredicateNode,
    QueryGraph,
)
from .graph_repository import (
    GraphRepository,
    encode,
    stable_id,
    utcnow,
)
from .language import UniversalLanguageAnalyzer
from .query_operator_learning import QueryOperatorLearner
from .swarm import GapSwarmCoordinator
from .question_family import (
    resolve_question_family,
    check_animacy_compatibility,
    AnimacyCompatibility,
)
from .gap_release import (
    GapReleaseSelector,
    GapReleaseDiagnostic,
    ReleaseDecision,
)
from .event_binding_frame import (
    EventBindingFrame,
    EventBindingFrameBuilder,
)
from .dialogue_context import DialogueContextState


class QueryGraphBuilder:
    """Project a linguistic analysis into a graph with one structural gap."""

    def __init__(
        self,
        repository: GraphRepository,
        morphology: Any,
    ) -> None:
        self.repository = repository
        self.morphology = morphology
        self.language = UniversalLanguageAnalyzer(morphology)
        self.observations = ObservationBuilder()
        self.slots = SlotLearner()
        self.constructions = ConstructionLearner()
        self.query_operators = QueryOperatorLearner()

    @staticmethod
    def _passive_perspective_relation(
        analysis: Any,
        predicate: Optional[PredicateNode],
        known_nodes: Sequence[MentionNode],
        questions: Sequence[Any],
    ) -> Dict[str, Any]:
        """Classify passive-result evidence without changing stored events."""
        if predicate is None or predicate.token_index is None:
            return {
                "passive_perspective_status": "REJECTED",
                "predicate_perspective_relation": None,
            }
        token = analysis.tokens[predicate.token_index]
        has_question = bool(questions)
        instrumental = any(
            node.features.get("case") == "ablt"
            for node in known_nodes
        )
        if token.pos == "PRTS" and has_question and instrumental:
            return {
                "passive_perspective_status": "CONFIRMED",
                "predicate_perspective_relation": {
                    "perspective": "PASSIVE_RESULT",
                    "origin": "MORPHOLOGICAL",
                    "confidence": 0.85,
                    "evidence": [
                        "short_passive_participle",
                        "question_operator",
                        "predicate_position",
                        "instrumental_participant",
                    ],
                },
            }
        if token.pos == "PRTF" or (
            token.pos == "PRTS"
            and any(item.lemma == "быть" and item.index < token.index
                    for item in analysis.tokens)
        ):
            return {
                "passive_perspective_status": "SHADOW",
                "predicate_perspective_relation": None,
            }
        return {
            "passive_perspective_status": (
                "AMBIGUOUS" if token.pos == "PRTS" else "REJECTED"
            ),
            "predicate_perspective_relation": None,
        }

    @staticmethod
    def _default_gap_kind(
        analysis: Any,
        question: Optional[Any] = None,
    ) -> GapKind:
        question = question or analysis.question_operator
        # A typed interrogative noun phrase is an attribute question only
        # when it is self-contained.  Once the clause supplies a predicate,
        # the noun constrains the missing event participant instead of naming
        # an already resolved node (``Какие кусочки лежат...``).
        typed_event_question = bool(
            question
            and question.type_constraint_token_index is not None
            and getattr(analysis, "predicate", None) is not None
        )
        if (
            question
            and str(getattr(question, "operator_type", "")) == "NODE_COMPONENT"
            and not typed_event_question
            and not (
                question.type_constraint_token_index is not None
                and question.token_indices[0] > 0
                and analysis.tokens[question.token_indices[0] - 1].pos == "PREP"
            )
        ):
            return GapKind.NODE_COMPONENT
        if question and question.type_constraint_token_index is not None:
            question_index = question.token_indices[0]
            if (
                question_index > 0
                and analysis.tokens[question_index - 1].pos == "PREP"
            ):
                return GapKind.RELATION_VALUE
            if typed_event_question:
                return GapKind.EVENT_ATTACHMENT
            return GapKind.NODE_COMPONENT
        if question:
            token = analysis.tokens[question.token_indices[0]]
            if token.pos == "NUMR":
                return GapKind.QUANTITY_VALUE
            if (
                token.pos in {"ADVB", "PRED"}
                or any(
                    hypothesis.pos in {"ADVB", "PRED"}
                    and hypothesis.confidence >= 0.05
                    for hypothesis in token.analyses
                )
            ):
                return GapKind.EVENT_PROPERTY
            return GapKind.EVENT_ATTACHMENT
        return (
            GapKind.BOOLEAN_RESULT
            if "?" in str(getattr(analysis.utterance, "raw_text", ""))
            else GapKind.WHOLE_EVENT
        )

    @staticmethod
    def _binding_as_mention(
        binding: CandidateBinding,
        graph_id: str,
    ) -> MentionNode:
        return MentionNode(
            id=stable_id("known-binding", graph_id, binding.resolved_node_id),
            head_lemma=binding.resolved_lemma,
            head_surface=binding.resolved_surface,
            surface=binding.resolved_surface,
            token_start=0,
            token_end=0,
            token_indices=(),
            features=dict(binding.resolved_features),
            preposition=str(
                binding.resolved_features.get("preposition") or ""
            ),
            entity_id=binding.resolved_concept_id,
            origin="RESOLVED_PREVIOUS_TARGET",
            source_query_graph_id=binding.query_graph_id,
            source_gap_id=binding.gap_node_id,
            source_binding_id=binding.id,
            replaceable=True,
            context_confidence=binding.total_score,
        )

    def _known_nodes(
        self,
        analysis: Any,
        query_graph_id: str,
        gap_kind: GapKind,
    ) -> List[MentionNode]:
        questions = getattr(analysis, "question_operators", ()) or (
            (analysis.question_operator,) if analysis.question_operator else ()
        )
        question_indices = {
            index for question in questions for index in question.token_indices
        }
        result: List[MentionNode] = []
        for mention in analysis.mentions:
            # In a typed modifier question the noun remains known while only
            # its interrogative component becomes the gap.
            typed_head = (
                any(question.type_constraint_token_index is not None
                    and int(question.type_constraint_token_index)
                    in mention.token_indices for question in questions)
            )
            if typed_head and gap_kind != GapKind.NODE_COMPONENT:
                continue
            if (
                set(mention.token_indices).issubset(question_indices)
                and not typed_head
            ):
                continue
            result.append(
                self.observations.mention_node(
                    analysis,
                    mention,
                    query_graph_id,
                )
            )
        return result

    def _attached_node_id(
        self,
        analysis: Any,
        known_nodes: Sequence[MentionNode],
    ) -> Optional[str]:
        question = analysis.question_operator
        if not question or question.type_constraint_token_index is None:
            return None
        target_index = int(question.type_constraint_token_index)
        return next(
            (
                node.id for node in known_nodes
                if target_index in node.token_indices
                or node.token_start <= target_index <= node.token_end
            ),
            None,
        )

    def _compatible_slots(
        self,
        conn: Any,
        predicate: Optional[PredicateNode],
        question_signature: ObservationSignature,
        construction: Any,
    ) -> Dict[str, float]:
        if not predicate:
            return {}
        compatibility: Dict[str, float] = {}
        for slot in self.slots.load_local_slots(conn, predicate.concept_id):
            score = signature_similarity(
                question_signature.values,
                slot.centroid_signature.values,
            )
            if score >= 0.18:
                compatibility[slot.id] = score
        if construction:
            for prototype_id, prototype_score in (
                construction.compatible_slot_prototypes.items()
            ):
                rows = conn.execute(
                    """SELECT local_slot_id,compatibility
                       FROM slot_prototype_members WHERE prototype_id=?""",
                    (prototype_id,),
                ).fetchall()
                for row in rows:
                    local_slot_id = str(row["local_slot_id"])
                    score = float(prototype_score) * float(row["compatibility"])
                    compatibility[local_slot_id] = max(
                        compatibility.get(local_slot_id, 0.0),
                        score,
                    )
        return dict(
            sorted(
                compatibility.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:8]
        )

    @staticmethod
    def _question_morphology_hypotheses(
        question_signature: ObservationSignature,
    ) -> Dict[str, float]:
        """Expose all retained question morphology alternatives in the graph."""
        return {
            key.removeprefix("morph:"): score
            for key, score in question_signature.values.items()
            if key.startswith("morph:")
        }

    @staticmethod
    def _implicit_gaps(
        analysis: Any,
        predicate: Optional[PredicateNode],
        graph_id: str,
        questions: Sequence[Any] = (),
        explicit_known_nodes: Sequence[MentionNode] = (),
    ) -> tuple[GapNode, ...]:
        """Represent possible omitted agreement controllers separately.

        This is deliberately not a binding target.  In particular, past
        masculine ``разрезал`` in ``Что разрезал?`` supplies evidence about an
        implicit participant, not a nominative proof for ``что``.
        """
        if not predicate:
            return ()
        # ``кто`` explicitly reserves the agreeing nominative participant;
        # adding a second hidden participant corrupts the configuration.
        if any(
            str(question.question_lemma).casefold() == "кто"
            for question in questions
        ):
            return ()
        features = dict(predicate.features)
        if str(features.get("tense") or "") != "past":
            return ()
        gender = str(features.get("gender") or "")
        number = str(features.get("number") or "")
        if not gender and not number:
            return ()
        # An overt current participant that agrees with the predicate already
        # explains its inflection.  Do not invent an implicit controller in
        # ``Что получил робот от механика?`` merely because ``получил`` is
        # masculine singular.
        for node in explicit_known_nodes:
            if node.origin != "EXPLICIT_CURRENT":
                continue
            comparable = [
                feature for feature in ("gender", "number")
                if node.features.get(feature) and features.get(feature)
            ]
            if comparable and all(
                node.features[feature] == features[feature]
                for feature in comparable
            ):
                return ()
        signature_values = {
            f"morph:{feature}:{value}": 0.80
            for feature, value in (("gender", gender), ("number", number))
            if value
        }
        return (GapNode(
            id=stable_id("implicit-gap", graph_id, predicate.token_index),
            gap_kind=GapKind.EVENT_ATTACHMENT,
            question_signature=ObservationSignature(signature_values),
            surface="",
            token_indices=(),
            requested=False,
            required=False,
            morphology_hypotheses={
                key.removeprefix("morph:"): value
                for key, value in signature_values.items()
            },
            evidence={
                "reason": "PAST_PREDICATE_AGREEMENT_MAY_DESCRIBE_OMITTED_PARTICIPANT",
                "predicate_gender": gender,
                "predicate_number": number,
            },
        ),)

    @staticmethod
    def _current_clause_completeness(
        analysis: Any,
        predicate_token: Optional[Any],
        questions: Sequence[Any],
        known_nodes: Sequence[MentionNode],
    ) -> float:
        """Estimate whether this turn already supplies its own query frame."""
        score = 0.0
        if predicate_token:
            score += 0.45
        if questions:
            score += 0.25
        if known_nodes:
            score += 0.20
        raw_text = str(getattr(analysis.utterance, "raw_text", ""))
        if "?" in raw_text:
            score += 0.10
        return min(1.0, score)

    @staticmethod
    def _self_contained_relational_evidence(
        analysis: Any,
        questions: Sequence[Any],
        known_nodes: Sequence[MentionNode],
        predicate_token: Optional[Any],
    ) -> Dict[str, Any]:
        """Recognise a complete relation question with an omitted predicate.

        This is deliberately grammatical rather than lexical: a question
        operator plus a prepositional noun phrase is enough to form a local
        relation frame.  The returned predicate family is an opaque retrieval
        hint, never a substituted verb such as ``находиться``.
        """
        relation_nodes = [node for node in known_nodes if node.preposition]
        evidence = []
        if questions:
            evidence.append("question_operator")
        if relation_nodes:
            evidence.append("prepositional_phrase")
        if any(node.head_lemma for node in relation_nodes):
            evidence.append("preposition_object")
        if not predicate_token:
            evidence.append("no_explicit_predicate")
        confidence = min(1.0, 0.20 * len(evidence))
        return {
            "is_self_contained_relational": bool(
                questions and relation_nodes and not predicate_token
            ),
            "predicate_origin": "IMPLICIT_RELATIONAL",
            "predicate_family": "existence_or_location",
            "surface": None,
            "confidence": confidence,
            "evidence": evidence,
        }

    def build(
        self,
        text: str,
        *,
        previous_graph: Optional[QueryGraph] = None,
        previous_binding: Optional[CandidateBinding] = None,
        previous_bindings: Sequence[CandidateBinding] = (),
        identity_context: str = "",
        conversation_id: str = "",
        turn_index: int = 0,
    ) -> tuple[QueryGraph, Dict[str, Any]]:
        analysis = self.language.analyze(
            text,
            detect_question=True,
            source_type="dialogue",
            speaker_role="user",
            conversation_id=conversation_id,
            turn_index=turn_index,
        )
        canonical_previous_bindings = tuple(previous_bindings) or (
            (previous_binding,) if previous_binding else ()
        )
        normalized_markers = {
            token.lemma.casefold() for token in analysis.tokens
        }
        repeats_previous_gap = bool(
            previous_graph and normalized_markers & {"ещё", "другой"}
        )
        graph_id = stable_id(
            "query-graph",
            text,
            previous_graph.id if previous_graph else "",
            ",".join(binding.id for binding in canonical_previous_bindings),
            identity_context,
        )
        if repeats_previous_gap and previous_graph:
            exclusions = list(previous_graph.exclusions)
            for binding in canonical_previous_bindings:
                exclusions.append({
                    "resolved_node_id": binding.resolved_node_id,
                    "resolved_concept_id": binding.resolved_concept_id,
                    "lemma": binding.resolved_lemma,
                })
            gap = replace(
                previous_graph.gap_node,
                id=stable_id("gap", graph_id),
            )
            graph = QueryGraph(
                id=graph_id,
                predicate=replace(
                    previous_graph.predicate,
                    token_index=None,
                    origin="INHERITED",
                    source_token_index=(
                        previous_graph.predicate.source_token_index
                        if previous_graph.predicate.source_token_index is not None
                        else previous_graph.predicate.token_index
                    ),
                    inherited_from_query_graph_id=previous_graph.id,
                ),
                known_nodes=tuple(previous_graph.known_nodes),
                gap_node=gap,
                target_gaps=(gap,),
                question_operators=tuple(previous_graph.question_operators),
                required_edges=tuple(previous_graph.required_edges),
                exclusions=tuple(exclusions),
                status=GraphStatus.READY,
                continuation_of=previous_graph.id,
                construction_ids=tuple(previous_graph.construction_ids),
                implicit_gaps=tuple(previous_graph.implicit_gaps),
                trace={
                    **dict(previous_graph.trace),
                    "continuation": "REUSED_GAP",
                    "reused_gap_signature": True,
                    "language_analysis": analysis.as_dict(),
                },
            )
            return graph, analysis.as_dict()

        questions = tuple(getattr(analysis, "question_operators", ()) or ())
        if not questions and analysis.question_operator:
            questions = (analysis.question_operator,)
        inferred_gap_kind = self._default_gap_kind(
            analysis, questions[0] if questions else None
        )
        predicate_token = analysis.predicate
        current_known_nodes = self._known_nodes(
            analysis,
            graph_id,
            inferred_gap_kind,
        )
        completeness = self._current_clause_completeness(
            analysis, predicate_token, questions, current_known_nodes,
        )
        implicit_relation = self._self_contained_relational_evidence(
            analysis, questions, current_known_nodes, predicate_token,
        )
        first_token = (
            analysis.tokens[0].normalized.casefold()
            if analysis.tokens else ""
        )
        has_referential_marker = first_token in {"а", "и"}
        # "А" is discourse glue, not proof of an ellipsis.  A complete
        # question supplies its own predicate, target and mentions; it may use
        # a prior answer as an event anchor, but not as structural material.
        continuation_mode = (
            "REFERENTIAL" if previous_graph and has_referential_marker
            and completeness >= 0.90 else
            "NONE" if implicit_relation["is_self_contained_relational"] else
            "STRUCTURAL" if previous_graph and completeness < 0.90 else
            "NONE"
        )
        is_structural_continuation = continuation_mode == "STRUCTURAL"
        is_referential_continuation = continuation_mode == "REFERENTIAL"
        inherited_predicate = previous_graph.predicate if previous_graph else None
        predicate = (
            PredicateNode(
                lemma=predicate_token.lemma.casefold(),
                surface=predicate_token.surface,
                concept_id=stable_id(
                    "predicate-concept",
                    predicate_token.lemma.casefold(),
                ),
                token_index=predicate_token.index,
                features=dict(predicate_token.features),
                origin="CURRENT_EXPLICIT",
            )
            if predicate_token else (
                replace(
                    inherited_predicate,
                    token_index=None,
                    origin="INHERITED",
                    source_token_index=(
                        inherited_predicate.source_token_index
                        if inherited_predicate.source_token_index is not None
                        else inherited_predicate.token_index
                    ),
                    inherited_from_query_graph_id=previous_graph.id,
                ) if (
                    inherited_predicate
                    and previous_graph
                    and is_structural_continuation
                ) else None
            )
        )
        passive_perspective = self._passive_perspective_relation(
            analysis,
            predicate,
            current_known_nodes,
            questions,
        )
        known_nodes = list(current_known_nodes)
        if previous_graph and not predicate_token and known_nodes and not questions:
            # A bare noun after an answered question replaces the known node
            # while retaining the previous predicate and structural gap:
            # ``Где лежит помидор?`` -> ``Яблоко?``.  It is not a Boolean
            # query, and previous entities must not leak into this turn.
            gap = replace(
                previous_graph.gap_node,
                id=stable_id("gap", graph_id),
            )
            graph = QueryGraph(
                id=graph_id,
                predicate=replace(
                    previous_graph.predicate,
                    token_index=None,
                    origin="INHERITED",
                    source_token_index=(
                        previous_graph.predicate.source_token_index
                        if previous_graph.predicate.source_token_index is not None
                        else previous_graph.predicate.token_index
                    ),
                    inherited_from_query_graph_id=previous_graph.id,
                ),
                known_nodes=tuple(known_nodes),
                gap_node=gap,
                target_gaps=(gap,),
                question_operators=tuple(previous_graph.question_operators),
                required_edges=tuple(previous_graph.required_edges),
                status=GraphStatus.READY,
                continuation_of=previous_graph.id,
                construction_ids=tuple(previous_graph.construction_ids),
                implicit_gaps=tuple(previous_graph.implicit_gaps),
                trace={
                    "continuation": "REPLACED_KNOWN_NODES",
                    "continuation_mode": "STRUCTURAL",
                    "continuation_strategy": "STRUCTURAL_CONTINUATION",
                    "current_clause_completeness": completeness,
                    "inherited_predicate": True,
                    "inherited_previous_binding": False,
                    "replaced_known_nodes": True,
                    "language_analysis": analysis.as_dict(),
                },
            )
            return graph, analysis.as_dict()
        inherited_binding = False
        released_node: Optional[MentionNode] = None
        release_diagnostic: Optional[Dict[str, Any]] = None
        if is_structural_continuation and previous_graph:
            inherited = [replace(
                node,
                origin="EXPLICIT_INHERITED",
                source_query_graph_id=previous_graph.id,
                replaceable=True,
                context_confidence=0.92,
            ) for node in previous_graph.known_nodes]
            for binding in canonical_previous_bindings:
                inherited.append(self._binding_as_mention(binding, graph_id))
            if canonical_previous_bindings:
                inherited_binding = True
            current_keys = {
                (node.head_lemma, node.qualified_key) for node in known_nodes
            }
            for node in inherited:
                if (node.head_lemma, node.qualified_key) not in current_keys:
                    known_nodes.append(node)
            # A new interrogative operator rotates the GAP inside the anchored
            # event: release the inherited value using QuestionFamily, animacy,
            # morphology, and lineage — not just flat case matching.
            if questions and known_nodes:
                question_surface = questions[0].surface if questions else ""
                question_family_key = resolve_question_family(question_surface)

                # Build a lightweight release frame from known nodes
                # when no persisted EventBindingFrame is available.
                # The frame is reconstructed from inherited nodes and bindings.
                current_explicit_ids = {
                    node.id for node in known_nodes
                    if node.origin == "EXPLICIT_CURRENT"
                }

                # Score each releasable node using question family + morphology
                def enhanced_release_score(node: MentionNode) -> float:
                    score = 0.0
                    node_case = str(node.features.get("case") or "")
                    node_animacy = str(node.features.get("animacy") or "")

                    # Case match (legacy compatibility)
                    requested_cases = {
                        key.removeprefix("morph:case:")
                        for key in signature.values
                        if key.startswith("morph:case:")
                    }
                    if requested_cases and node_case in requested_cases:
                        score += 0.50

                    # Question family match via animacy
                    if question_family_key and node_animacy:
                        compatibility = check_animacy_compatibility(
                            question_family_key, node_animacy,
                        )
                        if compatibility == AnimacyCompatibility.EXACT:
                            score += 0.65
                        elif compatibility == AnimacyCompatibility.COMPATIBLE:
                            score += 0.30
                        elif compatibility == AnimacyCompatibility.CONFLICTING:
                            score -= 1.50

                    # Source lineage bonus
                    if node.source_gap_id and question_family_key:
                        source_gap_family = resolve_question_family(
                            str(getattr(node, '_source_gap_surface', ''))
                        )
                        if source_gap_family == question_family_key:
                            score += 0.90

                    # Context confidence
                    score += 0.10 * float(node.context_confidence)

                    return score

                signature = self.observations.question_signature(
                    analysis, questions[0]
                )
                requested_cases = {
                    key.removeprefix("morph:case:")
                    for key in signature.values
                    if key.startswith("morph:case:")
                }

                releasable = [
                    node for node in known_nodes
                    if node.origin in {
                        "EXPLICIT_INHERITED",
                        "RESOLVED_PREVIOUS_TARGET",
                        "INFERRED_CONTEXT",
                    }
                ]

                if releasable:
                    # Score and select best candidate
                    scored = [
                        (enhanced_release_score(node), node)
                        for node in releasable
                    ]
                    scored.sort(key=lambda item: item[0], reverse=True)
                    best_score, best_node = scored[0]

                    second_score = scored[1][0] if len(scored) > 1 else -999.0
                    release_margin = best_score - second_score

                    # Only release if score is sufficient with clear margin
                    min_score = 0.25
                    min_margin = 0.12

                    if best_score >= min_score and release_margin >= min_margin:
                        released_node = best_node
                        known_nodes.remove(released_node)
                        release_diagnostic = {
                            "code": "RELEASED_WITH_QUESTION_FAMILY",
                            "released": released_node.as_dict(),
                            "score": best_score,
                            "second_score": second_score,
                            "margin": release_margin,
                            "question_family_key": question_family_key,
                            "candidates_considered": len(scored),
                        }
                    else:
                        release_diagnostic = {
                            "code": "AMBIGUOUS_GAP_RELEASE",
                            "best_score": best_score,
                            "second_score": second_score,
                            "margin": release_margin,
                            "min_score": min_score,
                            "min_margin": min_margin,
                            "question_family_key": question_family_key,
                        }
                else:
                    release_diagnostic = {
                        "code": "NO_RELEASABLE_NODES",
                        "question_family_key": question_family_key,
                    }
        structural = self.observations.structural_signature(
            analysis,
            gap_kind=inferred_gap_kind,
        )
        # The structural signature is strictly surface-observed.  An omitted
        # relational predicate may be useful as a weak retrieval hypothesis,
        # but it must never be recorded as an observed VERB and later learned
        # as though it occurred in the utterance.
        observed_features = structural.as_dict()
        inferred_features = (
            {
                "implicit_predicate_family:existence_or_location": 0.80,
                "equivalent_full_construction_support": 0.45,
            }
            if implicit_relation["is_self_contained_relational"] else {}
        )
        question_signature = self.observations.question_signature(
            analysis, questions[0] if questions else None
        )
        with self.repository.transaction() as conn:
            construction = self.constructions.best_match(
                conn,
                structural,
                gap_kind=inferred_gap_kind,
            )
            gap_kind = (
                construction.gap_kind
                if construction and construction.gap_kind
                else inferred_gap_kind
            )
            compatible_slots = self._compatible_slots(
                conn,
                predicate,
                question_signature,
                construction,
            )
        question = questions[0] if questions else None
        target_gaps: List[GapNode] = []
        coordination_group_id = (
            stable_id("gap-coordination", graph_id)
            if len(questions) > 1 else None
        )
        for index, operator in enumerate(questions or (None,)):
            signature = self.observations.question_signature(analysis, operator)
            operator_kind = self._default_gap_kind(analysis, operator)
            typed_event = bool(
                operator
                and operator.type_constraint_token_index is not None
                and operator_kind == GapKind.EVENT_ATTACHMENT
            )
            # A learned construction may refine ordinary questions, but it
            # must not turn the observed ``typed NP + predicate`` structure
            # back into an attribute lookup.
            current_kind = (
                operator_kind
                if typed_event or index > 0 else gap_kind
            )
            slots = compatible_slots if index == 0 else {}
            type_constraint: Dict[str, Any] = {}
            if operator and operator.type_constraint_token_index is not None:
                type_token = analysis.tokens[int(operator.type_constraint_token_index)]
                type_constraint = {
                    "lemma": type_token.lemma.casefold(),
                    "number": type_token.features.get("number"),
                    "origin": "EXPLICIT_TYPED_QUESTION",
                }
            target_gaps.append(GapNode(
                id=stable_id("gap", graph_id, index),
                gap_kind=current_kind,
                question_signature=signature,
                surface=operator.surface if operator else "",
                token_indices=tuple(operator.token_indices) if operator else (),
                attached_to_node_id=self._attached_node_id(analysis, known_nodes),
                compatible_slot_hypotheses=slots,
                requested=True,
                required=True,
                coordination_group_id=coordination_group_id,
                morphology_hypotheses=self._question_morphology_hypotheses(signature),
                evidence={
                    "question_surface": operator.surface if operator else "",
                    "requested": True,
                    "operator_index": index,
                    "type_constraint": type_constraint or None,
                    "type_constraint_lemma": type_constraint.get("lemma", ""),
                },
            ))
        # Query-operator profiles remain shadow evidence.  An unseen broad
        # attachment operator must not be prematurely narrowed to a property
        # role; specialization is allowed only after repeated confirmed fills.
        shadow_predictions: Dict[str, Dict[str, Any]] = {}
        with self.repository.transaction() as conn:
            enriched_gaps: List[GapNode] = []
            shadow_graph = QueryGraph(
                id=graph_id,
                predicate=predicate,
                known_nodes=tuple(known_nodes),
                gap_node=target_gaps[0],
                target_gaps=tuple(target_gaps),
                question_operators=(),
                status=GraphStatus.READY,
                continuation_of=(
                    previous_graph.id
                    if continuation_mode in {"STRUCTURAL", "REFERENTIAL"}
                    and previous_graph else None
                ),
                trace={"continuation_mode": continuation_mode},
            )
            for gap_index, current_gap in enumerate(target_gaps):
                prediction = self.query_operators.predict(
                    conn, shadow_graph, current_gap,
                )
                shadow_predictions[current_gap.id] = prediction
                operator = questions[gap_index] if gap_index < len(questions) else None
                profile_status = str(prediction.get("profile_status") or "UNSEEN").upper()
                support_count = int(prediction.get("support_count") or 0)
                broaden_unseen_attachment = bool(
                    current_gap.gap_kind == GapKind.EVENT_PROPERTY
                    and operator is not None
                    and str(getattr(operator, "operator_type", "")) == "EVENT_ATTACHMENT"
                    and continuation_mode == "NONE"
                    and (profile_status in {"UNSEEN", "SHADOW"} or support_count < 3)
                )
                enriched_gaps.append(replace(
                    current_gap,
                    gap_kind=(
                        GapKind.EVENT_ATTACHMENT
                        if broaden_unseen_attachment
                        else current_gap.gap_kind
                    ),
                    evidence={
                        **dict(current_gap.evidence),
                        "learned_gap_profile": prediction,
                        "preliminary_gap_kind": current_gap.gap_kind.value,
                        "operational_broadening": (
                            "UNSEEN_EVENT_ATTACHMENT"
                            if broaden_unseen_attachment
                            else None
                        ),
                    },
                ))
        target_gaps = enriched_gaps
        gap = target_gaps[0]
        implicit_gaps = self._implicit_gaps(
            analysis, predicate, graph_id, questions, current_known_nodes
        )
        required_edges: List[Dict[str, Any]] = []
        if inferred_gap_kind == GapKind.RELATION_VALUE and question:
            question_index = question.token_indices[0]
            preposition = (
                analysis.tokens[question_index - 1].normalized.casefold()
                if question_index > 0
                and analysis.tokens[question_index - 1].pos == "PREP"
                else ""
            )
            typed_mention = next(
                (
                    mention for mention in analysis.mentions
                    if question.type_constraint_token_index is not None
                    and int(question.type_constraint_token_index)
                    in mention.token_indices
                ),
                None,
            )
            required_edges.append({
                "edge_type": "VALUE_ATTACHED_TO_NODE",
                "preposition": preposition,
                "type_constraint_lemma": (
                    typed_mention.lemma if typed_mention else ""
                ),
            })
        status = (
            GraphStatus.READY
            if (predicate or implicit_relation["is_self_contained_relational"]) and (
                question is not None
                or gap_kind == GapKind.BOOLEAN_RESULT
            )
            else GraphStatus.INCOMPLETE
        )
        graph = QueryGraph(
            id=graph_id,
            predicate=predicate,
            known_nodes=tuple(known_nodes),
            gap_node=gap,
            target_gaps=tuple(target_gaps),
            question_operators=tuple(
                operator.as_dict() for operator in questions
            ),
            required_edges=tuple(required_edges),
            status=status,
            continuation_of=(
                previous_graph.id
                if continuation_mode in {"STRUCTURAL", "REFERENTIAL"}
                and previous_graph else None
            ),
            construction_ids=(construction.id,) if construction else (),
            implicit_gaps=implicit_gaps,
            trace={
                "preliminary_gap_kind": inferred_gap_kind.value,
                "structural_signature": observed_features,
                "structural_features": {
                    "observed_features": observed_features,
                    "inferred_features": inferred_features,
                    "memory_projected_features": {},
                    # Learning continues to consume only this conservative
                    # observed view until holdout validation permits more.
                    "selected_effective_features": observed_features,
                },
                "query_polarity": (
                    str(
                        getattr(
                            analysis.clauses[0].polarity,
                            "value",
                            analysis.clauses[0].polarity,
                        )
                    )
                    if analysis.clauses else "POSITIVE"
                ),
                "construction_feedback": (
                    construction.as_dict() if construction else None
                ),
                "continuation_mode": continuation_mode,
                "continuation_strategy": (
                    f"{continuation_mode}_CONTINUATION"
                    if continuation_mode != "NONE" else None
                ),
                "current_clause_completeness": completeness,
                "inherited_predicate": bool(is_structural_continuation),
                "predicate_hypothesis": (
                    implicit_relation
                    if implicit_relation["is_self_contained_relational"]
                    else None
                ),
                "inherited_previous_binding": inherited_binding,
                "event_anchor_id": (
                    canonical_previous_bindings[0].event_id
                    if canonical_previous_bindings
                    and len({
                        binding.event_id
                        for binding in canonical_previous_bindings
                    }) == 1
                    and continuation_mode in {
                        "STRUCTURAL", "REFERENTIAL",
                    } else None
                ),
                "event_anchor_inherited": bool(
                    canonical_previous_bindings
                    and len({
                        binding.event_id
                        for binding in canonical_previous_bindings
                    }) == 1
                    and continuation_mode in {
                        "STRUCTURAL", "REFERENTIAL",
                    }
                ),
                "anchor_event_predicate": (
                    previous_graph.predicate.as_dict()
                    if previous_graph and previous_graph.predicate
                    and is_referential_continuation else None
                ),
                "predicate_perspective_relation": (
                    passive_perspective["predicate_perspective_relation"]
                    if passive_perspective["predicate_perspective_relation"]
                    else {
                        "source_predicate": previous_graph.predicate.lemma,
                        "target_predicate": predicate.lemma if predicate else "",
                        "scope": "LOCAL_EVENT_ANCHOR",
                        "global_relation_used": False,
                    }
                    if previous_graph and previous_graph.predicate
                    and predicate and is_referential_continuation
                    and previous_graph.predicate.concept_id != predicate.concept_id
                    else None
                ),
                # A direct short-passive query has a local morphological
                # perspective.  Referential predicate changes retain their
                # established relation above; the two are never merged.
                "passive_perspective_status": passive_perspective[
                    "passive_perspective_status"
                ],
                "released_previous_node": released_node.as_dict()
                if released_node else None,
                # Pre-build release scoring is intentionally not promoted to
                # a trace diagnostic.  It has no execution/hypothesis owner;
                # only post-selection, owned diagnostics may leave the
                # service layer.
                "memory_feedback_limited_to_existing_hypotheses": True,
                "query_operator_shadow": shadow_predictions,
                "target_gap_requested": True,
                "target_gap_count": len(target_gaps),
                "implicit_gap_count": len(implicit_gaps),
                "language_analysis": analysis.as_dict(),
            },
        )
        return graph, analysis.as_dict()


class GraphMatcher:
    """Strict admission followed by soft ranking of admitted events."""

    SLOT_COMPATIBILITY_THRESHOLD = 0.18
    SELECTION_MARGIN = 0.08

    def __init__(self, repository: GraphRepository) -> None:
        self.repository = repository
        self.swarms = GapSwarmCoordinator(repository)

    @staticmethod
    def _known_event_prefilter(
        known_nodes: Sequence[MentionNode],
    ) -> tuple[str, tuple[Any, ...]]:
        """Use participant indexes before loading complete event graphs."""
        clauses: List[str] = []
        params: List[Any] = []
        for node in known_nodes:
            clauses.append(
                """ AND EXISTS (
                      SELECT 1
                      FROM graph_participants prefilter_p
                      JOIN graph_mentions prefilter_m
                        ON prefilter_m.id=prefilter_p.mention_id
                      WHERE prefilter_p.event_id=e.id
                        AND (
                          prefilter_m.entity_id=?
                          OR prefilter_m.qualified_key=?
                        )
                    )"""
            )
            params.extend([
                node.entity_id or "",
                node.qualified_key,
            ])
        return "".join(clauses), tuple(params)

    @staticmethod
    def _component_lemmas(node: MentionNode) -> set[str]:
        return {
            component.lemma.casefold()
            for component in node.components
            if component.required
        }

    def _match_known_nodes(
        self,
        known_nodes: Sequence[MentionNode],
        event: Any,
        *,
        allow_anchor_preposition_variance: bool = False,
    ) -> tuple[Optional[List[Any]], Optional[Dict[str, Any]]]:
        matched: List[Any] = []
        used: set[str] = set()
        for known in known_nodes:
            same_head = [
                participant for participant in event.participants
                if (
                    participant.id not in used
                    and (
                        participant.mention.entity_id == known.entity_id
                        or participant.mention.head_lemma == known.head_lemma
                    )
                )
            ]
            required_components = self._component_lemmas(known)
            compatible = [
                participant for participant in same_head
                if required_components.issubset(
                    self._component_lemmas(participant.mention)
                )
                and (
                    not known.preposition
                    or participant.mention.preposition == known.preposition
                    or allow_anchor_preposition_variance
                )
            ]
            if not compatible:
                if same_head and required_components:
                    return None, {
                        "status": "REJECTED",
                        "failed_constraint": "KNOWN_MENTION_COMPONENT",
                        "query": {
                            "head": known.head_lemma,
                            "required_components": sorted(required_components),
                        },
                        "event": {
                            "head": same_head[0].mention.head_lemma,
                            "components": sorted(
                                self._component_lemmas(same_head[0].mention)
                            ),
                        },
                        "reason": "REQUIRED_COMPONENT_MISMATCH",
                    }
                if same_head and known.preposition:
                    return None, {
                        "status": "REJECTED",
                        "failed_constraint": "KNOWN_RELATION",
                        "query": {
                            "head": known.head_lemma,
                            "preposition": known.preposition,
                        },
                        "event": {
                            "head": same_head[0].mention.head_lemma,
                            "preposition": same_head[0].mention.preposition,
                        },
                        "reason": "REQUIRED_RELATION_MISMATCH",
                    }
                return None, {
                    "status": "REJECTED",
                    "failed_constraint": "KNOWN_NODE",
                    "query": {
                        "head": known.head_lemma,
                        "qualified_key": known.qualified_key,
                    },
                    "reason": "REQUIRED_NODE_MISSING",
                    "query_surface": known.head_surface,
                    "selected_lemma": known.head_lemma,
                    "alternative_lemmas": list(
                        known.features.get("morphology_alternatives") or ()
                    ),
                    "rejection_entity": known.head_lemma,
                    "preposition_support": (
                        known.features.get("preposition_support")
                        or known.preposition
                    ),
                }
            winner = max(
                compatible,
                key=lambda participant: participant.confidence,
            )
            matched.append(winner)
            used.add(winner.id)
        return matched, None

    @staticmethod
    def _known_node_score(
        known_nodes: Sequence[MentionNode],
        matched: Sequence[Any],
    ) -> float:
        if not known_nodes:
            return 1.0
        scores: List[float] = []
        for known, participant in zip(known_nodes, matched):
            observed = participant.mention
            available = 0.0
            matching = 0.0
            for feature, weight in (
                ("case", 0.22),
                ("number", 0.08),
                ("gender", 0.06),
                ("animacy", 0.04),
            ):
                expected = known.features.get(feature)
                actual = observed.features.get(feature)
                if expected and actual:
                    available += weight
                    if expected == actual:
                        matching += weight
            if known.preposition:
                available += 0.10
                if known.preposition == observed.preposition:
                    matching += 0.10
            scores.append(
                0.55 + (0.45 * matching / available if available else 0.45)
            )
        return sum(scores) / len(scores)

    @staticmethod
    def _excluded(graph: QueryGraph, *, node_id: str, concept_id: str, lemma: str) -> bool:
        return any(
            (
                exclusion.get("resolved_node_id") == node_id
                or exclusion.get("resolved_concept_id") == concept_id
                or exclusion.get("lemma") == lemma
            )
            for exclusion in graph.exclusions
        )

    @staticmethod
    def _slot_scores(
        graph: QueryGraph,
        participant: Any,
    ) -> tuple[float, float, float, str]:
        direct = signature_similarity(
            graph.gap_node.question_signature.values,
            participant.observation_signature.values,
        )
        learned = max(
            (
                hypothesis.compatibility
                * graph.gap_node.compatible_slot_hypotheses.get(
                    hypothesis.local_slot_id,
                    0.0,
                )
                for hypothesis in participant.slot_hypotheses
            ),
            default=0.0,
        )
        local_slot_ids = [
            hypothesis.local_slot_id
            for hypothesis in participant.slot_hypotheses
        ]
        if learned >= GraphMatcher.SLOT_COMPATIBILITY_THRESHOLD:
            state = "compatible"
        elif local_slot_ids:
            state = "below_threshold"
        else:
            state = "fallback"
        return max(direct, learned), direct, learned, state

    @staticmethod
    def _structural_attachment_fit(
        graph: QueryGraph,
        signature: Optional[ObservationSignature],
    ) -> float:
        """Prefer tightly attached candidates in explicit voice constructions.

        This is a surface-structural observation shared by all question
        operators.  It does not map any interrogative lexeme to an answer
        type, and it stays inactive for ordinary finite-verb clauses.
        """
        structural = dict(graph.trace.get("structural_signature") or {})
        if (
            signature is None
            or not any(key.startswith("voice:") for key in structural)
        ):
            return 0.0
        distances = []
        for key in signature.values:
            if not key.startswith("distance:predicate:"):
                continue
            try:
                distances.append(
                    max(1, int(key.rsplit(":", 1)[-1]))
                )
            except ValueError:
                continue
        return 1.0 / min(distances) if distances else 0.0

    @staticmethod
    def _passive_perspective_fit(
        graph: QueryGraph,
        participant: Optional[Any],
    ) -> tuple[float, float, float]:
        """Return local support/conflict/pivot scores for a short passive.

        The calculation intentionally uses only observed morphology of the
        current event.  It is a query-local projection, never a role written
        back to the event graph.
        """
        relation = graph.trace.get("predicate_perspective_relation") or {}
        if (
            relation.get("perspective") != "PASSIVE_RESULT"
            or participant is None
        ):
            return 0.0, 0.0, 0.0
        observed_cases = {
            str(item.get("case") or "")
            for item in participant.mention.features.get(
                "morphology_alternatives", []
            )
        }
        observed_cases.add(str(participant.mention.features.get("case") or ""))
        # Surface-syncretic inanimate nouns retain their accusative analysis;
        # do not discard it merely because the parser chose nominative as its
        # default standalone reading.
        if "accs" in observed_cases:
            return 1.0, 0.0, 0.90
        if "nomn" in observed_cases:
            return 0.0, 1.0, 0.0
        return 0.0, 0.0, 0.0

    @staticmethod
    def _question_morphology_fit(
        graph: QueryGraph,
        signature: Optional[ObservationSignature],
    ) -> tuple[float, float, Dict[str, Any]]:
        """Score a binding against retained, rather than forced, case forms."""
        hypotheses = {
            key.removeprefix("case:"): float(value)
            for key, value in graph.gap_node.morphology_hypotheses.items()
            if key.startswith("case:")
        }
        if not hypotheses or signature is None:
            return 1.0, 0.0, {"mode": "not_applicable"}
        candidate_cases = {
            key.removeprefix("morph:case:")
            for key in signature.values
            if key.startswith("morph:case:")
        }
        compatible = {
            case: hypotheses[case]
            for case in candidate_cases & set(hypotheses)
        }
        ranked = sorted(hypotheses.items(), key=lambda item: item[1], reverse=True)
        dominant = (
            len(ranked) == 1
            or ranked[0][1] - ranked[1][1] >= 0.20
        )
        fit = max(compatible.values(), default=0.0)
        # A syncretic surface can support both cases (``яблоко``); once it
        # contains the dominant case it has no morphology conflict merely
        # because the retained hypothesis itself has confidence below 1.0.
        conflict = (
            0.0
            if dominant and ranked[0][0] in candidate_cases
            else (1.0 - fit) if dominant else 0.0
        )
        return fit, conflict, {
            "mode": "case_hypotheses",
            "question_cases": hypotheses,
            "candidate_cases": sorted(candidate_cases),
            "dominant": dominant,
            "compatible_cases": compatible,
        }

    @staticmethod
    def _requires_selection_margin(graph: QueryGraph) -> bool:
        cases = sorted(
            float(value)
            for key, value in graph.gap_node.morphology_hypotheses.items()
            if key.startswith("case:")
        )
        return (
            len(cases) >= 2
            and cases[-1] - cases[-2] < 0.20
            and not graph.known_nodes
        )

    @staticmethod
    def _component_candidates(
        graph: QueryGraph,
        matched: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        attached = next(
            (
                known for known in graph.known_nodes
                if known.id == graph.gap_node.attached_to_node_id
            ),
            None,
        )
        relevant = [
            participant for participant in matched
            if attached and (
                participant.mention.head_lemma == attached.head_lemma
                or participant.mention.entity_id == attached.entity_id
            )
        ]
        candidates: List[Dict[str, Any]] = []
        for participant in relevant:
            known_components = (
                GraphMatcher._component_lemmas(attached)
                if attached else set()
            )
            for component in participant.mention.components:
                if component.lemma in known_components:
                    continue
                candidates.append({
                    "node_id": component.id,
                    "concept_id": stable_id(
                        "component-concept",
                        component.lemma,
                    ),
                    "lemma": component.lemma,
                    "surface": component.surface,
                    "features": component.attachment_signature.as_dict(),
                    "signature": component.attachment_signature,
                    "slot_score": signature_similarity(
                        graph.gap_node.question_signature.values,
                        component.attachment_signature.values,
                    ),
                })
        return candidates

    @staticmethod
    def _type_constraint(gap: GapNode) -> Mapping[str, Any]:
        if gap.gap_kind != GapKind.EVENT_ATTACHMENT:
            return {}
        value = gap.evidence.get("type_constraint") or {}
        return value if isinstance(value, Mapping) else {}

    @classmethod
    def _matches_type_constraint(
        cls,
        gap: GapNode,
        *,
        lemma: str,
        features: Mapping[str, Any],
    ) -> bool:
        constraint = cls._type_constraint(gap)
        required_lemma = str(constraint.get("lemma") or "").casefold()
        if required_lemma and str(lemma).casefold() != required_lemma:
            return False
        required_number = str(constraint.get("number") or "")
        actual_number = str(features.get("number") or "")
        return not (
            required_number and actual_number and required_number != actual_number
        )

    @staticmethod
    def _required_relation(graph: QueryGraph) -> Mapping[str, Any]:
        return next(
            (
                edge for edge in graph.required_edges
                if edge.get("edge_type") == "VALUE_ATTACHED_TO_NODE"
            ),
            {},
        )

    def _search_multiple(
        self,
        graph: QueryGraph,
        *,
        limit: int,
        candidate_event_ids: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """Bind all requested gaps as one distinct participant configuration."""
        accepted: List[CandidateBinding] = []
        rejected: List[Dict[str, Any]] = []
        configurations: List[tuple[float, List[CandidateBinding], str]] = []
        anchor_id = str(graph.trace.get("event_anchor_id") or "")
        with self.repository.transaction() as conn:
            where_anchor = " AND e.id=?" if anchor_id else ""
            known_clause, known_params = self._known_event_prefilter(
                graph.known_nodes
            )
            params: tuple[Any, ...] = (
                graph.predicate.concept_id,
                *known_params,
            )
            if anchor_id:
                params += (anchor_id,)
            params += (max(1, min(int(limit), 512)),)
            rows = conn.execute(
                """SELECT e.id FROM graph_events e
                   JOIN knowledge_sources s ON s.id=e.source_id
                   WHERE e.predicate_concept_id=? AND s.status='CONFIRMED'
                     AND e.actuality='ACTUAL'""" + known_clause + where_anchor +
                " ORDER BY e.confidence DESC,e.created_at,e.id LIMIT ?",
                params,
            ).fetchall()
            for row in rows:
                if candidate_event_ids is not None and str(row["id"]) not in candidate_event_ids:
                    continue
                event = EventGraphPipeline.load_event(conn, str(row["id"]))
                matched, failure = self._match_known_nodes(graph.known_nodes, event)
                if failure:
                    rejected.append({"event_id": event.id, **failure})
                    continue
                assert matched is not None
                unavailable = {item.id for item in matched}
                alternatives: List[List[tuple[Any, float, float]]] = []
                for gap in graph.target_gaps:
                    choices: List[tuple[Any, float, float]] = []
                    type_constraint = self._type_constraint(gap)
                    cases = {
                        key.removeprefix("morph:case:")
                        for key in gap.question_signature.values
                        if key.startswith("morph:case:")
                    }
                    for participant in event.participants:
                        if participant.id in unavailable:
                            continue
                        if type_constraint and not self._matches_type_constraint(
                            gap,
                            lemma=participant.mention.head_lemma,
                            features=participant.mention.features,
                        ):
                            continue
                        observed_case = str(
                            participant.mention.features.get("case") or ""
                        )
                        if cases and observed_case and observed_case not in cases:
                            continue
                        slot, direct, _, _ = self._slot_scores(
                            replace(graph, gap_node=gap), participant
                        )
                        case_fit = 1.0 if not cases or observed_case in cases else 0.0
                        choices.append((participant, slot, case_fit))
                    alternatives.append(choices)
                if not all(alternatives):
                    rejected.append({
                        "event_id": event.id,
                        "status": "REJECTED",
                        "failed_constraint": (
                            "TYPE_CONSTRAINT"
                            if any(self._type_constraint(gap) and not choices
                                   for gap, choices in zip(graph.target_gaps, alternatives))
                            else "GAP_BINDING"
                        ),
                        "reason": "NOT_ALL_REQUESTED_GAPS_BOUND",
                    })
                    continue
                best: Optional[List[CandidateBinding]] = None
                best_score = -1.0
                for combination in product(*alternatives):
                    participants = [item[0] for item in combination]
                    if len({participant.id for participant in participants}) != len(participants):
                        continue
                    bindings: List[CandidateBinding] = []
                    for gap, (participant, slot, case_fit) in zip(
                        graph.target_gaps, combination
                    ):
                        (
                            _,
                            direct_slot_score,
                            learned_slot_score,
                            compatibility_state,
                        ) = self._slot_scores(
                            replace(graph, gap_node=gap),
                            participant,
                        )
                        total = max(0.0, min(1.0,
                            0.42 * self._known_node_score(graph.known_nodes, matched)
                            + 0.22 * max(0.16, slot)
                            + 0.16 * event.confidence + 0.20 * case_fit,
                        ))
                        bindings.append(CandidateBinding(
                            id=stable_id("binding", graph.id, event.id, gap.id, participant.id),
                            query_graph_id=graph.id,
                            event_id=event.id,
                            gap_node_id=gap.id,
                            resolved_node_id=participant.id,
                            resolved_concept_id=(participant.mention.entity_id or stable_id("entity", participant.mention.head_lemma)),
                            resolved_lemma=participant.mention.head_lemma,
                            resolved_surface=participant.mention.surface,
                            resolved_features={
                                **dict(participant.mention.features),
                                "preposition": participant.mention.preposition,
                            },
                            structural_score=self._known_node_score(graph.known_nodes, matched),
                            signature_score=max(0.16, slot),
                            evidence_score=event.confidence,
                            total_score=total,
                            status=BindingStatus.ACCEPTED,
                            evidence=({
                                "gap_kind": gap.gap_kind.value,
                                "configuration_binding": True,
                                "local_slot_ids": [
                                    hypothesis.local_slot_id
                                    for hypothesis in participant.slot_hypotheses
                                ],
                                "score_components": {
                                    "slot_similarity": direct_slot_score,
                                    "learned_substitution_score": learned_slot_score,
                                    "question_morphology": case_fit,
                                },
                                "supporting_event_ids": [event.id],
                                "independent_source_count": 1,
                            },),
                            slot_compatibility_state=compatibility_state,
                        ))
                    score = sum(binding.total_score for binding in bindings) / len(bindings)
                    if score > best_score:
                        best, best_score = bindings, score
                if best:
                    accepted.extend(best)
                    configurations.append((best_score, best, event.source_surface))
        if not configurations:
            return {
                "accepted": accepted,
                "rejected": rejected,
                "selected_bindings": [],
                "status": AnswerStatus.UNRESOLVED.value,
                "reason": "INCOMPLETE_BINDING_CONFIGURATION",
            }
        configuration_event_ids = {
            bindings[0].event_id
            for _, bindings, _ in configurations
            if bindings
        }
        selected_configurations = (
            configurations if len(configuration_event_ids) > 1
            else [max(configurations, key=lambda item: item[0])]
        )
        selected_bindings: List[CandidateBinding] = []
        configuration_views: List[Dict[str, Any]] = []
        for score, bindings, source_surface in selected_configurations:
            event_id = bindings[0].event_id
            configuration_id = stable_id("binding-configuration", graph.id, event_id)
            scoped_bindings = [replace(
                binding,
                status=BindingStatus.SELECTED,
                selection_status="SELECTED",
                configuration_id=configuration_id,
            ) for binding in bindings]
            selected_bindings.extend(scoped_bindings)
            configuration_views.append({
                "configuration_id": configuration_id,
                "query_graph_id": graph.id,
                "event_id": event_id,
                "bindings_by_gap": {
                    binding.gap_node_id: binding.as_dict()
                    for binding in scoped_bindings
                },
                "all_required_gaps_bound": True,
                "distinct_node_count": len({
                    binding.resolved_node_id for binding in scoped_bindings
                }),
                "configuration_score": max(0.0, min(1.0, score)),
                "status": "SELECTED",
                "source_surface": source_surface,
            })
        selected_ids = {item.id for item in selected_bindings}
        accepted = [
            next((selected for selected in selected_bindings if selected.id == item.id), item)
            if item.id in selected_ids else item
            for item in accepted
        ]
        result = {
            "accepted": accepted,
            "rejected": rejected,
            "selected_bindings": selected_bindings,
            "status": AnswerStatus.RESOLVED.value,
            "selection_scope": (
                "MULTI_EVENT" if len(configuration_views) > 1 else "SINGLE_EVENT"
            ),
            "binding_configurations": configuration_views,
        }
        return result

    def search(
        self,
        graph: QueryGraph,
        *,
        limit: int = 128,
    ) -> Dict[str, Any]:
        implicit_relation = bool(
            (graph.trace.get("predicate_hypothesis") or {}).get("predicate_origin")
            == "IMPLICIT_RELATIONAL"
        )
        component_anchor_query = bool(
            graph.gap_node.gap_kind == GapKind.NODE_COMPONENT
            and graph.gap_node.attached_to_node_id
            and graph.known_nodes
        )
        if graph.status != GraphStatus.READY or (
            not graph.predicate and not implicit_relation
            and not component_anchor_query
        ):
            return {
                "accepted": [],
                "rejected": [],
                "selected_bindings": [],
                "status": AnswerStatus.UNRESOLVED.value,
            }
        swarm = self.swarms.discover(graph)
        candidate_event_ids = set(swarm.get("candidate_event_ids") or [])
        if len(graph.target_gaps) > 1:
            result = self._search_multiple(
                graph, limit=limit, candidate_event_ids=candidate_event_ids,
            )
            result["swarm"] = swarm
            return result
        accepted: List[CandidateBinding] = []
        rejected: List[Dict[str, Any]] = []
        with self.repository.transaction() as conn:
            anchor_id = str(graph.trace.get("event_anchor_id") or "")
            where_anchor = " AND e.id=?" if anchor_id else ""
            anchored_perspective_check = bool(
                anchor_id
                and graph.trace.get("continuation_mode") == "REFERENTIAL"
            )
            predicate_clause = "" if (
                anchored_perspective_check or implicit_relation or component_anchor_query
            ) else (
                " AND e.predicate_concept_id=?"
            )
            known_clause, known_params = self._known_event_prefilter(
                graph.known_nodes
            )
            params: tuple[Any, ...] = () if (
                anchored_perspective_check or implicit_relation or component_anchor_query
            ) else (
                graph.predicate.concept_id,
            )
            params += known_params
            if anchor_id:
                params += (anchor_id,)
            params += (max(1, min(int(limit), 512)),)
            rows = conn.execute(
                """SELECT e.id
                   FROM graph_events e
                   JOIN knowledge_sources s ON s.id=e.source_id
                   WHERE s.status='CONFIRMED'
                     AND e.actuality='ACTUAL'
                   """ + predicate_clause + known_clause + where_anchor + """
                   ORDER BY e.confidence DESC,e.created_at,e.id
                   LIMIT ?""",
                params,
            ).fetchall()
            for row in rows:
                if candidate_event_ids and str(row["id"]) not in candidate_event_ids:
                    continue
                event = EventGraphPipeline.load_event(conn, str(row["id"]))
                query_polarity = str(
                    graph.trace.get("query_polarity") or "POSITIVE"
                )
                if (
                    graph.gap_node.gap_kind != GapKind.BOOLEAN_RESULT
                    and event.polarity != query_polarity
                ):
                    rejected.append({
                        "event_id": event.id,
                        "status": "REJECTED",
                        "failed_constraint": "POLARITY",
                        "reason": "EVENT_POLARITY_MISMATCH",
                    })
                    continue
                matched, failure = self._match_known_nodes(
                    graph.known_nodes,
                    event,
                    allow_anchor_preposition_variance=anchored_perspective_check,
                )
                if failure:
                    rejected.append({
                        "event_id": event.id,
                        **failure,
                    })
                    continue
                assert matched is not None
                matched_ids = {participant.id for participant in matched}
                candidates: List[Dict[str, Any]] = []
                if graph.gap_node.gap_kind == GapKind.NODE_COMPONENT:
                    candidates = self._component_candidates(graph, matched)
                elif graph.gap_node.gap_kind == GapKind.BOOLEAN_RESULT:
                    surface = (
                        "Да"
                        if event.polarity == query_polarity
                        else "Нет"
                    )
                    candidates = [{
                        "node_id": stable_id("boolean", event.id, event.polarity),
                        "concept_id": stable_id("boolean-concept", event.polarity),
                        "lemma": surface.casefold(),
                        "surface": surface,
                        "features": {"polarity": event.polarity},
                        "slot_score": 1.0,
                    }]
                elif graph.gap_node.gap_kind == GapKind.WHOLE_EVENT:
                    candidates = [{
                        "node_id": event.id,
                        "concept_id": event.predicate.concept_id,
                        "lemma": event.predicate.lemma,
                        "surface": event.source_surface,
                        "features": dict(event.predicate.features),
                        "slot_score": 1.0,
                    }]
                else:
                    required_preposition = str(
                        self._required_relation(graph).get("preposition") or ""
                    )
                    for participant in event.participants:
                        if participant.id in matched_ids:
                            continue
                        if (
                            graph.gap_node.gap_kind == GapKind.RELATION_VALUE
                            and required_preposition
                            and participant.mention.preposition
                            != required_preposition
                        ):
                            continue
                        if (
                            graph.gap_node.gap_kind == GapKind.QUANTITY_VALUE
                            and "morph:part_of_speech:NUMR"
                            not in participant.observation_signature.values
                        ):
                            continue
                        slot_score, direct_score, learned_score, compatibility_state = (
                            self._slot_scores(graph, participant)
                        )
                        effective_slot_score = (
                            direct_score
                            if compatibility_state == "below_threshold"
                            else slot_score
                        )
                        candidates.append({
                            "participant": participant,
                            "node_id": participant.id,
                            "concept_id": (
                                participant.mention.entity_id
                                or stable_id(
                                    "entity",
                                    participant.mention.head_lemma,
                                )
                            ),
                            "lemma": participant.mention.head_lemma,
                            "surface": participant.mention.surface,
                            "features": {
                                **dict(participant.mention.features),
                                "preposition": participant.mention.preposition,
                            },
                            "signature": participant.observation_signature,
                            # A sub-threshold learned slot is retained in
                            # evidence but cannot improve ranking.  Otherwise
                            # weak historical proximity can beat an equally
                            # valid structural candidate merely by a few
                            # thousandths.
                            "slot_score": effective_slot_score,
                            "direct_slot_similarity": direct_score,
                            "learned_substitution_score": learned_score,
                            "slot_compatibility_state": compatibility_state,
                            "local_slot_ids": [
                                hypothesis.local_slot_id
                                for hypothesis in participant.slot_hypotheses
                            ],
                        })
                if not candidates:
                    rejected.append({
                        "event_id": event.id,
                        "status": "REJECTED",
                        "failed_constraint": "GAP_BINDING",
                        "reason": "NO_COMPATIBLE_UNBOUND_NODE",
                    })
                    continue
                type_constraint = self._type_constraint(graph.gap_node)
                if type_constraint:
                    typed_candidates = [
                        candidate for candidate in candidates
                        if self._matches_type_constraint(
                            graph.gap_node,
                            lemma=str(candidate["lemma"]),
                            features=dict(candidate.get("features") or {}),
                        )
                    ]
                    if not typed_candidates:
                        rejected.append({
                            "event_id": event.id,
                            "status": "REJECTED",
                            "failed_constraint": "TYPE_CONSTRAINT",
                            "reason": "TYPED_EVENT_ATTACHMENT_MISMATCH",
                            "type_constraint": dict(type_constraint),
                        })
                        continue
                    candidates = typed_candidates
                for candidate in candidates:
                    if self._excluded(
                        graph,
                        node_id=str(candidate["node_id"]),
                        concept_id=str(candidate["concept_id"]),
                        lemma=str(candidate["lemma"]),
                    ):
                        rejected.append({
                            "event_id": event.id,
                            "resolved_node_id": candidate["node_id"],
                            "status": "REJECTED",
                            "failed_constraint": "EXCLUDED_BINDING",
                            "reason": "VALUE_ALREADY_RETURNED",
                        })
                        continue
                    structural_score = self._known_node_score(
                        graph.known_nodes,
                        matched,
                    )
                    signature_score = max(
                        0.16,
                        min(1.0, float(candidate.get("slot_score", 0.0))),
                    )
                    evidence_score = event.confidence
                    structural_attachment_fit = (
                        self._structural_attachment_fit(
                            graph,
                            candidate.get("signature"),
                        )
                    )
                    morphology_fit, morphology_conflict, morphology_evidence = (
                        self._question_morphology_fit(
                            graph,
                            candidate.get("signature"),
                        )
                    )
                    perspective_support, perspective_conflict, surface_pivot_support = (
                        self._passive_perspective_fit(
                            graph, candidate.get("participant"),
                        )
                    )
                    requested_gap_conflict = (
                        0.12 * (1.0 - structural_attachment_fit)
                        if candidate.get("slot_compatibility_state")
                        == "below_threshold"
                        else 0.0
                    )
                    # Slot proximity is useful but cannot eclipse explicit
                    # morphology or a tightly attached structural candidate.
                    total = max(0.0, min(
                        1.0,
                        0.42 * structural_score
                        + 0.24 * signature_score
                        + 0.16 * evidence_score
                        + 0.12 * morphology_fit
                        + 0.06 * structural_attachment_fit
                        + 0.18 * perspective_support
                        + 0.06 * surface_pivot_support
                        - 0.28 * morphology_conflict
                        - 0.34 * perspective_conflict
                        - requested_gap_conflict,
                    ))
                    binding = CandidateBinding(
                        id=stable_id(
                            "binding",
                            graph.id,
                            event.id,
                            candidate["node_id"],
                        ),
                        query_graph_id=graph.id,
                        event_id=event.id,
                        gap_node_id=graph.gap_node.id,
                        resolved_node_id=str(candidate["node_id"]),
                        resolved_concept_id=str(candidate["concept_id"]),
                        resolved_lemma=str(candidate["lemma"]),
                        resolved_surface=str(candidate["surface"]),
                        resolved_features=dict(candidate.get("features") or {}),
                        structural_score=structural_score,
                        signature_score=signature_score,
                        evidence_score=evidence_score,
                        total_score=total,
                        status=BindingStatus.ACCEPTED,
                        evidence=({
                            "known_node_count": len(matched),
                            "gap_kind": graph.gap_node.gap_kind.value,
                            "local_slot_ids": candidate.get(
                                "local_slot_ids",
                                [],
                            ),
                            "event_confidence": event.confidence,
                            "anchored_event_evidence": anchored_perspective_check,
                            "score_components": {
                                "event_score": evidence_score,
                                "slot_similarity": candidate.get(
                                    "direct_slot_similarity", signature_score,
                                ),
                                "learned_substitution_score": candidate.get(
                                    "learned_substitution_score", 0.0,
                                ),
                                "question_morphology": morphology_fit,
                                "structural_attachment": structural_attachment_fit,
                                "morphology_conflict": morphology_conflict,
                                "requested_gap_conflict": requested_gap_conflict,
                                "perspective_support": perspective_support,
                                "perspective_conflict": perspective_conflict,
                                "surface_pivot_support": surface_pivot_support,
                            },
                            "question_morphology_evidence": morphology_evidence,
                            "slot_compatibility": {
                                "state": candidate.get(
                                    "slot_compatibility_state", "fallback",
                                ),
                                "reason": (
                                    "LOCAL_SLOT_IN_COMPATIBLE_HYPOTHESES"
                                    if candidate.get("slot_compatibility_state") == "compatible"
                                    else "LOCAL_SLOT_NOT_IN_COMPATIBLE_HYPOTHESES"
                                    if candidate.get("slot_compatibility_state") == "below_threshold"
                                    else "NO_LOCAL_SLOT_HYPOTHESIS"
                                ),
                            },
                            "independent_key": str(conn.execute(
                                """SELECT independent_key
                                   FROM knowledge_sources
                                   WHERE id=(
                                     SELECT source_id FROM graph_events
                                     WHERE id=?
                                   )""",
                                (event.id,),
                            ).fetchone()[0]),
                        },),
                        slot_compatibility_state=str(
                            candidate.get("slot_compatibility_state")
                            or "fallback"
                        ),
                    )
                    accepted.append(binding)
            # Canonical deduplication keeps repeated mentions from becoming
            # independent values.  The strongest source remains visible.
            grouped: Dict[tuple[str, str], List[CandidateBinding]] = {}
            for binding in accepted:
                key = (
                    binding.resolved_concept_id,
                    binding.resolved_lemma,
                )
                grouped.setdefault(key, []).append(binding)
            deduplicated: Dict[tuple[str, str], CandidateBinding] = {}
            for key, group in grouped.items():
                current = max(
                    group,
                    key=lambda item: (
                        item.total_score,
                        item.evidence_score,
                    ),
                )
                supporting_event_ids = sorted({item.event_id for item in group})
                independent_keys = sorted({
                    str(evidence.get("independent_key") or "")
                    for item in group
                    for evidence in item.evidence
                    if evidence.get("independent_key")
                })
                current = replace(
                    current,
                    evidence=tuple(current.evidence) + ({
                        "independent_source_count": len(independent_keys),
                        "independent_keys": independent_keys,
                        "supporting_event_ids": supporting_event_ids,
                        "unique_event_count": len(supporting_event_ids),
                    },),
                )
                deduplicated[key] = current
            accepted = sorted(
                deduplicated.values(),
                key=lambda item: (item.total_score, item.evidence_score),
                reverse=True,
            )
            selected: Optional[CandidateBinding] = None
            status = AnswerStatus.UNRESOLVED
            multiple_unconstrained_events = (
                len({item.event_id for item in accepted}) > 1
                and not graph.known_nodes
                and not anchor_id
            )
            unstable_leading_margin = bool(
                len(accepted) > 1
                and accepted[0].total_score - accepted[1].total_score
                < self.SELECTION_MARGIN
                and self._requires_selection_margin(graph)
            )
            reason = ""
            if accepted and (
                multiple_unconstrained_events
                or unstable_leading_margin
            ):
                status = AnswerStatus.AMBIGUOUS_BINDING
                reason = (
                    "MULTIPLE_UNCONSTRAINED_EVENTS"
                    if multiple_unconstrained_events
                    else "UNSTABLE_SELECTION_MARGIN"
                )
            elif accepted:
                selected = replace(
                    accepted[0],
                    status=BindingStatus.SELECTED,
                )
                accepted[0] = selected
                status = AnswerStatus.RESOLVED
            result = {
                "accepted": accepted,
                "rejected": rejected,
                "selected_bindings": [selected] if selected else [],
                "status": status.value,
                "reason": reason,
            }
            result["swarm"] = swarm
            return result


class GraphResponsePlanner:
    """Generate and validate answers from the selected gap binding."""

    def __init__(self, morphology: Any) -> None:
        self.morphology = morphology

    @staticmethod
    def _punctuate(surface: str) -> str:
        value = str(surface or "").strip()
        if not value:
            return value
        return value if value[-1:] in ".?!" else value + "."

    def _realize_binding(
        self,
        graph: QueryGraph,
        binding: CandidateBinding,
    ) -> str:
        surface = binding.resolved_surface
        if (
            graph.gap_node.gap_kind != GapKind.EVENT_ATTACHMENT
            or " " in surface.strip()
        ):
            return self._with_preposition(graph, binding, surface)
        target_features: Dict[str, str] = {}
        # Interrogative pronouns have their own lexical gender (``кто`` is
        # masculine, ``что`` neuter); that is not a constraint on the entity
        # being realized.  Case and number are the transferable observations.
        for feature in ("case", "number"):
            prefix = f"morph:{feature}:"
            candidates = [
                (float(weight), key[len(prefix):])
                for key, weight in graph.gap_node.question_signature.values.items()
                if key.startswith(prefix)
            ]
            if candidates:
                target_features[feature] = max(candidates)[1]
        if not target_features:
            return self._with_preposition(graph, binding, surface)
        generated = self.morphology.inflect(
            binding.resolved_lemma,
            target_features,
        )
        if surface[:1].isupper() and generated:
            generated = generated[:1].upper() + generated[1:]
        return self._with_preposition(graph, binding, generated or surface)

    @staticmethod
    def _with_preposition(
        graph: QueryGraph,
        binding: CandidateBinding,
        surface: str,
    ) -> str:
        if graph.gap_node.gap_kind != GapKind.EVENT_PROPERTY:
            return surface
        preposition = str(binding.resolved_features.get("preposition") or "")
        if not preposition or not surface:
            return surface
        phrase = f"{preposition} {surface}"
        return phrase[:1].upper() + phrase[1:]

    def plan(
        self,
        graph: QueryGraph,
        search: Mapping[str, Any],
        *,
        event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        selected_bindings = [
            item for item in search.get("selected_bindings", [])
            if isinstance(item, CandidateBinding)
        ]
        selected = selected_bindings[0] if selected_bindings else None
        status = str(search.get("status") or AnswerStatus.UNRESOLVED.value)
        selection_scope = str(search.get("selection_scope") or "SINGLE_EVENT")
        if status in {
            AnswerStatus.AMBIGUOUS.value,
            AnswerStatus.AMBIGUOUS_BINDING.value,
        }:
            ambiguity_candidates = [
                binding for binding in search.get("accepted", [])
                if isinstance(binding, CandidateBinding)
            ]
            gap_ids = {binding.gap_node_id for binding in ambiguity_candidates}
            if len(gap_ids) > 1:
                return {
                    "status": AnswerStatus.CONFLICTED.value,
                    "short_answer": None,
                    "full_answer": None,
                    "surface": None,
                    "validation": {
                        "valid": False,
                        "reason": "CROSS_GAP_AMBIGUITY_ERROR",
                    },
                }
            alternatives = [
                binding.resolved_surface
                for binding in search.get("accepted", [])[:2]
            ]
            return {
                "status": status,
                "short_answer": None,
                "full_answer": None,
                "surface": (
                    "Неясно, какое значение связать с пропуском: "
                    + " или ".join(alternatives)
                    + "."
                ),
                "validation": {
                    "valid": True,
                    "reason": (
                        search.get("reason")
                        or "leading bindings have no stable margin"
                    ),
                },
            }
        if (
            status == AnswerStatus.RESOLVED.value
            and selection_scope == "MULTI_EVENT"
        ):
            configurations = list(search.get("binding_configurations") or [])
            surfaces = [
                self._punctuate(str(item.get("source_surface") or ""))
                for item in configurations
                if item.get("source_surface")
            ]
            complete = all(
                bool(item.get("all_required_gaps_bound"))
                and len(item.get("bindings_by_gap") or {}) == len(graph.target_gaps)
                for item in configurations
            )
            # Keep each source event intact: a concise stylistic realization
            # can be added later, but no renderer may split a pair into
            # cross-GAP alternatives.
            surface = " ".join(surfaces)
            valid = bool(surfaces) and complete
            return {
                "status": AnswerStatus.RESOLVED.value if valid else AnswerStatus.BUILD_FAILED.value,
                "short_answer": surface if valid else None,
                "full_answer": surface if valid else None,
                "surface": surface if valid else None,
                "resolution_class": "MULTI_EVENT_RESOLVED" if valid else "",
                "selection_scope": "MULTI_EVENT",
                "binding_configurations": configurations,
                "selected_bindings": [item.as_dict() for item in selected_bindings],
                "provenance": {
                    "source_event_ids": [
                        str(item.get("event_id")) for item in configurations
                    ],
                    "independent_source_count": len(configurations),
                },
                "validation": {
                    "valid": valid,
                    "configuration_count": len(configurations),
                    "all_configurations_complete": complete,
                    "selection_scope": "MULTI_EVENT",
                },
                "versions": ModelVersions().as_dict(),
            }
        if not isinstance(selected, CandidateBinding):
            return {
                "status": AnswerStatus.UNRESOLVED.value,
                "short_answer": None,
                "full_answer": None,
                "surface": "В доступной памяти других подходящих значений нет.",
                "selected_bindings": [],
                "validation": {
                    "valid": True,
                    "reason": (
                        search.get("reason")
                        or "no admitted gap binding"
                    ),
                },
            }
        if len(graph.target_gaps) > 1:
            bound_gap_ids = {item.gap_node_id for item in selected_bindings}
            missing = [
                gap.id for gap in graph.target_gaps if gap.id not in bound_gap_ids
            ]
            distinct = len({item.resolved_node_id for item in selected_bindings}) == len(selected_bindings)
            same_event = len({
                item.event_id for item in selected_bindings
            }) == 1
            full = self._punctuate(event.source_surface) if event else None
            contains_every_binding = bool(full) and all(
                item.resolved_surface.casefold() in full.casefold()
                for item in selected_bindings
            )
            valid = (
                not missing
                and distinct
                and same_event
                and full is not None
                and contains_every_binding
            )
            return {
                "status": AnswerStatus.RESOLVED.value if valid else AnswerStatus.BUILD_FAILED.value,
                "short_answer": full,
                "full_answer": full,
                "surface": full if valid else None,
                "selected_bindings": [item.as_dict() for item in selected_bindings],
                "provenance": {"source_event_ids": [selected.event_id], "independent_source_count": 1},
                "validation": {
                    "valid": valid,
                    "all_requested_gaps_bound": not missing,
                    "bindings_are_distinct_when_required": distinct,
                    "all_bindings_belong_to_same_event": same_event,
                    "surface_contains_every_binding": contains_every_binding,
                    "event_identity": event.id if event else None,
                },
                "versions": ModelVersions().as_dict(),
            }
        short = self._punctuate(self._realize_binding(graph, selected))
        full = self._punctuate(event.source_surface) if event else None
        validation = self.validate(graph, selected, short, event=event)
        overall = (
            AnswerStatus.RESOLVED
            if validation["valid"] and full
            else AnswerStatus.PARTIALLY_RESOLVED
            if validation["valid"]
            else AnswerStatus.BUILD_FAILED
        )
        return {
            "status": overall.value,
            "short_answer": short,
            "full_answer": full,
            "surface": short if validation["valid"] else None,
            "selected_bindings": [selected.as_dict()],
            "provenance": {
                # The selected answer is causally grounded in the selected
                # event.  Corroborating events remain evidence, but cannot be
                # mixed into answer provenance or downstream learning lineage.
                "source_event_ids": [selected.event_id],
                "supporting_event_ids": sorted({
                    event_id
                    for evidence in selected.evidence
                    for event_id in evidence.get("supporting_event_ids", [])
                } or {selected.event_id}),
                "independent_source_count": max(1, max(
                    (
                        int(evidence.get("independent_source_count", 0))
                        for evidence in selected.evidence
                    ),
                    default=1,
                )),
            },
            "validation": validation,
            "versions": ModelVersions().as_dict(),
        }

    def validate(
        self,
        graph: QueryGraph,
        binding: CandidateBinding,
        surface: str,
        *,
        event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        failures: List[str] = []
        if binding.gap_node_id != graph.gap_node.id:
            failures.append("BINDING_TARGETS_ANOTHER_GAP")
        if not binding.resolved_node_id or not binding.resolved_lemma:
            failures.append("EMPTY_RESOLVED_NODE")
        if GraphMatcher._excluded(
            graph,
            node_id=binding.resolved_node_id,
            concept_id=binding.resolved_concept_id,
            lemma=binding.resolved_lemma,
        ):
            failures.append("EXCLUDED_VALUE_RETURNED")
        realized_tokens = [
            token.strip(".,!?;:()[]{}«»\"").casefold()
            for token in surface.split()
            if token.strip(".,!?;:()[]{}«»\"")
        ]
        realized_lemmas = {
            hypothesis.lemma.casefold()
            for token in realized_tokens
            for hypothesis in self.morphology.parse_variants(token)
        }
        if binding.resolved_lemma.casefold() not in realized_lemmas:
            failures.append("SURFACE_LOST_RESOLVED_VALUE")
        if event and event.id != binding.event_id:
            failures.append("FULL_ANSWER_USES_ANOTHER_EVENT")
        return {
            "valid": not failures,
            "failures": failures,
            "checks": [
                "gap_binding",
                "known_nodes_retained_by_event_admission",
                "excluded_values",
                "surface_contains_binding",
                "event_identity",
            ],
        }


def persist_query_result(
    conn: Any,
    graph: QueryGraph,
    source_text: str,
    *,
    hive_id: Optional[str],
    search: Mapping[str, Any],
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO query_graphs
           (id,hive_id,source_text,graph_json,status,continuation_of,
            query_graph_version,created_at)
           VALUES(?,?,?,?,?,?,?,?)""",
        (
            graph.id,
            hive_id,
            source_text,
            encode(graph.as_dict()),
            graph.status.value,
            graph.continuation_of,
            graph.versions.query_graph,
            utcnow(),
        ),
    )
    for binding in search.get("accepted", []):
        conn.execute(
            """INSERT OR REPLACE INTO candidate_bindings
               (id,query_graph_id,event_id,gap_node_id,resolved_node_id,
                resolved_concept_id,resolved_lemma,resolved_surface,
                resolved_features_json,structural_score,signature_score,
                evidence_score,total_score,status,failed_constraint,
                evidence_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                binding.id,
                binding.query_graph_id,
                binding.event_id,
                binding.gap_node_id,
                binding.resolved_node_id,
                binding.resolved_concept_id,
                binding.resolved_lemma,
                binding.resolved_surface,
                encode(binding.resolved_features),
                binding.structural_score,
                binding.signature_score,
                binding.evidence_score,
                binding.total_score,
                binding.status.value,
                binding.failed_constraint,
                encode(binding.evidence),
                utcnow(),
            ),
        )
