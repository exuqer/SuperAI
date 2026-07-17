"""Role distributions from grammar, learned constructions and spatial form."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..language.noun_phrase_parser import MentionDraft
from .predicate_valency_profile import PredicateValencyProfile
from .spatial_relation_resolver import SpatialRelationResolver


class RoleHypothesisResolver:
    DEFAULT_WEIGHTS = {
        "grammar_prior": .58,
        "construction_support": .42,
    }
    GRAMMAR_PRIORS: Dict[str, Sequence[tuple[str, float]]] = {
        "subject": (("agent", .72), ("theme", .46), ("cause", .34)),
        "subject_intransitive": (("theme", .68), ("agent", .58), ("cause", .30)),
        "direct_object": (("patient", .74), ("theme", .64), ("object", .56)),
        "indirect_object": (("recipient", .78), ("experiencer", .50), ("object", .30)),
        "source_oblique": (("source", .92), ("location", .28)),
        "destination_oblique": (("destination", .92), ("location", .42)),
        "location_oblique": (("location", .90), ("destination", .32)),
        "reference_oblique": (("location", .55), ("object", .24)),
        "instrumental": (("instrument", .76), ("material", .52), ("theme", .36)),
        "purpose_oblique": (("purpose", .90),),
        "cause_oblique": (("cause", .88),),
        "object": (("object", .55), ("theme", .45), ("patient", .40)),
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.spatial = SpatialRelationResolver()
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def grammatical_slot(
        self,
        mention: MentionDraft,
        predicate_index: int,
        has_preverbal_subject: bool,
        *,
        predicate_profile: Optional[Dict[str, float]] = None,
    ) -> str:
        case = mention.features.get("case")
        prep = mention.preposition.casefold().split()[-1] if mention.preposition else ""
        if mention.relation_function in {
            "location", "destination", "reference", "instrument", "cause",
        }:
            return {
                "location": "location_oblique",
                "destination": "destination_oblique",
                "reference": "reference_oblique",
                "instrument": "instrumental",
                "cause": "cause_oblique",
            }[mention.relation_function]
        spatial = self.spatial.resolve(
            preposition=mention.preposition,
            grammatical_case=case,
            relation_function=mention.relation_function,
            predicate_profile=predicate_profile,
        )
        best_spatial = max(spatial, key=spatial.get)
        if spatial[best_spatial] >= .72:
            return {
                "source": "source_oblique",
                "destination": "destination_oblique",
                "location": "location_oblique",
                "reference": "reference_oblique",
            }[best_spatial]
        if prep in {"для"}:
            return "purpose_oblique"
        if prep in {"к", "ко"} and case in {"datv", None}:
            return "indirect_object"
        if case == "datv":
            return "indirect_object"
        if case == "ablt":
            return "instrumental"
        if not prep and case in {"accs", "gent"}:
            return "direct_object"
        if predicate_index < 0 and case == "nomn":
            # In an elliptical relational clause ("X из Y", "X на Y") the
            # nominative phrase is the related theme, not an asserted agent.
            return "object"
        if case == "nomn" and mention.start < predicate_index:
            return "subject"
        if case == "nomn" and mention.start > predicate_index and has_preverbal_subject:
            return "direct_object"
        if case == "nomn":
            return "subject"
        if mention.start < predicate_index:
            return "subject"
        return "object"

    @staticmethod
    def _construction_support(
        conn: Any,
        predicate_lemma: str,
        slot: str,
    ) -> Dict[str, float]:
        if not conn or not predicate_lemma:
            return {}
        return PredicateValencyProfile.load(
            conn, predicate_lemma
        ).distribution(slot)

    def hypotheses(
        self,
        slot: str,
        has_direct_object: bool,
        *,
        predicate_lemma: str = "",
        conn: Any = None,
    ) -> List[Dict[str, Any]]:
        prior_slot = (
            "subject"
            if slot == "subject" and has_direct_object
            else "subject_intransitive"
            if slot == "subject"
            else slot
        )
        priors = dict(self.GRAMMAR_PRIORS.get(prior_slot, (("object", .5),)))
        construction = self._construction_support(conn, predicate_lemma, slot)
        roles = set(priors) | set(construction)
        scored: List[Dict[str, Any]] = []
        for role in roles:
            prior = priors.get(role, .18)
            learned = construction.get(role)
            confidence = (
                prior
                + self.weights["construction_support"]
                * learned
                * (1.0 - prior)
                if learned is not None
                else prior
            )
            evidence = [f"grammar_slot:{slot}"]
            source_type = "grammar_prior"
            if learned is not None:
                evidence.append("predicate_valency_profile")
                source_type = "grammar+construction"
            scored.append({
                "role": role,
                "confidence": min(1.0, confidence),
                "score": min(1.0, confidence),
                "source_type": source_type,
                "evidence": evidence,
            })
        return sorted(
            scored,
            key=lambda item: (-float(item["confidence"]), str(item["role"])),
        )

    def resolve_mentions(
        self,
        mentions: Sequence[MentionDraft],
        *,
        predicate_index: int,
        predicate_lemma: str = "",
        conn: Any = None,
    ) -> List[Dict[str, Any]]:
        profile = self.spatial.predicate_profile(conn, predicate_lemma)
        has_preverbal_subject = any(
            mention.start < predicate_index
            and mention.features.get("case") in {"nomn", None}
            for mention in mentions
        )
        slots = [
            self.grammatical_slot(
                mention,
                predicate_index,
                has_preverbal_subject,
                predicate_profile=profile,
            )
            for mention in mentions
        ]
        has_direct_object = "direct_object" in slots
        return [
            {
                "mention": mention,
                "grammatical_slot": slot,
                "hypotheses": self.hypotheses(
                    slot,
                    has_direct_object,
                    predicate_lemma=predicate_lemma,
                    conn=conn,
                ),
            }
            for mention, slot in zip(mentions, slots)
        ]
