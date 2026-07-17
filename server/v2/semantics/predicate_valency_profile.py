"""Learned predicate valency projected from construction observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PredicateValencyProfile:
    predicate_lemma: str
    argument_patterns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    evidence_count: int = 0

    @classmethod
    def load(cls, conn: Any, predicate_lemma: str) -> "PredicateValencyProfile":
        profile = cls(predicate_lemma=predicate_lemma)
        if not conn or not predicate_lemma:
            return profile
        rows = conn.execute(
            """SELECT ca.grammatical_slot,ca.semantic_role,ca.confidence,
                      ct.confidence AS template_confidence,ct.evidence_count
               FROM construction_arguments ca
               JOIN construction_templates ct ON ct.id=ca.construction_id
               WHERE ct.predicate_lemma=?
               ORDER BY ct.confidence DESC,ca.confidence DESC""",
            (predicate_lemma,),
        ).fetchall()
        profile.evidence_count = max(
            (int(row["evidence_count"]) for row in rows),
            default=0,
        )
        for row in rows:
            slot = str(row["grammatical_slot"])
            role = str(row["semantic_role"])
            evidence_factor = min(
                1.0, .45 + .12 * int(row["evidence_count"])
            )
            score = (
                float(row["confidence"])
                * float(row["template_confidence"])
                * evidence_factor
            )
            distribution = profile.argument_patterns.setdefault(slot, {})
            distribution[role] = max(distribution.get(role, 0.0), score)
        return profile

    def distribution(self, grammatical_slot: str) -> Dict[str, float]:
        return dict(self.argument_patterns.get(grammatical_slot, {}))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "predicate_lemma": self.predicate_lemma,
            "argument_patterns": {
                slot: {"role_distribution": dict(distribution)}
                for slot, distribution in self.argument_patterns.items()
            },
            "evidence_count": self.evidence_count,
        }
