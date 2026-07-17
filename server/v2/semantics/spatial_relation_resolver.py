"""Probabilistic spatial role resolution from construction evidence."""

from __future__ import annotations

from typing import Any, Dict, Optional


class SpatialRelationResolver:
    SOURCE_PREPOSITIONS = {"из", "от", "с", "со"}
    DESTINATION_PREPOSITIONS = {"в", "во", "на", "к", "ко"}
    LOCATION_PREPOSITIONS = {
        "в", "во", "на", "у", "около", "возле", "рядом", "под", "над",
    }

    @staticmethod
    def predicate_profile(conn: Any, predicate_lemma: str) -> Dict[str, float]:
        if not conn or not predicate_lemma:
            return {"motion_score": .5, "state_score": .5, "evidence_count": 0}
        rows = conn.execute(
            """SELECT ca.semantic_role,ca.confidence,ct.evidence_count
               FROM construction_arguments ca
               JOIN construction_templates ct ON ct.id=ca.construction_id
               WHERE ct.predicate_lemma=?
                 AND ca.semantic_role IN ('source','destination','location')""",
            (predicate_lemma,),
        ).fetchall()
        motion = sum(
            float(row["confidence"]) * max(1, int(row["evidence_count"]))
            for row in rows
            if row["semantic_role"] in {"source", "destination"}
        )
        state = sum(
            float(row["confidence"]) * max(1, int(row["evidence_count"]))
            for row in rows
            if row["semantic_role"] == "location"
        )
        total = motion + state
        if total <= 0:
            return {"motion_score": .5, "state_score": .5, "evidence_count": 0}
        return {
            "motion_score": motion / total,
            "state_score": state / total,
            "evidence_count": len(rows),
        }

    def resolve(
        self,
        *,
        preposition: str,
        grammatical_case: Optional[str],
        relation_function: Optional[str] = None,
        predicate_profile: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        prep = preposition.casefold().split()[-1] if preposition else ""
        profile = predicate_profile or {
            "motion_score": .5,
            "state_score": .5,
        }
        scores = {
            "location": .05,
            "destination": .05,
            "source": .05,
            "reference": .05,
        }
        if relation_function == "location":
            scores["location"] += .85
            scores["reference"] += .35
        elif relation_function == "destination":
            scores["destination"] += .85
        elif relation_function == "reference":
            scores["reference"] += .82
            scores["location"] += .32
        if prep in self.SOURCE_PREPOSITIONS and grammatical_case == "gent":
            scores["source"] += .88
        if prep in {"в", "во", "на"} and grammatical_case == "accs":
            scores["destination"] += .66 + .22 * profile["motion_score"]
            scores["location"] += .12 * profile["state_score"]
        elif prep in {"в", "во", "на"} and grammatical_case in {"loct", "loc2"}:
            scores["location"] += .72 + .18 * profile["state_score"]
            scores["destination"] += .08 * profile["motion_score"]
        elif prep in {"к", "ко"} and grammatical_case in {"datv", None}:
            scores["destination"] += .76 + .15 * profile["motion_score"]
        elif prep in self.LOCATION_PREPOSITIONS:
            scores["location"] += .64 + .18 * profile["state_score"]
        return {
            role: min(1.0, score)
            for role, score in scores.items()
        }
