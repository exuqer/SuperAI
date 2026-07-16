from __future__ import annotations

from typing import Any, Iterable, Mapping

from server.spaces import CloudObject, WordSpace


class LexicalFactory:
    def register_candidates(
        self,
        candidates: Iterable[Mapping[str, Any]],
        space: WordSpace,
    ) -> list[CloudObject]:
        clouds: list[CloudObject] = []
        for index, candidate in enumerate(candidates):
            lemma = str(
                candidate.get("lemma")
                or candidate.get("normalized")
                or candidate.get("surface")
                or ""
            ).casefold()
            surface = str(candidate.get("surface") or candidate.get("candidate_text") or lemma)
            if not lemma:
                continue
            object_id = f"word:{lemma}:{surface.casefold()}"
            cloud = CloudObject(
                object_id=object_id,
                label=surface,
                dimensions={
                    "lemma": lemma,
                    "surface": surface,
                    "part_of_speech": candidate.get("part_of_speech")
                    or candidate.get("pos")
                    or "unknown",
                    "grammatical_features": candidate.get("grammatical_features")
                    or candidate.get("morphology")
                    or {},
                    "style": candidate.get("style") or "neutral",
                    "collocations": candidate.get("collocations") or [],
                    "concept": candidate.get("concept") or lemma,
                    "role": candidate.get("role") or candidate.get("resolved_role") or "",
                    "frequency": float(candidate.get("frequency") or 0.5),
                    "morphological_similarity": float(
                        candidate.get("morphological_similarity") or 1.0
                    ),
                },
                density=float(candidate.get("confidence") or candidate.get("score") or 0.72),
                halo=0.2,
                links={
                    "up:concept_space": [f"concept:{candidate.get('concept') or lemma}"],
                    "down:morpheme_space": [],
                },
                provenance={
                    "source": candidate.get("form_provenance", {}).get("source_type")
                    if isinstance(candidate.get("form_provenance"), Mapping)
                    else candidate.get("source") or "lexical_factory",
                    "candidate_index": index,
                },
                metadata=dict(candidate),
            )
            space.register(cloud)
            clouds.append(cloud)
        return clouds

    def find(
        self,
        space: WordSpace,
        concept: str,
        features: Mapping[str, Any] | None = None,
        *,
        min_relevance: float = 0.65,
    ) -> list[dict[str, Any]]:
        desired = {"concept": concept.casefold()}
        if features:
            desired["grammatical_features"] = dict(features)
        return [item.to_dict() for item in space.activate(desired, min_relevance=min_relevance)]
