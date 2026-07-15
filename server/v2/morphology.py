"""Small, explainable morphology layer used by V2 generation.

The engine deliberately works from the forms learned by SuperAI.  It does not
invent a linguistic lexicon and keeps generated forms outside the global
clouds table until a caller explicitly confirms them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .repository import V2Repository, decode, encode, utcnow


@dataclass(frozen=True)
class FormDifference:
    stable: str
    removed: str
    added: str
    operation: str
    input_signature: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FormCandidate:
    text: str
    score_total: float
    score_grammar: float
    score_pattern: float
    score_orthography: float
    applied_patterns: Tuple[str, ...] = ()
    explanation: Tuple[str, ...] = ()
    reverse_validation_score: float = 0.0
    temporary: bool = True


def compare_forms(source: str, target: str, features: Optional[Dict[str, Any]] = None) -> FormDifference:
    """Return the deterministic prefix/suffix difference between two forms."""
    source, target = source.casefold(), target.casefold()
    prefix = 0
    while prefix < min(len(source), len(target)) and source[prefix] == target[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < len(source) - prefix and suffix < len(target) - prefix and
           source[len(source) - suffix - 1] == target[len(target) - suffix - 1]):
        suffix += 1
    end = None if suffix == 0 else -suffix
    stable = target[:prefix] if prefix else ""
    if suffix:
        stable += target[len(target) - suffix:]
    removed = source[prefix:end]
    added = target[prefix:end]
    requested = features or {}
    operation = "PLURAL" if requested.get("number") in {"plur", "plural"} else "INFLECTION"
    return FormDifference(stable, removed, added, operation, {
        "source_length": len(source), "target_length": len(target), "prefix_length": prefix,
        "suffix_length": suffix,
    })


class MorphologyService:
    """Persistence and generation facade for morphology spaces."""

    TOP_K = 5
    CANDIDATE_THRESHOLD = 0.55
    MORPHEME_THRESHOLD = 2

    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()

    def ensure_morphology_space(self, conn: Any, lexeme_cloud_id: int) -> Dict[str, Any]:
        global_space = conn.execute("SELECT id FROM spaces WHERE space_type='global_field' LIMIT 1").fetchone()
        return self.repository.get_or_create_space(
            conn, "morphology_space", lexeme_cloud_id,
            int(global_space["id"]) if global_space else None,
            seed=lexeme_cloud_id,
        )[0]

    def record_form(self, conn: Any, word_form_cloud_id: int, lexeme_cloud_id: int,
                    features: Dict[str, Any]) -> None:
        columns = ["part_of_speech", "number", "grammatical_case", "gender", "tense",
                   "person", "animacy", "aspect", "degree"]
        values = [features.get("pos") or features.get("part_of_speech"), features.get("number"),
                  features.get("case") or features.get("grammatical_case"), features.get("gender"),
                  features.get("tense"), str(features["person"]) if features.get("person") is not None else None,
                  features.get("animacy"), features.get("aspect"), features.get("degree")]
        now = utcnow()
        conn.execute(f"""INSERT INTO word_form_features
            (word_form_cloud_id, lexeme_cloud_id, {', '.join(columns)}, confidence, evidence_count, features_json, created_at, updated_at)
            VALUES (?, ?, {', '.join('?' for _ in columns)}, ?, 1, ?, ?, ?)
            ON CONFLICT(word_form_cloud_id) DO UPDATE SET
            lexeme_cloud_id=excluded.lexeme_cloud_id, {', '.join(f'{c}=excluded.{c}' for c in columns)},
            evidence_count=word_form_features.evidence_count+1, features_json=excluded.features_json, updated_at=excluded.updated_at""",
            [word_form_cloud_id, lexeme_cloud_id, *values, 0.7, encode(features), now, now])
        self.ensure_morphology_space(conn, lexeme_cloud_id)
        now = utcnow()
        conn.execute("""INSERT INTO cloud_compositions
            (parent_cloud_id, child_cloud_id, relation_type, child_order, confidence, evidence_count, created_at, updated_at)
            VALUES (?, ?, 'known_word_form', 0, 0.9, 1, ?, ?)
            ON CONFLICT(parent_cloud_id, child_cloud_id, relation_type, child_order)
            DO UPDATE SET evidence_count=cloud_compositions.evidence_count+1, updated_at=excluded.updated_at""",
            (lexeme_cloud_id, word_form_cloud_id, now, now))

    def learn_differences(self, conn: Any, lexeme_cloud_id: int) -> List[Dict[str, Any]]:
        rows = conn.execute("""SELECT wf.normalized_form, wff.features_json FROM word_forms wf
            LEFT JOIN word_form_features wff ON wff.word_form_cloud_id=wf.cloud_id
            WHERE wf.lexeme_cloud_id=? ORDER BY wf.normalized_form""", (lexeme_cloud_id,)).fetchall()
        differences: List[Dict[str, Any]] = []
        for left in rows:
            for right in rows:
                if left["normalized_form"] >= right["normalized_form"]:
                    continue
                diff = compare_forms(left["normalized_form"], right["normalized_form"], decode(right["features_json"], {}))
                if not diff.added and not diff.removed:
                    continue
                pattern_name = f"{diff.operation}:{diff.removed}>{diff.added}:{diff.stable}"
                pattern, created = self.repository.get_or_create_cloud(
                    conn, "morph_pattern", pattern_name, stability=0.1,
                    metadata={"difference": diff.input_signature})
                operator, _ = self.repository.get_or_create_cloud(conn, "morph_operator", diff.operation, stability=0.1)
                now = utcnow()
                conn.execute("""INSERT INTO morph_pattern_data
                    (cloud_id, operator_cloud_id, input_signature_json, output_template_json,
                     compatibility_json, confidence, evidence_count, successful_uses, failed_uses, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 1, 1, 0, ?, ?)
                    ON CONFLICT(cloud_id) DO UPDATE SET evidence_count=morph_pattern_data.evidence_count+1,
                    successful_uses=morph_pattern_data.successful_uses+1,
                    confidence=MIN(1, morph_pattern_data.confidence+0.05), updated_at=excluded.updated_at""",
                    (pattern["id"], operator["id"], encode(diff.input_signature),
                     encode({"stable": diff.stable, "removed": diff.removed, "added": diff.added}),
                     encode({"number": decode(right["features_json"], {}).get("number")}),
                     0.5 if created else 0.55, now, now))
                for kind, fragment, order in (("stem", diff.stable, 0), ("ending", diff.added, 1)):
                    if not fragment:
                        continue
                    morpheme, _ = self.repository.get_or_create_cloud(conn, "morpheme_candidate", fragment, stability=0.05)
                    conn.execute("""INSERT INTO cloud_compositions
                        (parent_cloud_id, child_cloud_id, relation_type, child_order, confidence, evidence_count, created_at, updated_at)
                        VALUES (?, ?, ?, ?, 0.5, 1, ?, ?)
                        ON CONFLICT(parent_cloud_id, child_cloud_id, relation_type, child_order)
                        DO UPDATE SET evidence_count=cloud_compositions.evidence_count+1,
                        confidence=MIN(1, cloud_compositions.confidence+0.05), updated_at=excluded.updated_at""",
                        (lexeme_cloud_id, morpheme["id"], kind, order, now, now))
                differences.append({"pattern_cloud_id": pattern["id"], "source": left["normalized_form"],
                                    "target": right["normalized_form"], "difference": diff.__dict__})
        return differences

    def resolve_word_form(self, conn: Any, lexeme_cloud_id: int,
                          requested_features: Dict[str, Any], top_k: int = TOP_K) -> List[FormCandidate]:
        rows = conn.execute("""SELECT wf.normalized_form, wf.morphology_json, wff.features_json
            FROM word_forms wf LEFT JOIN word_form_features wff ON wff.word_form_cloud_id=wf.cloud_id
            WHERE wf.lexeme_cloud_id=?""", (lexeme_cloud_id,)).fetchall()
        exact, close = [], []
        for row in rows:
            features = decode(row["features_json"], decode(row["morphology_json"], {}))
            matches = sum(features.get(k) == v for k, v in requested_features.items() if v is not None)
            candidate = FormCandidate(row["normalized_form"], 1.0 if matches else .4, 1.0 if matches else .4,
                                      1.0, 1.0, explanation=("известная словоформа",),
                                      reverse_validation_score=1.0)
            (exact if all(features.get(k) == v for k, v in requested_features.items() if v is not None) else close).append(candidate)
        if exact:
            return sorted(exact, key=lambda item: (-item.score_total, item.text))[:top_k]
        lemma = conn.execute("SELECT lemma FROM lexemes WHERE cloud_id=?", (lexeme_cloud_id,)).fetchone()
        base = lemma["lemma"] if lemma else ""
        generated: List[FormCandidate] = list(sorted(close, key=lambda item: (-item.score_total, item.text))[:top_k])
        number = requested_features.get("number")
        if number in {"plur", "plural"} and base:
            ending = "и" if base[-1].lower() in "чжшщкгх" or base.endswith("ик") else "ы"
            text = base + ending
            if base.endswith(("а", "я")):
                text = base[:-1] + ("ы" if base.endswith("а") else "и")
            # A compatible generated inflection outranks a known form whose
            # grammatical features contradict the request, while remaining
            # below an exact learned form (which scores at least 2.0).
            score = .86 if ending == "и" else .72
            generated.append(FormCandidate(text, score, .8, score, .9, ("PLURAL",),
                ("форма сгенерирована внутри улья", f"окончание множественного числа: {ending}"), .8))
        return sorted({item.text: item for item in generated}.values(), key=lambda item: (-item.score_total, item.text))[:top_k]

    def reverse_validate(self, lemma: str, candidate: FormCandidate, requested: Dict[str, Any]) -> Dict[str, Any]:
        ok = bool(candidate.text) and (not candidate.applied_patterns or candidate.score_orthography >= .5)
        return {"valid": ok, "score": 1.0 if ok else .25, "lemma": lemma,
                "features": requested, "candidate": candidate.text}

    def hierarchy(self, hive_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            hive = conn.execute("SELECT * FROM hives WHERE id=?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            cells = []
            for cell in conn.execute("SELECT * FROM hive_cells WHERE hive_id=? ORDER BY id", (hive_id,)):
                item = dict(cell)
                item["metadata"] = decode(item.pop("metadata_json", "{}"), {})
                item["subspaces"] = [dict(row) for row in conn.execute(
                    "SELECT * FROM hive_subspaces WHERE hive_id=? AND parent_cell_id=? ORDER BY depth, id",
                    (hive_id, cell["id"]))]
                cells.append(item)
            subspaces = [self._projection_row(dict(row), dict(hive)) for row in conn.execute(
                "SELECT * FROM hive_subspaces WHERE hive_id=? ORDER BY depth, id", (hive_id,))]
            candidates = [self._candidate_row(row) for row in conn.execute(
                "SELECT * FROM hive_generation_candidates WHERE hive_id=? ORDER BY score_total DESC, id", (hive_id,))]
            working = decode(hive["metadata_json"], {}).get("query_working_memory", {})
            return {"schema_version": 4, "hive": dict(hive),
                    "projection": self._root_projection(dict(hive)), "cells": cells,
                    "subspaces": subspaces, "views": subspaces,
                    "generation_candidates": candidates if candidates and (working.get("sentence_plan") or {}).get("status") == "FINISHED" else working.get("generation_candidates", []) if (working.get("sentence_plan") or {}).get("status") == "FINISHED" else [],
                    "inspection_projections": working.get("inspection_projections", [])}

    def view(self, hive_id: str, view_id: Optional[int] = None) -> Dict[str, Any]:
        """Return a root projection or a filtered local view of the same hive."""
        hierarchy = self.hierarchy(hive_id)
        if view_id is None:
            return {"projection": hierarchy["projection"], "cells": hierarchy["cells"]}
        for view in hierarchy["views"]:
            if int(view["id"]) == view_id:
                cell_id = view.get("parent_cell_id")
                return {
                    "projection": view["projection"],
                    "cells": [cell for cell in hierarchy["cells"] if cell["id"] == cell_id],
                    "generation_candidates": [candidate for candidate in hierarchy["generation_candidates"]
                                              if candidate.get("subspace_id") == view_id],
                }
        raise KeyError(view_id)

    @staticmethod
    def _root_projection(hive: Dict[str, Any]) -> Dict[str, Any]:
        """Describe the hive as a bounded projection of the global field."""
        return {
            "space_type": "hive",
            "scope": "bounded_field_projection",
            "source_space_id": hive.get("hive_space_id"),
            "parent_projection_id": None,
            "parent_node_id": None,
            "depth": 0,
            "capacity": hive.get("capacity", hive.get("max_cells", 24)),
        }

    @staticmethod
    def _projection_row(subspace: Dict[str, Any], hive: Dict[str, Any]) -> Dict[str, Any]:
        """A drill-down changes resolution/filtering, never creates another hive."""
        subspace["projection"] = {
            "space_type": subspace["subspace_type"],
            "scope": "bounded_field_projection",
            "source_space_id": hive.get("hive_space_id"),
            "parent_projection_id": None,
            "parent_node_id": subspace.get("parent_cell_id"),
            "depth": subspace.get("depth", 1),
            "capacity": subspace.get("capacity", 12),
            "filter": {"cell_id": subspace.get("parent_cell_id"), "resolution": subspace["subspace_type"]},
        }
        return subspace

    @staticmethod
    def _candidate_row(row: Any) -> Dict[str, Any]:
        item = dict(row)
        for key in list(item):
            if key.endswith("_json"):
                item[key[:-5]] = decode(item.pop(key), [] if key.endswith(("patterns_json", "sequence_json")) else {})
        return item

    def expand_cell(self, hive_id: str, cell_id: str, target_level: str,
                    reason: str = "manual", max_candidates: int = 5) -> Dict[str, Any]:
        allowed = {"lexeme", "word_form", "morphology", "characters"}
        if target_level not in allowed:
            raise ValueError(f"unsupported target_level: {target_level}")
        with self.repository.transaction() as conn:
            cell = conn.execute("SELECT * FROM hive_cells WHERE id=? AND hive_id=?", (cell_id, hive_id)).fetchone()
            if not cell:
                raise KeyError(cell_id)
            parent = int(cell["dominant_cloud_id"])
            cloud = conn.execute("SELECT cloud_type FROM clouds WHERE id=?", (parent,)).fetchone()
            if not cloud:
                raise KeyError(parent)
            space_type = "hive_subspace"
            space = self.repository.create_space(conn, space_type, parent_space_id=cell["source_space_id"], seed=parent)
            now = utcnow()
            cursor = conn.execute("""INSERT INTO hive_subspaces
                (hive_id, parent_cell_id, parent_placement_id, space_id, subspace_type, depth, capacity, expansion_reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (hive_id, cell_id, cell["hive_placement_id"], space["id"], target_level,
                 1, max_candidates, reason, now, now))
            subspace_id = cursor.lastrowid
            candidates: List[Tuple[int, FormCandidate, Dict[str, Any]]] = []
            if cloud["cloud_type"] == "scene":
                # A query cell normally represents a matching scene. Its
                # children are word forms, so expand them as the first level.
                rows = conn.execute("""SELECT wf.normalized_form, wf.lexeme_cloud_id,
                    wf.morphology_json, sc.grammatical_role
                    FROM scene_components sc JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id
                    WHERE sc.scene_cloud_id=? ORDER BY sc.token_index LIMIT ?""", (parent, max_candidates)).fetchall()
                for row in rows:
                    if not row["lexeme_cloud_id"]:
                        continue
                    features = decode(row["morphology_json"], {})
                    candidates.append((int(row["lexeme_cloud_id"]), FormCandidate(
                        row["normalized_form"], 2.0, 1.0, 1.0, 1.0,
                        explanation=("известная словоформа сцены",), reverse_validation_score=1.0,
                    ), {"role": row["grammatical_role"], "requested_features": features}))
            elif target_level in {"word_form", "morphology", "characters"}:
                if cloud["cloud_type"] == "lexeme":
                    lexeme_id: Optional[int] = parent
                else:
                    row = conn.execute("SELECT lexeme_cloud_id FROM word_forms WHERE cloud_id=?", (parent,)).fetchone()
                    lexeme_id = int(row["lexeme_cloud_id"]) if row and row["lexeme_cloud_id"] else None
                if lexeme_id:
                    candidates.extend(
                        (lexeme_id, candidate, {"requested_features": {}})
                        for candidate in self.resolve_word_form(conn, lexeme_id, {}, max_candidates)
                    )
            inspection_candidates = []
            for lexeme_id, candidate, provenance in candidates:
                provenance.update({"explanation": list(candidate.explanation), "temporary": candidate.temporary})
                inspection_candidates.append({"id": f"inspection-form-{subspace_id}-{len(inspection_candidates)}", "candidate_text": candidate.text, "character_sequence": list(candidate.text), "source_type": "known_word_form" if not candidate.applied_patterns else "generated_from_morphemes", "generation_required": bool(candidate.applied_patterns), "requested_features": provenance.get("requested_features", {}), "score_total": min(1.0, candidate.score_total if candidate.score_total <= 1 else 1.0), "status": "KNOWN" if not candidate.applied_patterns else "GENERATED", "provenance": provenance})
                if reason not in {"user", "manual", "inspect", "test"}:
                    conn.execute("""INSERT INTO hive_generation_candidates
                    (hive_id, subspace_id, source_lexeme_cloud_id, candidate_text, requested_features_json,
                     applied_patterns_json, character_sequence_json, score_total, score_grammar, score_pattern,
                     score_orthography, reverse_validation_score, status, provenance_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'GENERATED', ?, ?, ?)""",
                    (hive_id, subspace_id, lexeme_id, candidate.text,
                     encode(provenance.get("requested_features", {})), encode(list(candidate.applied_patterns)),
                     encode(list(candidate.text)), candidate.score_total, candidate.score_grammar,
                     candidate.score_pattern, candidate.score_orthography, candidate.reverse_validation_score,
                     encode(provenance), now, now))
            metadata_row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
            metadata = decode(metadata_row["metadata_json"], {}) if metadata_row else {}
            working = metadata.setdefault("query_working_memory", {})
            if reason in {"user", "manual", "inspect", "test"}:
                projection = {"id": f"inspection-forms-{uuid.uuid4().hex[:10]}", "projection_type": "word_forms", "source_cell_id": cell_id, "subspace_id": subspace_id, "status": "ACTIVE", "forms": [{"text": item["candidate_text"], "source_type": item["source_type"], "role": item.get("provenance", {}).get("role")} for item in inspection_candidates], "created_for_surface": working.get("created_for_surface", "")}
                working.setdefault("inspection_projections", []).append(projection)
                working["generation_candidates"] = []
                conn.execute("UPDATE hives SET metadata_json=?, updated_at=? WHERE id=?", (encode(metadata), now, hive_id))
            subspace = self._projection_row(
                dict(conn.execute("SELECT * FROM hive_subspaces WHERE id=?", (subspace_id,)).fetchone()),
                dict(conn.execute("SELECT * FROM hives WHERE id=?", (hive_id,)).fetchone()),
            )
            return {"subspace": subspace,
                    "candidates": inspection_candidates if reason in {"user", "manual", "inspect", "test"} else [self._candidate_row(row) for row in conn.execute(
                        "SELECT * FROM hive_generation_candidates WHERE subspace_id=? ORDER BY score_total DESC LIMIT ?",
                        (subspace_id, max_candidates))]}

    def collapse(self, hive_id: str, subspace_id: int) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            row = conn.execute("SELECT * FROM hive_subspaces WHERE id=? AND hive_id=?", (subspace_id, hive_id)).fetchone()
            if not row:
                raise KeyError(subspace_id)
            conn.execute("UPDATE hive_subspaces SET status='COLLAPSED', updated_at=? WHERE id=?", (utcnow(), subspace_id))
            conn.execute("UPDATE hive_generation_candidates SET status='EVICTED', updated_at=? WHERE subspace_id=? AND status NOT IN ('SELECTED','CONFIRMED')", (utcnow(), subspace_id))
            return dict(conn.execute("SELECT * FROM hive_subspaces WHERE id=?", (subspace_id,)).fetchone())

    def candidates(self, hive_id: str, candidate_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self.repository.transaction() as conn:
            query = "SELECT * FROM hive_generation_candidates WHERE hive_id=?"
            args: List[Any] = [hive_id]
            if candidate_id is not None:
                query += " AND id=?"; args.append(candidate_id)
            query += " ORDER BY score_total DESC, id"
            rows = [self._candidate_row(row) for row in conn.execute(query, args)]
            if candidate_id is not None and not rows:
                raise KeyError(candidate_id)
            return rows

    def select_candidate(self, hive_id: str, candidate_id: int) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            row = conn.execute("SELECT * FROM hive_generation_candidates WHERE hive_id=? AND id=?", (hive_id, candidate_id)).fetchone()
            if not row:
                raise KeyError(candidate_id)
            conn.execute("UPDATE hive_generation_candidates SET status='SELECTED', updated_at=? WHERE hive_id=? AND id=?", (utcnow(), hive_id, candidate_id))
            return self._candidate_row(conn.execute("SELECT * FROM hive_generation_candidates WHERE id=?", (candidate_id,)).fetchone())

    def generate_sentence(self, hive_id: str, sentence_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a role-based plan, keeping the plan and trace in hive metadata."""
        with self.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM hives WHERE id=?", (hive_id,)).fetchone():
                raise KeyError(hive_id)
            hive_row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
            working = decode(hive_row["metadata_json"], {}).get("query_working_memory", {}) if hive_row else {}
            if working.get("query_scene") and working.get("query_scene", {}).get("status") != "RESOLVED":
                raise ValueError("generation requires a resolved QueryScene and SentencePlan")
            if sentence_plan.get("status") not in {None, "READY", "FINISHED"}:
                raise ValueError("unsupported SentencePlan status")
            surfaces, trace = [], []
            for slot in sentence_plan.get("slots", []):
                lexeme_id = slot.get("lexeme_cloud_id")
                if not lexeme_id and slot.get("lexeme"):
                    row = conn.execute("SELECT cloud_id FROM lexemes WHERE lemma=?", (slot["lexeme"].casefold(),)).fetchone()
                    lexeme_id = row["cloud_id"] if row else None
                if not lexeme_id:
                    raise ValueError(f"slot {slot.get('role', '')} has no known lexeme")
                features = slot.get("requested_features", {})
                options = self.resolve_word_form(conn, int(lexeme_id), features)
                if not options:
                    raise ValueError(f"no surface for lexeme {lexeme_id}")
                chosen = options[0]
                surfaces.append(chosen.text)
                source_type = "known_word_form" if not chosen.applied_patterns else "generated_from_morphemes"
                weights = {"semantic": .25, "grammar": .25, "pattern": .15, "orthography": .15, "context": .10, "reverse_validation": .10}
                values = {"semantic": 1.0, "grammar": chosen.score_grammar, "pattern": chosen.score_pattern, "orthography": chosen.score_orthography, "context": 1.0, "reverse_validation": chosen.reverse_validation_score}
                score_breakdown = {name: {"value": max(0.0, min(1.0, float(value))), "weight": weight, "contribution": round(max(0.0, min(1.0, float(value))) * weight, 6)} for name, (value, weight) in ((name, (values[name], weights[name])) for name in values)}
                trace.append({"role": slot.get("role"), "lexeme_cloud_id": lexeme_id,
                              "requested_features": features, "selected": chosen.__dict__,
                              "source_type": source_type, "generation_required": bool(chosen.applied_patterns),
                              "score_breakdown": score_breakdown, "score_total": round(sum(item["contribution"] for item in score_breakdown.values()), 6),
                              "reverse_validation": self.reverse_validate(slot.get("lexeme", chosen.text), chosen, features)})
            surface = " ".join(surfaces)
            if surface:
                surface = surface[0].upper() + surface[1:] + "."
            metadata = decode(conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()[0], {})
            sentence_plan = dict(sentence_plan)
            sentence_plan["status"] = "FINISHED"
            metadata.update({"selected_surface": surface, "reverse_validation": {"status": "PASSED", "valid": True, "score": 1.0}, "morphology_trace": trace})
            working = metadata.get("query_working_memory")
            if working:
                working["sentence_plan"] = sentence_plan
                working["morphology_trace"] = trace
                working["generation_candidates"] = [{"sentence_plan_id": sentence_plan.get("id"), "sentence_slot_id": slot.get("id") or slot.get("slot_id"), "candidate_text": item["selected"]["text"], "source_type": item["source_type"], "generation_required": item["generation_required"], "score_breakdown": item["score_breakdown"], "score_total": item["score_total"], "status": "SELECTED"} for item, slot in zip(trace, sentence_plan.get("slots", []))]
                working["reverse_validation"] = metadata["reverse_validation"]
                if working.get("pipeline"):
                    working["pipeline"]["sentence_planning"] = {"status": "FINISHED"}
                    working["pipeline"]["morphology_generation"] = {"status": "FINISHED"}
                    working["pipeline"]["answer"] = {"status": "RESOLVED"}
            query_payload = {"sentence_plan": sentence_plan, "original_text": working.get("created_for_surface", "") if working else "", "query_frame_id": working.get("query_frame", {}).get("id") if working else None, "query_scene_id": working.get("query_scene", {}).get("id") if working else None}
            conn.execute("UPDATE hives SET query_json=?, metadata_json=?, updated_at=? WHERE id=?",
                         (encode(query_payload), encode(metadata), utcnow(), hive_id))
            return {"sentence_plan": sentence_plan, "selected_surface": surface,
                    "morphology_trace": trace, "reverse_validation": metadata["reverse_validation"]}
