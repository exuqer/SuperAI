"""Lexical and structural resonance probes, isolated from query scenes."""

from __future__ import annotations

import math
import re
import uuid
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from .repository import V2Repository, decode, encode, utcnow
from .unknown_search import signature


QUESTION_WORDS = {"кто", "кого", "кому", "что", "где", "куда", "откуда", "когда", "как", "почему", "зачем", "чем", "сколько"}
QUESTION_OPERATORS = QUESTION_WORDS | {"ли"}
PROBE_INTENTS = {"LEXICAL_PROBE", "MORPHOLOGICAL_PROBE", "STRUCTURAL_PROBE"}
SCOPES = {"LOCAL_ONLY", "LOCAL_THEN_GLOBAL", "GLOBAL_ONLY"}


class InputIntentClassifier:
    """Classifies input before any QueryFrame or QueryScene is constructed."""

    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()
        from .training import RussianMorphology
        self.morphology = RussianMorphology()

    def classify(self, text: str) -> str:
        normalized = text.strip().casefold()
        tokens = re.findall(r"[\wё-]+", normalized, flags=re.IGNORECASE)
        if not tokens:
            return "STRUCTURAL_PROBE"
        from .intent import IntentClassifier
        dialogue_intent = IntentClassifier().classify(text)["intent"]
        if dialogue_intent in {"GREETING", "SMALL_TALK", "GREETING_WITH_SMALL_TALK"}:
            return dialogue_intent
        if "+" in normalized:
            return "MORPHOLOGICAL_PROBE"
        if normalized.startswith(("найди похож", "похожее на", "структур")):
            return "STRUCTURAL_PROBE"
        has_question_operator = any(token in QUESTION_OPERATORS for token in tokens)
        question = has_question_operator or "?" in normalized
        has_predicate = any(self.morphology.parse(token).pos_tag in {"VERB", "INFN"} for token in tokens)
        if has_question_operator:
            return "SCENE_QUESTION"
        if question and (has_predicate or len(tokens) >= 3):
            return "SCENE_QUESTION"
        if len(tokens) > 1 and has_predicate:
            return "SCENE_STATEMENT"
        if len(tokens) != 1:
            return "SCENE_STATEMENT"
        token = tokens[0]
        with self.repository.transaction() as conn:
            known = conn.execute(
                """SELECT 1 FROM word_forms WHERE normalized_form=?
                   UNION SELECT 1 FROM lexemes WHERE lemma=? LIMIT 1""",
                (token, token),
            ).fetchone()
        return "SCENE_STATEMENT" if known else "STRUCTURAL_PROBE"


class LexicalCandidateResolver:
    """Finds lexical candidates; it does not perform hive resonance."""

    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()
        self.classifier = InputIntentClassifier(self.repository)

    def classify(self, text: str) -> Dict[str, str]:
        return {"intent": self.classifier.classify(text)}

    def resolve(self, hive_id: str, text: str, *, use_global: bool = True,
                max_candidates: int = 12) -> List[Dict[str, Any]]:
        """Resolve lexical candidates for every meaningful token in *text*.

        The result deliberately contains lexical scores only.  Dynamic activation,
        competition, and stabilization belong to ``HiveResonanceEngine``.
        """
        tokens = list(dict.fromkeys(re.findall(r"[\wё-]+", text.casefold(), flags=re.IGNORECASE)))
        candidates: Dict[int, Dict[str, Any]] = {}
        with self.repository.transaction() as conn:
            hive = conn.execute("SELECT id FROM hives WHERE id=?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            for token in tokens:
                lemma = self.classifier.morphology.parse(token).lemma.casefold()
                probe = {"surface": token, "signature": self._signature(token)}
                lemma_probe = {"surface": lemma, "signature": self._signature(lemma)}
                local: List[Dict[str, Any]] = []
                for level, level_probe in (("EXACT", probe), ("LEMMA", lemma_probe), ("STRUCTURE", probe)):
                    found = self._search(conn, hive_id, level_probe, "LOCAL", level)
                    if found:
                        local = found
                        break
                found = local
                scope = "LOCAL"
                if not found and use_global:
                    for level, level_probe in (("EXACT", probe), ("LEMMA", lemma_probe), ("STRUCTURE", probe)):
                        found = self._search(conn, hive_id, level_probe, "GLOBAL", level)
                        if found:
                            scope = "GLOBAL"
                            break
                for item in found:
                    matched_by = {
                        "EXACT_FORM_MATCH": "exact",
                        "EXACT_LEXEME_MATCH": "lemma",
                        "STEM_MATCH": "root",
                    }.get(item["match_type"], "structure")
                    score = float(item["score"])
                    result = {
                        **item,
                        "conceptId": str(item["candidate_cloud_id"]),
                        "matchedBy": matched_by,
                        "lexicalScore": score,
                        "lexical_score": score,
                        "source": scope.casefold(),
                        "temporary": scope == "GLOBAL",
                        "input_token": token,
                    }
                    cloud_id = int(item["candidate_cloud_id"])
                    existing = candidates.get(cloud_id)
                    if existing is None or score > float(existing["lexicalScore"]):
                        candidates[cloud_id] = result
        return sorted(candidates.values(), key=lambda item: (-float(item["lexicalScore"]), item["value"]))[:max_candidates]

    def create(self, hive_id: str, text: str, scope: str = "LOCAL_THEN_GLOBAL") -> Dict[str, Any]:
        scope = scope if scope in SCOPES else "LOCAL_THEN_GLOBAL"
        surface = text.strip().casefold()
        if not surface:
            raise ValueError("text must not be empty")
        intent = self.classifier.classify(surface)
        if intent not in PROBE_INTENTS:
            raise ValueError("resonance probes require a lexical or structural input")
        with self.repository.transaction() as conn:
            hive = conn.execute("SELECT id FROM hives WHERE id=?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            message_id = self._record_message(conn, hive_id, text, intent)
            probe = {
                "id": f"probe-{uuid.uuid4().hex[:12]}", "message_id": message_id, "hive_id": hive_id,
                "input": text, "surface": surface, "probe_type": intent, "scope": scope,
                "status": "CREATED", "signature": self._signature(surface),
                "local_search": {"status": "CREATED", "matches": []},
                "global_search": {"status": "CREATED", "matches": []},
                "local_results": [], "global_results": [], "selected_match_id": None,
                "selected_result": None, "imported_cells": [], "imported_results": [],
                "created_at": utcnow(), "completed_at": None,
            }
            state = self._working_state(conn, hive_id)
            state.pop("query_frame", None)
            state.pop("query_scene", None)
            state.pop("answer", None)
            state.pop("role_searches", None)
            state.pop("sentence_plan", None)
            state["resonance_probe_mode"] = True
            state["resonance_probes"] = [item for item in state.get("resonance_probes", []) if item.get("id") != probe["id"]] + [probe]
            state["active_resonance_probe_id"] = probe["id"]
            state["local_resonance"] = {"latest_probe_id": probe["id"], "probe_text": text, "status": "RESONANCE_CREATED", "matched_form": None, "matched_lexeme": None}
            self._save_state(conn, hive_id, state)
            return deepcopy(probe)

    def step(self, hive_id: str, probe_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            state, probe = self._find_probe(conn, hive_id, probe_id)
            self._advance(conn, hive_id, probe, one_step=True)
            self._persist_probe(conn, hive_id, state, probe)
            return deepcopy(probe)

    def run(self, hive_id: str, probe_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            state, probe = self._find_probe(conn, hive_id, probe_id)
            self._advance(conn, hive_id, probe, one_step=False)
            self._persist_probe(conn, hive_id, state, probe)
            return {"probe": deepcopy(probe)}

    def get(self, hive_id: str, probe_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            _, probe = self._find_probe(conn, hive_id, probe_id)
            return deepcopy(probe)

    def related_scenes(self, hive_id: str, probe_id: str, match_id: str = "") -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            _, probe = self._find_probe(conn, hive_id, probe_id)
            matches = self._all_matches(probe)
            target = next((item for item in matches if item["id"] == match_id), None) if match_id else None
            if target is None:
                target = probe.get("selected_result") or (matches[0] if matches else None)
            return {"candidate": target, "related_scenes": target.get("related_scenes", []) if target else []}

    def import_match(self, hive_id: str, probe_id: str, match_id: str, include_scenes: bool = False) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            state, probe = self._find_probe(conn, hive_id, probe_id)
            match = next((item for item in self._all_matches(probe) if item["id"] == match_id), None)
            if not match:
                raise KeyError(match_id)
            hive = conn.execute("SELECT space_id, max_cells FROM hives WHERE id=?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            cell = self._place_seed(conn, hive_id, int(hive["space_id"]), match, probe)
            imported = [cell]
            if include_scenes:
                for scene in match.get("related_scenes", [])[:1]:
                    imported.append(self._place_scene_source(conn, hive_id, int(hive["space_id"]), scene, probe))
            probe["selected_match_id"] = match_id
            probe["selected_result"] = match
            probe["imported_cells"].extend(item for item in imported if item)
            probe["imported_results"].append(match_id)
            probe["status"] = "IMPORTED_TO_HIVE"
            self._persist_probe(conn, hive_id, state, probe)
            return {"probe": deepcopy(probe), "imported_cells": [item for item in imported if item]}

    def state(self, hive_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            state = self._working_state(conn, hive_id)
            active_id = state.get("active_resonance_probe_id")
            probe = next((item for item in state.get("resonance_probes", []) if item.get("id") == active_id), None)
            if not probe:
                raise KeyError(hive_id)
            cells = self._cells(conn, hive_id)
            reasoning = [item for item in cells if item.get("component_class") not in {"memory_source"}]
            local = probe.get("local_search", {})
            global_ = probe.get("global_search", {})
            display = self._display_status(probe)
            result = {
                "id": hive_id, "resonance_probe": deepcopy(probe), "resonance_probes": deepcopy(state.get("resonance_probes", [])),
                "local_resonance": {"latest_probe_id": probe["id"], "latest_surface": probe["input"], "probe_text": probe["input"], "status": display,
                                    "matched_form": (local.get("matches") or global_.get("matches") or [{}])[0].get("value"), "matched_lexeme": None},
                "display_status": display, "header_input": probe["input"], "query_frame": None, "query_scene": None,
                "answer": None, "role_searches": [], "candidates": [], "memory_scenes": [], "cells": cells,
                "vibration": {"status": "BLOCKED" if not reasoning else "READY", "enabled": bool(reasoning), "reason": None if reasoning else "reasoning_cells >= 1 required"},
                "hive": {"id": hive_id, "display_status": display, "query": {"original_text": probe["input"], "query_frame_id": None, "query_scene_id": None},
                         "capacity": {"working_cells": len(reasoning), "memory_sources": len(cells) - len(reasoning), "total_placements": len(cells)}, "local_resonance": state.get("local_resonance")},
                "hive_structure": {"placements": {"working_cells": len(reasoning), "memory_sources": len(cells) - len(reasoning), "total": len(cells)}},
                "stats": {"working_cells": len(reasoning), "memory_sources": len(cells) - len(reasoning), "total_placements": len(cells)},
            }
            from .validation import StateConsistencyValidator
            result["state_consistency_errors"] = StateConsistencyValidator().validate(result)
            return result

    def _advance(self, conn: Any, hive_id: str, probe: Dict[str, Any], one_step: bool) -> None:
        phases: List[tuple[str, str, str]] = []
        if probe["scope"] != "GLOBAL_ONLY":
            phases += [("LOCAL", "EXACT", "LOCAL_EXACT_SEARCH"), ("LOCAL", "LEMMA", "LOCAL_LEMMA_SEARCH"), ("LOCAL", "STRUCTURE", "LOCAL_STRUCTURE_SEARCH")]
        if probe["scope"] != "LOCAL_ONLY":
            phases += [("GLOBAL", "EXACT", "GLOBAL_EXACT_SEARCH"), ("GLOBAL", "LEMMA", "GLOBAL_LEMMA_SEARCH"), ("GLOBAL", "STRUCTURE", "GLOBAL_STRUCTURE_SEARCH")]
        current = probe.get("status", "CREATED")
        start = next((index for index, (_, _, status) in enumerate(phases) if status == current), -1) + 1
        for scope, level, status in phases[start:]:
            probe["status"] = status
            found = self._search(conn, hive_id, probe, scope, level)
            target = probe["local_search"] if scope == "LOCAL" else probe["global_search"]
            target["matches"] = found or target.get("matches", [])
            if found:
                target["status"] = f"{scope}_{level}_HIT" if level == "EXACT" else "MATCHES_FOUND"
                probe["local_results" if scope == "LOCAL" else "global_results"] = found
                if level == "EXACT":
                    probe["status"] = f"{scope}_EXACT_HIT"
                elif scope == "LOCAL":
                    probe["status"] = "LOCAL_MATCHES_FOUND"
                else:
                    probe["status"] = "GLOBAL_MATCHES_FOUND"
                probe["completed_at"] = utcnow()
                return
            target["status"] = "COMPLETED_NO_MATCH"
            if scope == "LOCAL" and level == "STRUCTURE":
                probe["status"] = "LOCAL_NO_MATCH"
            if one_step:
                return
        if not self._all_matches(probe):
            probe["status"] = "COMPLETED_NO_MATCH"
        probe["completed_at"] = utcnow()

    def _search(self, conn: Any, hive_id: str, probe: Dict[str, Any], scope: str, level: str) -> List[Dict[str, Any]]:
        surface = probe["surface"]
        ids = self._scope_cloud_ids(conn, hive_id, scope)
        if scope == "LOCAL" and not ids:
            return []
        params: List[Any] = []
        where = ""
        if ids:
            marks = ",".join("?" for _ in ids)
            where = f" AND c.id IN ({marks})"
            params.extend(ids)
        if level == "EXACT":
            sql = """SELECT c.id, c.cloud_type, c.canonical_name, c.mass, c.observation_count, wf.lexeme_cloud_id, l.lemma
                     FROM clouds c LEFT JOIN word_forms wf ON wf.cloud_id=c.id LEFT JOIN lexemes l ON l.cloud_id=wf.lexeme_cloud_id
                     WHERE c.cloud_type IN ('word_form','lexeme') AND (c.canonical_name=? OR wf.normalized_form=? OR l.lemma=?)""" + where
            rows = conn.execute(sql, [surface, surface, surface, *params]).fetchall()
            return self._matches(conn, probe, rows, scope, "EXACT_FORM_MATCH", 1.0)
        if level == "LEMMA":
            sql = """SELECT c.id, c.cloud_type, c.canonical_name, c.mass, c.observation_count, wf.lexeme_cloud_id, l.lemma
                     FROM clouds c LEFT JOIN word_forms wf ON wf.cloud_id=c.id LEFT JOIN lexemes l ON l.cloud_id=wf.lexeme_cloud_id
                     WHERE c.cloud_type IN ('word_form','lexeme') AND (l.lemma=? OR c.canonical_name=?)""" + where
            rows = conn.execute(sql, [surface, surface, *params]).fetchall()
            return self._matches(conn, probe, rows, scope, "EXACT_LEXEME_MATCH", .95)
        fragments = [item for item in probe["signature"]["possible_stems"] if len(item) >= 2] + probe["signature"]["trigrams"] + probe["signature"]["prefixes"]
        fragments = list(dict.fromkeys(item for item in fragments if len(item) >= 2))
        if not fragments:
            return []
        marks = ",".join("?" for _ in fragments)
        indexed = [int(row["cloud_id"]) for row in conn.execute(
            f"SELECT DISTINCT cloud_id FROM structural_index WHERE fragment IN ({marks}) LIMIT 100", fragments
        ).fetchall()]
        if not indexed:
            return []
        indexed_marks = ",".join("?" for _ in indexed)
        sql = f"""SELECT c.id, c.cloud_type, c.canonical_name, c.mass, c.observation_count, wf.lexeme_cloud_id, l.lemma
                    FROM clouds c LEFT JOIN word_forms wf ON wf.cloud_id=c.id LEFT JOIN lexemes l ON l.cloud_id=wf.lexeme_cloud_id
                    WHERE c.id IN ({indexed_marks})""" + where
        rows = conn.execute(sql, [*indexed, *params]).fetchall()
        return self._matches(conn, probe, rows, scope, "STEM_MATCH", None)

    def _matches(self, conn: Any, probe: Dict[str, Any], rows: Iterable[Any], scope: str, default_type: str, fixed_score: Optional[float]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen: set[int] = set()
        surface = probe["surface"]
        for row in rows:
            value = str(row["canonical_name"]).casefold()
            if int(row["id"]) in seen or (default_type == "STEM_MATCH" and not value.startswith(surface)):
                continue
            seen.add(int(row["id"]))
            candidate_type = str(row["cloud_type"])
            shared = self._shared_prefix(surface, value)
            match_type = default_type
            score = fixed_score if fixed_score is not None else self._stem_score(surface, value, candidate_type)
            result.append({
                "id": f"match-{scope.casefold()}-{row['id']}", "candidate_cloud_id": int(row["id"]), "cloud_id": int(row["id"]),
                "candidate_type": candidate_type, "type": candidate_type, "canonical_name": row["canonical_name"], "value": row["canonical_name"],
                "lemma": row["lemma"] or (row["canonical_name"] if candidate_type == "lexeme" else None), "lemma_cloud_id": int(row["lexeme_cloud_id"]) if row["lexeme_cloud_id"] else None,
                "match_type": match_type, "shared_structure": shared, "score": round(score, 4), "scope": scope,
                "scores": {"exact_form_score": 1.0 if match_type == "EXACT_FORM_MATCH" else 0.0, "exact_lexeme_score": 1.0 if match_type == "EXACT_LEXEME_MATCH" else 0.0,
                           "stem_score": score if match_type == "STEM_MATCH" else 0.0, "morpheme_score": 0.0, "prefix_score": score if match_type == "STEM_MATCH" else 0.0,
                           "ngram_score": 0.0, "morphological_score": 0.0, "semantic_score": 0.0, "scene_support_score": 0.0, "total_resonance": round(score, 4)},
                "global_mass": float(row["mass"]), "observation_count": int(row["observation_count"]), "related_scenes": self._related_scenes(conn, int(row["id"]), row["lexeme_cloud_id"]),
                "probe_activation": score,
            })
        result.sort(key=lambda item: (self._family_rank(surface, item), -item["score"], item["value"]))
        return result[:20]

    @staticmethod
    def _stem_score(surface: str, value: str, candidate_type: str) -> float:
        extension = max(0, len(value) - len(surface))
        base = 1.0 if candidate_type == "lexeme" else .88
        if extension > 1:
            base -= .18 * (extension - 1)
        return max(.4, base)

    @staticmethod
    def _family_rank(surface: str, item: Dict[str, Any]) -> tuple[int, int, int]:
        value = str(item["value"]).casefold()
        extension = max(0, len(value) - len(surface))
        return (0 if extension <= 1 else 1, extension, 0 if item["type"] == "lexeme" else 1)

    def _related_scenes(self, conn: Any, cloud_id: int, lemma_cloud_id: Any) -> List[Dict[str, Any]]:
        key = int(lemma_cloud_id) if lemma_cloud_id else cloud_id
        rows = conn.execute("""SELECT DISTINCT s.cloud_id, s.sentence_text, c.mass FROM scenes s
            JOIN scene_components sc ON sc.scene_cloud_id=s.cloud_id JOIN clouds c ON c.id=s.cloud_id
            WHERE (sc.word_form_cloud_id=? OR sc.lexeme_cloud_id=?)
              AND s.knowledge_status<>'RETRACTED'
            ORDER BY c.mass DESC, s.cloud_id LIMIT 8""", (cloud_id, key)).fetchall()
        return [{"scene_cloud_id": int(row["cloud_id"]), "text": row["sentence_text"], "support_score": round(min(1.0, .55 + float(row["mass"]) / 10), 4)} for row in rows]

    @staticmethod
    def _scope_cloud_ids(conn: Any, hive_id: str, scope: str) -> List[int]:
        if scope == "GLOBAL":
            return []
        return [int(row["cloud_id"]) for row in conn.execute("""SELECT DISTINCT cloud_id FROM hive_cell_components WHERE cell_id IN
            (SELECT id FROM hive_cells WHERE hive_id=?) UNION SELECT DISTINCT dominant_cloud_id AS cloud_id FROM hive_cells WHERE hive_id=?""", (hive_id, hive_id)).fetchall()]

    @staticmethod
    def _signature(surface: str) -> Dict[str, Any]:
        value = signature(surface, possible=True)
        return {"surface": surface, "length": len(surface), "characters": list(surface), "prefixes": value["prefixes"], "suffixes": value["suffixes"],
                "bigrams": value["bigrams"], "trigrams": value["trigrams"], "possible_stems": value["possible_stems"],
                "possible_morphemes": value["possible_stems"], "possible_lemma": None}

    @staticmethod
    def _shared_prefix(left: str, right: str) -> str:
        index = 0
        while index < min(len(left), len(right)) and left[index] == right[index]:
            index += 1
        return left[:index]

    def _find_probe(self, conn: Any, hive_id: str, probe_id: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        state = self._working_state(conn, hive_id)
        if not state:
            raise KeyError(hive_id)
        probe = next((item for item in state.get("resonance_probes", []) if item.get("id") == probe_id), None)
        if not probe:
            raise KeyError(probe_id)
        return state, probe

    def _persist_probe(self, conn: Any, hive_id: str, state: Dict[str, Any], probe: Dict[str, Any]) -> None:
        state["resonance_probes"] = [probe if item.get("id") == probe["id"] else item for item in state.get("resonance_probes", [])]
        state["active_resonance_probe_id"] = probe["id"]
        state["local_resonance"] = {"latest_probe_id": probe["id"], "latest_surface": probe["input"], "probe_text": probe["input"], "status": self._display_status(probe), "matched_form": None, "matched_lexeme": None}
        self._save_state(conn, hive_id, state)

    @staticmethod
    def _all_matches(probe: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(probe.get("local_results", [])) + list(probe.get("global_results", []))

    @staticmethod
    def _display_status(probe: Dict[str, Any]) -> str:
        return {"CREATED": "RESONANCE_CREATED", "LOCAL_EXACT_SEARCH": "LOCAL_SEARCH", "LOCAL_LEMMA_SEARCH": "LOCAL_SEARCH", "LOCAL_STRUCTURE_SEARCH": "LOCAL_SEARCH", "LOCAL_EXACT_HIT": "LOCAL_EXACT_HIT", "GLOBAL_EXACT_HIT": "GLOBAL_EXACT_HIT", "LOCAL_NO_MATCH": "LOCAL_NO_MATCH", "LOCAL_MATCHES_FOUND": "LOCAL_MATCHES_FOUND", "GLOBAL_EXACT_SEARCH": "GLOBAL_SEARCH", "GLOBAL_LEMMA_SEARCH": "GLOBAL_SEARCH", "GLOBAL_STRUCTURE_SEARCH": "GLOBAL_SEARCH", "GLOBAL_MATCHES_FOUND": "GLOBAL_MATCHES_FOUND", "IMPORTED_TO_HIVE": "IMPORTED_TO_HIVE"}.get(probe.get("status"), probe.get("status", "RESONANCE_CREATED"))

    def _record_message(self, conn: Any, hive_id: str, text: str, intent: str) -> str:
        turn = int(conn.execute("SELECT COALESCE(MAX(turn_index), 0) FROM hive_messages WHERE hive_id=?", (hive_id,)).fetchone()[0]) + 1
        message_id = f"message-{uuid.uuid4().hex[:12]}"
        conn.execute("INSERT INTO hive_messages(id,hive_id,turn_index,text,parsed_json,created_at) VALUES(?,?,?,?,?,?)", (message_id, hive_id, turn, text, encode({"intent": intent}), utcnow()))
        return message_id

    @staticmethod
    def _working_state(conn: Any, hive_id: str) -> Dict[str, Any]:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        return decode(row["metadata_json"], {}).get("query_working_memory", {}) if row else {}

    @staticmethod
    def _save_state(conn: Any, hive_id: str, state: Dict[str, Any]) -> None:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        metadata = decode(row["metadata_json"], {})
        metadata["query_working_memory"] = state
        conn.execute("UPDATE hives SET query_text=?, query_json=?, metadata_json=?, updated_at=? WHERE id=?", (state.get("local_resonance", {}).get("probe_text", ""), encode({"resonance_probe_id": state.get("active_resonance_probe_id")}), encode(metadata), utcnow(), hive_id))

    def _place_seed(self, conn: Any, hive_id: str, space_id: int, match: Dict[str, Any], probe: Dict[str, Any]) -> Dict[str, Any]:
        existing = conn.execute("SELECT id FROM hive_cells WHERE hive_id=? AND dominant_cloud_id=? AND component_class='lexical_seed'", (hive_id, match["candidate_cloud_id"])).fetchone()
        if existing:
            return {"id": existing["id"], "component_class": "lexical_seed", "existing": True}
        count = int(conn.execute("SELECT COUNT(*) FROM hive_cells WHERE hive_id=?", (hive_id,)).fetchone()[0])
        angle = count * 2.399963
        strength = float(match["score"])
        placement = self.repository.create_placement(conn, match["candidate_cloud_id"], space_id, 420 + math.cos(angle) * (80 + 42 * math.sqrt(count + 1)), 280 + math.sin(angle) * (80 + 42 * math.sqrt(count + 1)), local_activation=strength, local_gravity=strength, metadata={"placement_kind": "lexical_seed", "source_probe_id": probe["id"]})
        source = conn.execute("""SELECT p.id,p.space_id FROM cloud_placements p JOIN spaces s ON s.id=p.space_id WHERE p.cloud_id=? AND s.space_type='global_field' LIMIT 1""", (match["candidate_cloud_id"],)).fetchone()
        cell_id, now = f"cell-{uuid.uuid4().hex}", utcnow()
        metadata = {"component_class": "lexical_seed", "source_probe_id": probe["id"], "source_signal": probe["input"], "global_cloud_id": match["candidate_cloud_id"], "lexeme": match.get("lemma") or match["value"], "match_type": match["match_type"], "resonance_score": strength}
        conn.execute("""INSERT INTO hive_cells(id,hive_id,dominant_cloud_id,hive_placement_id,source_cloud_id,source_placement_id,source_space_id,stored_strength,retention,local_activation,component_class,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (cell_id,hive_id,match["candidate_cloud_id"],placement["id"],match["candidate_cloud_id"],source["id"] if source else None,source["space_id"] if source else None,strength,strength,strength,"lexical_seed",encode(metadata),now,now))
        conn.execute("""INSERT INTO hive_cell_components(cell_id,cloud_id,composition_share,local_activation,role,effective_strength,component_class,source_cloud_id,source_placement_id,source_space_id,provenance_json) VALUES(?,?,1,?,'context',?,'lexical_seed',?,?,?,?)""", (cell_id,match["candidate_cloud_id"],strength,strength,match["candidate_cloud_id"],source["id"] if source else None,source["space_id"] if source else None,encode(metadata)))
        return {"id": cell_id, "component_class": "lexical_seed", **metadata}

    def _place_scene_source(self, conn: Any, hive_id: str, space_id: int, scene: Dict[str, Any], probe: Dict[str, Any]) -> Dict[str, Any]:
        scene_id = int(scene["scene_cloud_id"])
        existing = conn.execute("SELECT id FROM hive_cells WHERE hive_id=? AND source_scene_cloud_id=? AND component_class='memory_source'", (hive_id, scene_id)).fetchone()
        if existing:
            return {"id": existing["id"], "component_class": "memory_source", "existing": True}
        count = int(conn.execute("SELECT COUNT(*) FROM hive_cells WHERE hive_id=?", (hive_id,)).fetchone()[0])
        angle = count * 2.399963
        placement = self.repository.create_placement(conn, scene_id, space_id, 420 + math.cos(angle) * (80 + 42 * math.sqrt(count + 1)), 280 + math.sin(angle) * (80 + 42 * math.sqrt(count + 1)), local_activation=.5, local_gravity=.5, metadata={"placement_kind": "memory_source", "source_probe_id": probe["id"]})
        source = conn.execute("""SELECT p.id,p.space_id FROM cloud_placements p JOIN spaces s ON s.id=p.space_id WHERE p.cloud_id=? AND s.space_type='global_field' LIMIT 1""", (scene_id,)).fetchone()
        cell_id, now = f"cell-{uuid.uuid4().hex}", utcnow()
        metadata = {"component_class": "memory_source", "source_probe_id": probe["id"], "source_signal": probe["input"], "source_scene_id": scene_id, "source_text": scene["text"]}
        conn.execute("""INSERT INTO hive_cells(id,hive_id,dominant_cloud_id,hive_placement_id,source_cloud_id,source_placement_id,source_space_id,source_scene_cloud_id,stored_strength,retention,local_activation,component_class,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (cell_id,hive_id,scene_id,placement["id"],scene_id,source["id"] if source else None,source["space_id"] if source else None,scene_id,.5,.5,.5,"memory_source",encode(metadata),now,now))
        conn.execute("""INSERT INTO hive_cell_components(cell_id,cloud_id,composition_share,local_activation,role,effective_strength,component_class,source_cloud_id,source_placement_id,source_space_id,provenance_json) VALUES(?,?,1,.5,'context',.5,'memory_source',?,?,?,?)""", (cell_id,scene_id,scene_id,source["id"] if source else None,source["space_id"] if source else None,encode(metadata)))
        return {"id": cell_id, "component_class": "memory_source", **metadata}

    @staticmethod
    def _cells(conn: Any, hive_id: str) -> List[Dict[str, Any]]:
        rows = conn.execute("""SELECT hc.*, c.canonical_name AS label FROM hive_cells hc JOIN clouds c ON c.id=hc.dominant_cloud_id WHERE hc.hive_id=? ORDER BY hc.created_at""", (hive_id,)).fetchall()
        return [dict(row) for row in rows]


# Compatibility for integrations that still use the former service name.  New
# code must use LexicalCandidateResolver to make its lexical-only role explicit.
ResonanceProbeService = LexicalCandidateResolver
