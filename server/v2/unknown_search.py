"""Bounded, evidence-first resolution for unknown query tokens."""

from __future__ import annotations

import difflib
import math
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from .repository import V2Repository, decode, encode, utcnow


MAX_LEMMA_HYPOTHESES = 5
MAX_STRUCTURAL_CANDIDATES = 20
MAX_CONTEXT_CANDIDATES = 8
MAX_SCENES_PER_CANDIDATE = 20
MAX_VIBRATION_STEPS = 5
MAX_STRUCTURAL_CONFIDENCE_WITHOUT_CONTEXT = 0.60
MIN_SHARED_BASE_LENGTH = 3
MIN_STEM_SIMILARITY = 0.45
MIN_STRUCTURAL_SCORE = 0.40
MAX_EDIT_OPERATIONS = 2
MAX_EDIT_DISTANCE_RATIO = 0.45
ENDING_CANDIDATES = ("иями", "ями", "ами", "ого", "ему", "ыми", "ими", "иях", "ах", "ях", "ов", "ев", "ом", "ам", "ям", "ой", "ей", "ую", "юю", "а", "я", "ы", "и", "у", "ю", "е", "о")


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def ngrams(text: str, size: int) -> List[str]:
    return [text[index:index + size] for index in range(max(0, len(text) - size + 1))]


def shared_prefix(left: str, right: str) -> str:
    index = 0
    while index < min(len(left), len(right)) and left[index] == right[index]:
        index += 1
    return left[:index]


def signature(text: str, *, possible: bool = False) -> Dict[str, Any]:
    text = text.casefold().strip()
    endings = [ending for ending in ENDING_CANDIDATES if text.endswith(ending) and len(text) > len(ending) + 1]
    ending = max(endings, key=len, default="")
    stem = text[:-len(ending)] if ending else text
    stems = [stem]
    if len(stem) > 3 and stem[-1:] in {"к", "ч", "ш"}:
        stems.append(stem[:-1])
    return {
        "text": text,
        "length": len(text),
        "characters": list(text),
        "prefixes": [text[:size] for size in range(1, min(5, len(text)) + 1)],
        "suffixes": [text[-size:] for size in range(1, min(4, len(text)) + 1)],
        "bigrams": ngrams(text, 2),
        "trigrams": ngrams(text, 3),
        "possible_stems" if possible else "known_stems": list(dict.fromkeys(stems)),
        "possible_endings" if possible else "known_endings": [ending] if ending else [],
    }


class StructuralIndexService:
    """Persistent fragment index populated only while learning global memory."""

    def record(self, conn: Any, cloud_id: int, text: str) -> Dict[str, Any]:
        value = signature(text)
        conn.execute(
            """INSERT INTO structural_signatures(cloud_id, text, signature_json, updated_at)
            VALUES (?, ?, ?, ?) ON CONFLICT(cloud_id) DO UPDATE SET
            text=excluded.text, signature_json=excluded.signature_json, updated_at=excluded.updated_at""",
            (cloud_id, value["text"], encode(value), utcnow()),
        )
        conn.execute("DELETE FROM structural_index WHERE cloud_id=?", (cloud_id,))
        fragments = {
            "prefix": value["prefixes"], "suffix": value["suffixes"], "bigram": value["bigrams"],
            "trigram": value["trigrams"], "stem": value["known_stems"],
        }
        for kind, values in fragments.items():
            conn.executemany(
                "INSERT OR IGNORE INTO structural_index(index_type, fragment, cloud_id) VALUES (?, ?, ?)",
                [(kind, fragment, cloud_id) for fragment in values if len(fragment) >= 2],
            )
        return value


class UnknownTokenSearchService:
    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()
        from .training import RussianMorphology
        self.morphology = RussianMorphology()

    def start(self, hive_id: str, surface: str, token_index: int, query_role: str = "", query_scene_id: str = "") -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            self._require_hive(conn, hive_id)
            search = self._new_search(hive_id, surface, token_index, query_role, query_scene_id)
            self._store_search(conn, hive_id, search)
            return search

    def step(self, hive_id: str, search_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            search = self._find_search(conn, hive_id, search_id)
            self._advance(conn, search, one_step=True)
            self._store_search(conn, hive_id, search)
            return self._step_response(search)

    def run(self, hive_id: str, search_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            search = self._find_search(conn, hive_id, search_id)
            self._advance(conn, search, one_step=False)
            self._apply_bridge_to_working_hive(conn, hive_id, search)
            self._store_search(conn, hive_id, search)
            return {"search": search, "current_mode": search["current_mode"], "candidates": search["semantic_candidates"]}

    def vibrate(self, hive_id: str, search_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            search = self._find_search(conn, hive_id, search_id)
            changes = []
            for candidate in search["semantic_candidates"]:
                before = candidate["scores"]["semantic_total"]
                support = candidate["scores"].get("context_support", 0.0)
                contradiction = candidate["scores"].get("contradiction", 0.0)
                after = clamp(before * .60 + support * .45 - contradiction * .35)
                candidate["scores"]["semantic_total"] = after
                changes.append({"candidate_cloud_id": candidate["candidate_cloud_id"], "before": round(before, 4), "after": round(after, 4)})
            search["vibration_steps"] += 1
            search["semantic_candidates"].sort(key=lambda item: (-item["scores"]["semantic_total"], item["candidate_lexeme"]))
            if search["semantic_candidates"]:
                search["selected_candidate"] = search["semantic_candidates"][0]
                search["status"] = "resolved" if search["selected_candidate"]["scores"]["semantic_total"] >= .88 else "probable_match"
            self._apply_bridge_to_working_hive(conn, hive_id, search)
            self._store_search(conn, hive_id, search)
            return {"step": search["vibration_steps"], "candidate_changes": changes, "evicted": [], "winner": search["selected_candidate"]}

    def get(self, hive_id: str, search_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            return self._find_search(conn, hive_id, search_id)

    def evidence(self, hive_id: str, search_id: str) -> List[Dict[str, Any]]:
        return self.get(hive_id, search_id)["evidence"]

    def routes(self, hive_id: str, search_id: str) -> List[Dict[str, Any]]:
        return self.get(hive_id, search_id)["bee_missions"]

    def confirm(self, hive_id: str, search_id: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            search = self._find_search(conn, hive_id, search_id)
            if not search.get("selected_candidate"):
                raise ValueError("a candidate must be selected before confirmation")
            search["training_mode"] = "user_confirmed"
            search["status"] = "resolved"
            self._store_search(conn, hive_id, search)
            return search

    def resolve_query_unknowns(self, hive_id: str) -> List[Dict[str, Any]]:
        """Run only for query tokens missing from the global word-form index."""
        with self.repository.transaction() as conn:
            state = self._working_state(conn, hive_id)
            if not state:
                return []
            role_values = state.get("query_frame", {}).get("roles", {})
            searches = []
            for role, token in role_values.items():
                if token.get("status") != "fixed" or not token.get("surface"):
                    continue
                if token.get("part_of_speech") not in {"NOUN", "NPRO"}:
                    continue
                exact = conn.execute("SELECT 1 FROM word_forms WHERE normalized_form=?", (token["normalized"],)).fetchone()
                if exact:
                    continue
                search = self._new_search(hive_id, token["surface"], int(token.get("index", 0)), role, state.get("query_scene", {}).get("id", ""))
                search.update({"query_text": state.get("query_frame", {}).get("source_text", ""), "conversation_id": state.get("conversation_id", ""), "message_id": state.get("message_id", ""), "query_frame_id": state.get("query_frame", {}).get("id", ""), "created_for_surface": token.get("surface", "")})
                self._advance(conn, search, one_step=False)
                self._apply_bridge_to_working_hive(conn, hive_id, search, state)
                searches.append(search)
            if searches:
                self._store_searches(conn, hive_id, searches, state)
            return searches

    def _new_search(self, hive_id: str, surface: str, token_index: int, query_role: str, query_scene_id: str) -> Dict[str, Any]:
        return {
            "id": f"unknown-search-{uuid.uuid4().hex[:12]}", "hive_id": hive_id,
            "query_text": "", "surface": surface.casefold(), "created_for_surface": surface, "token_index": token_index,
            "query_role": query_role, "query_scene_id": query_scene_id, "status": "created",
            "conversation_id": "", "message_id": "", "query_frame_id": "",
            "current_mode": "exact_search", "exact_form_found": False, "exact_lexeme_found": False,
            "lemma_hypotheses": [], "structural_candidates": [], "semantic_candidates": [],
            "candidate_bridges": [], "bee_missions": [], "evidence": [], "selected_candidate": None, "selected_semantic_candidate": None,
            "vibration_steps": 0, "training_mode": "temporary", "local_projection": None,
        }

    def _advance(self, conn: Any, search: Dict[str, Any], one_step: bool) -> None:
        modes = ("exact_search", "lemma_search", "morphology_search", "structural_search", "context_verification", "hive_reasoning")
        start = modes.index(search["current_mode"]) if search["current_mode"] in modes else 0
        for index in range(start, len(modes)):
            mode = modes[index]
            search["current_mode"] = mode
            search["status"] = mode
            if mode == "exact_search":
                exact = conn.execute("SELECT wf.cloud_id, wf.lexeme_cloud_id, l.lemma FROM word_forms wf LEFT JOIN lexemes l ON l.cloud_id=wf.lexeme_cloud_id WHERE wf.normalized_form=?", (search["surface"],)).fetchone()
                search["exact_form_found"] = bool(exact)
                self._mission(search, "EXACT_FORM_BEE", "word_form", [], "exact_form", 1.0 if exact else 0.0)
                if exact:
                    search["exact_lexeme_found"] = bool(exact["lexeme_cloud_id"])
                    search["selected_candidate"] = {"candidate_cloud_id": exact["lexeme_cloud_id"], "candidate_lexeme": exact["lemma"], "scores": {"semantic_total": 1.0}}
                    search["status"], search["current_mode"] = "resolved", "exact_search"
                    return
            elif mode == "lemma_search":
                morph = self.morphology.parse(search["surface"])
                lemma = morph.lemma.casefold()
                search["lemma_hypotheses"] = [{"lemma": lemma, "base": "", "ending": "", "confidence": .78, "part_of_speech": morph.pos_tag}]
                exact = conn.execute("SELECT cloud_id FROM lexemes WHERE lemma=?", (lemma,)).fetchone()
                search["exact_lexeme_found"] = bool(exact)
                self._mission(search, "LEMMA_BEE", "lexeme", [f"lemma_hypothesis:{lemma}"], "lemma", .78)
                if exact:
                    search["selected_candidate"] = {"candidate_cloud_id": int(exact["cloud_id"]), "candidate_lexeme": lemma, "scores": {"semantic_total": .90}}
                    search["status"] = "resolved"
                    return
            elif mode == "morphology_search":
                self._morphology(search)
            elif mode == "structural_search":
                self._structural(conn, search)
            elif mode == "context_verification":
                self._context(conn, search)
            elif mode == "hive_reasoning":
                self._bridge(search)
                search["status"] = "probable_match" if search["selected_candidate"] else "unresolved"
            if one_step:
                if index + 1 < len(modes):
                    search["current_mode"] = modes[index + 1]
                return

    def _morphology(self, search: Dict[str, Any]) -> None:
        raw = signature(search["surface"], possible=True)
        hypotheses = search["lemma_hypotheses"]
        lemma = hypotheses[0]["lemma"] if hypotheses else search["surface"]
        lemma_signature = signature(lemma, possible=True)
        ending = raw["possible_endings"][0] if raw["possible_endings"] else ""
        base = raw["possible_stems"][0]
        hypotheses[0].update({"base": base, "ending": ending, "signature": lemma_signature})
        self._mission(search, "MORPHOLOGY_BEE", "morphology", [f"surface:{search['surface']}", f"base:{base}", f"ending:{ending}"], "morphology", hypotheses[0]["confidence"])

    def _structural(self, conn: Any, search: Dict[str, Any]) -> None:
        lemmas = [item["lemma"] for item in search["lemma_hypotheses"][:MAX_LEMMA_HYPOTHESES]] or [search["surface"]]
        candidate_ids: set[int] = set()
        for lemma in lemmas:
            value = signature(lemma, possible=True)
            keys = [("stem", stem) for stem in value["possible_stems"] if len(stem) >= 3]
            keys += [("trigram", item) for item in value["trigrams"]] + [("prefix", item) for item in value["prefixes"] if len(item) >= 3]
            keys += [("bigram", item) for item in value["bigrams"] if len(item) >= 2]
            for kind, fragment in keys:
                candidate_ids.update(int(row["cloud_id"]) for row in conn.execute("SELECT cloud_id FROM structural_index WHERE index_type=? AND fragment=? LIMIT ?", (kind, fragment, MAX_STRUCTURAL_CANDIDATES)))
        if not candidate_ids:
            self._mission(search, "STRUCTURE_BEE", "word_structure", [], "no_structural_candidate", 0.0)
            return
        marks = ",".join("?" for _ in candidate_ids)
        rows = conn.execute(
            f"""SELECT l.cloud_id, l.lemma, l.pos_tag, ss.signature_json FROM lexemes l
            JOIN structural_signatures ss ON ss.cloud_id=l.cloud_id WHERE l.cloud_id IN ({marks})""", tuple(candidate_ids)
        ).fetchall()
        by_lemma: Dict[str, Dict[str, Any]] = {}
        source_lemma = lemmas[0]
        source_signature = signature(source_lemma, possible=True)
        for row in rows:
            candidate_signature = decode(row["signature_json"], {})
            candidate = str(row["lemma"])
            shared = shared_prefix(source_lemma, candidate)
            source_trigrams, target_trigrams = set(source_signature["trigrams"]), set(candidate_signature.get("trigrams", []))
            trigram = len(source_trigrams & target_trigrams) / max(1, len(source_trigrams | target_trigrams))
            prefix = len(shared) / max(1, min(len(source_lemma), len(candidate)))
            stem = float(any(shared.startswith(item[:3]) and len(shared) >= 3 for item in source_signature["possible_stems"]))
            edit = difflib.SequenceMatcher(a=source_lemma, b=candidate).ratio()
            morphology = .65 if row["pos_tag"] in {search["lemma_hypotheses"][0].get("part_of_speech"), "NOUN"} else .35
            edits = self._edits(candidate, source_lemma)
            edit_ratio = len(edits) / max(1, max(len(candidate), len(source_lemma)))
            raw = clamp(prefix * .32 + trigram * .18 + stem * .25 + edit * .17 + morphology * .08)
            total = min(MAX_STRUCTURAL_CONFIDENCE_WITHOUT_CONTEXT, raw)
            scene_roles = {str(item["grammatical_role"]) for item in conn.execute("SELECT DISTINCT grammatical_role FROM scene_components WHERE lexeme_cloud_id=?", (int(row["cloud_id"]),)).fetchall()}
            rejection_reasons = []
            if len(shared) < MIN_SHARED_BASE_LENGTH:
                rejection_reasons.append("shared_base_too_short")
            if stem < MIN_STEM_SIMILARITY:
                rejection_reasons.append("stem_similarity_zero")
            if total < MIN_STRUCTURAL_SCORE:
                rejection_reasons.append("structural_score_too_low")
            if len(edits) > MAX_EDIT_OPERATIONS or edit_ratio > MAX_EDIT_DISTANCE_RATIO:
                rejection_reasons.append("edit_distance_too_high")
            if scene_roles and search["query_role"] and search["query_role"] not in scene_roles:
                rejection_reasons.append("query_role_mismatch")
            current = by_lemma.get(candidate)
            item = {"id": f"structural-{uuid.uuid4().hex[:10]}", "unknown_surface": search["surface"], "unknown_lemma_hypothesis": source_lemma, "candidate_cloud_id": int(row["cloud_id"]), "candidate_lexeme": candidate, "shared_base": shared, "edit_operations": edits, "hypothesis_type": "possible_typo" if edit >= .6 and len(shared) < 3 else "possible_derivation", "scores": {"prefix_similarity": round(prefix, 4), "trigram_similarity": round(trigram, 4), "stem_similarity": stem, "edit_similarity": round(edit, 4), "edit_distance_ratio": round(edit_ratio, 4), "morphology_similarity": morphology, "structural_total": round(total, 4)}, "status": "REJECTED" if rejection_reasons else "unverified", "rejection_reasons": rejection_reasons, "scene_roles": sorted(scene_roles)}
            if not current or item["scores"]["structural_total"] > current["scores"]["structural_total"]:
                by_lemma[candidate] = item
        search["structural_candidates"] = sorted(by_lemma.values(), key=lambda item: (-item["scores"]["structural_total"], item["candidate_lexeme"]))[:MAX_STRUCTURAL_CANDIDATES]
        for candidate in search["structural_candidates"]:
            self._evidence(search, "STRUCTURE_BEE", "word_structure", candidate["candidate_cloud_id"], "shared_stem" if candidate["status"] != "REJECTED" else "structural_rejection", candidate["shared_base"], candidate["scores"]["structural_total"], [f"query_word_form:{search['surface']}", f"lemma_hypothesis:{source_lemma}", f"global_lexeme:{candidate['candidate_lexeme']}"], polarity="negative" if candidate["status"] == "REJECTED" else "positive", rejection_reasons=candidate.get("rejection_reasons", []))
        self._mission(search, "STRUCTURE_BEE", "word_structure", [item["id"] for item in search["structural_candidates"]], "structural_candidates", .6 if search["structural_candidates"] else 0.0)

    def _context(self, conn: Any, search: Dict[str, Any]) -> None:
        semantic = []
        for structural in search["structural_candidates"][:MAX_CONTEXT_CANDIDATES]:
            scenes = conn.execute(
                """SELECT s.cloud_id, s.sentence_text, sc.grammatical_role FROM scene_components sc
                JOIN scenes s ON s.cloud_id=sc.scene_cloud_id WHERE sc.lexeme_cloud_id=? LIMIT ?""",
                (structural["candidate_cloud_id"], MAX_SCENES_PER_CANDIDATE),
            ).fetchall()
            if structural.get("status") == "REJECTED":
                for row in scenes:
                    self._evidence(search, "SCENE_BEE", "scene", structural["candidate_cloud_id"], "role_mismatch", row["grammatical_role"], .8, [f"global_lexeme:{structural['candidate_lexeme']}", f"scene:{row['cloud_id']}"], polarity="negative")
                continue
            role_support = max((.88 if row["grammatical_role"] in {"object", "definition"} and search["query_role"] == "object" else .45 for row in scenes), default=0.0)
            scene_support = .0
            for row in scenes:
                words = row["sentence_text"].casefold()
                if search["query_role"] == "object" and any(word in words for word in ("прода", "куп", "рынк")):
                    scene_support = max(scene_support, .90)
                else:
                    scene_support = max(scene_support, .45)
                polarity = "positive" if row["grammatical_role"] == search["query_role"] else "negative"
                self._evidence(search, "SCENE_BEE", "scene", structural["candidate_cloud_id"], "scene_role" if polarity == "positive" else "role_mismatch", row["grammatical_role"], role_support if polarity == "positive" else .8, [f"global_lexeme:{structural['candidate_lexeme']}", f"scene:{row['cloud_id']}"], polarity=polarity)
            context = clamp(role_support * .45 + scene_support * .55)
            total = clamp(structural["scores"]["structural_total"] * .35 + context * .65)
            semantic.append({**structural, "supporting_scenes": [{"cloud_id": int(row["cloud_id"]), "text": row["sentence_text"], "role": row["grammatical_role"]} for row in scenes], "scores": {**structural["scores"], "role_support": role_support, "context_support": context, "contradiction": 0.0, "semantic_total": total}, "status": "verified" if context >= .6 else "unverified"})
        search["semantic_candidates"] = sorted(semantic, key=lambda item: (-item["scores"]["semantic_total"], item["candidate_lexeme"]))
        self._mission(search, "CONTEXT_BEE", "scene", [item["id"] for item in search["semantic_candidates"]], "context_verification", max((item["scores"]["context_support"] for item in semantic), default=0.0))

    def _bridge(self, search: Dict[str, Any]) -> None:
        if not search["semantic_candidates"]:
            return
        winner = search["semantic_candidates"][0]
        search["selected_candidate"] = winner
        bridge = {"id": f"bridge-{uuid.uuid4().hex[:12]}", "hive_id": search["hive_id"], "conversation_id": search.get("conversation_id", ""), "message_id": search.get("message_id", ""), "query_frame_id": search.get("query_frame_id", ""), "query_scene_id": search.get("query_scene_id", ""), "created_for_surface": search.get("created_for_surface", search["surface"]), "unknown_token": {"surface": search.get("created_for_surface", search["surface"]), "lemma_hypothesis": search["lemma_hypotheses"][0]["lemma"]}, "global_candidate": {"cloud_id": winner["candidate_cloud_id"], "lexeme": winner["candidate_lexeme"]}, "shared_base": winner["shared_base"], "confidence": winner["scores"]["semantic_total"], "status": "temporary", "evidence_ids": [item["id"] for item in search["evidence"] if item.get("candidate_cloud_id") == winner["candidate_cloud_id"]]}
        search["candidate_bridges"] = [bridge]
        search["selected_semantic_candidate"] = winner
        search["local_projection"] = {"surface": search["surface"], "lemma_hypothesis": search["lemma_hypotheses"][0]["lemma"], "global_lexeme": winner["candidate_lexeme"], "global_cloud_id": winner["candidate_cloud_id"], "status": "temporary"}

    def _apply_bridge_to_working_hive(self, conn: Any, hive_id: str, search: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> None:
        if not search.get("selected_candidate"):
            state = state or self._working_state(conn, hive_id)
            if search.get("current_mode") == "hive_reasoning" and search.get("status") == "unresolved":
                for token in state.get("query_frame", {}).get("tokens", []):
                    if token.get("surface", "").casefold() == search.get("surface"):
                        token["resolution_state"] = "MISS"
                state["display_status"] = "MISS"
                self._save_working_state(conn, hive_id, state)
            return
        state = state or self._working_state(conn, hive_id)
        winner = search["selected_candidate"]
        target = winner["candidate_lexeme"]
        if not state.get("memory_scenes"):
            from .query_scene import QuerySceneService
            query_service = QuerySceneService(self.repository)
            bridge_frame = {
                "requested_role": state.get("query_frame", {}).get("requested_role"),
                "roles": {
                    str(search.get("query_role") or "object"): {
                        "status": "fixed",
                        "lemma": target,
                        "surface": target,
                        "normalized": target,
                    },
                },
            }
            global_scenes = query_service._memory_scenes(
                conn,
                None,
                bridge_frame,
            )
            state["memory_scenes"] = [
                query_service._score_scene(state.get("query_frame", {}), scene, conn)
                for scene in global_scenes
                if any(value.get("lemma") == target for value in scene.get("roles", {}).values() if isinstance(value, dict))
            ][:8]
        self._place_bridge(conn, hive_id, search, winner)
        for token in state.get("query_frame", {}).get("roles", {}).values():
            if token.get("surface", "").casefold() == search["surface"]:
                token["semantic_lemma"] = target
                token["semantic_bridge_id"] = search["candidate_bridges"][0]["id"] if search["candidate_bridges"] else None
        for scene in state.get("memory_scenes", []):
            values = scene.get("roles", {})
            matched = any(value.get("lemma") == target for value in values.values() if isinstance(value, dict))
            if not matched:
                continue
            scores = scene.get("scores", {})
            scores["object_match"] = max(float(scores.get("object_match", 0)), winner["scores"]["semantic_total"])
            scores["semantic_match"] = max(float(scores.get("semantic_match", 0)), winner["scores"]["semantic_total"])
            scores["total_score"] = clamp(float(scores.get("total_score", 0)) + winner["scores"]["semantic_total"] * .35)
        requested = state.get("query_frame", {}).get("requested_role")
        if requested:
            additions = self._role_candidates_from_scenes(state, requested, target, winner)
            existing = {item["lemma"]: item for item in state.get("candidates", [])}
            for addition in additions:
                current = existing.get(addition["lemma"])
                if current is None:
                    state.setdefault("candidates", []).append(addition)
                elif addition["scores"]["total"] > current["scores"].get("total", 0.0):
                    current.update(addition)
            for candidate in additions:
                candidate["cell_id"] = self._place_role_candidate(conn, hive_id, candidate)
                for source_id in candidate.get("sources", []):
                    scene_id = str(source_id).removeprefix("scene-")
                    scene = next((item for item in state.get("memory_scenes", []) if item.get("id") == source_id), None)
                    if scene and scene_id.isdigit():
                        sources = state.setdefault("memory_sources", [])
                        if not any(item.get("source_scene_id") == int(scene_id) for item in sources):
                            sources.append({"id": f"memory-source-{scene_id}", "component_class": "memory_source", "source_scene_id": int(scene_id), "source_text": scene.get("source_text", ""), "query_frame_id": state.get("query_frame_id"), "conversation_id": state.get("conversation_id"), "message_id": state.get("message_id")})
            state["candidates"].sort(key=lambda item: (-item["scores"]["total"], item["lemma"]))
            for role_search in state.setdefault("role_searches", []):
                if role_search.get("target_role") == requested:
                    role_search["role_candidates"] = deepcopy(state["candidates"])
                    role_search["selected_role_candidate"] = None
            for slot in state.get("query_scene", {}).get("slots", []):
                if slot.get("status") == "empty" and slot.get("role") == requested:
                    slot["candidates"] = [item["id"] for item in state["candidates"]]
            state["result_type"] = "PROBABLE_MATCH"
            state["unknown_token_status"] = "PROBABLE_MATCH"
            self._set_pipeline(state, search, len(additions))
            state.setdefault("reasoning_trace", {}).setdefault("stages", []).append({
                "id": f"semantic-bridge-{search['id']}", "stage": "SEMANTIC_BRIDGE",
                "status": "CANDIDATES_FOUND" if additions else "NO_CANDIDATES",
                "output": {
                    "surface": search.get("surface"), "lemma_hypotheses": deepcopy(search.get("lemma_hypotheses", [])),
                    "selected_candidate": deepcopy(search.get("selected_candidate")),
                    "candidate_bridges": deepcopy(search.get("candidate_bridges", [])),
                    "evidence": deepcopy(search.get("evidence", [])),
                    "role_candidates": [item.get("id") for item in additions],
                },
            })
        self._save_working_state(conn, hive_id, state)

    def _place_memory_source(self, conn: Any, hive_id: str, scene_id: int, scene: Dict[str, Any], search: Dict[str, Any]) -> Optional[str]:
        exists = conn.execute("SELECT id FROM hive_cells WHERE hive_id=? AND source_scene_cloud_id=? AND component_class='memory_source' LIMIT 1", (hive_id, scene_id)).fetchone()
        if exists:
            return str(exists["id"])
        hive = conn.execute("SELECT space_id FROM hives WHERE id=?", (hive_id,)).fetchone()
        cloud = conn.execute("SELECT id FROM clouds WHERE id=?", (scene_id,)).fetchone()
        if not hive or not cloud:
            return None
        index = conn.execute("SELECT COUNT(*) FROM hive_cells WHERE hive_id=?", (hive_id,)).fetchone()[0]
        angle = index * 2.399963
        source_confidence = max(.05, min(.95, float(scene.get("scores", {}).get("source_confidence", scene.get("scores", {}).get("total_score", .5)) or .5)))
        placement = self.repository.create_placement(conn, scene_id, int(hive["space_id"]), 420 + math.cos(angle) * (80 + 42 * math.sqrt(index + 1)), 280 + math.sin(angle) * (80 + 42 * math.sqrt(index + 1)), local_activation=source_confidence, local_gravity=source_confidence, metadata={"placement_kind": "memory_source", "scene_id": scene_id, "search_id": search["id"], "selection_status": "SELECTED"})
        source = conn.execute("SELECT p.id, p.space_id FROM cloud_placements p JOIN spaces s ON s.id=p.space_id WHERE p.cloud_id=? AND s.space_type='global_field' LIMIT 1", (scene_id,)).fetchone()
        cell_id, now = f"cell-{uuid.uuid4().hex}", utcnow()
        metadata = {"component_class": "memory_source", "selection_status": "SELECTED", "query_session_id": search.get("query_session_id"), "source_scene_id": scene_id, "source_text": scene.get("source_text", ""), "query_frame_id": search.get("query_frame_id"), "message_id": search.get("message_id"), "created_for_surface": search.get("created_for_surface", ""), "retrieval_scope": scene.get("retrieval_scope", "LOCAL"), "provenance": scene.get("provenance", {})}
        conn.execute("""INSERT INTO hive_cells(id,hive_id,dominant_cloud_id,hive_placement_id,source_cloud_id,source_placement_id,source_space_id,source_scene_cloud_id,stored_strength,retention,local_activation,component_class,metadata_json,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (cell_id,hive_id,scene_id,placement["id"],scene_id,source["id"] if source else None,source["space_id"] if source else None,scene_id,source_confidence,source_confidence,source_confidence,"memory_source",encode(metadata),now,now))
        components = conn.execute("""SELECT sc.word_form_cloud_id, sc.grammatical_role, wf.normalized_form FROM scene_components sc JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id WHERE sc.scene_cloud_id=? ORDER BY sc.token_index""", (scene_id,)).fetchall()
        for component in components:
            conn.execute("""INSERT INTO hive_cell_components(cell_id,cloud_id,composition_share,local_activation,role,effective_strength,component_class,source_cloud_id,source_placement_id,source_space_id,provenance_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (cell_id,int(component["word_form_cloud_id"]),1/max(1,len(components)),source_confidence,component["grammatical_role"],source_confidence,"memory_source",scene_id,source["id"] if source else None,source["space_id"] if source else None,encode({"source_scene_id":scene_id,"surface":component["normalized_form"]})))
        return cell_id

    def _place_bridge(self, conn: Any, hive_id: str, search: Dict[str, Any], winner: Dict[str, Any]) -> None:
        """Place the global candidate in the hive; the bridge itself remains local metadata."""
        existing = conn.execute(
            "SELECT id FROM hive_cells WHERE hive_id=? AND dominant_cloud_id=? AND component_class='semantic_bridge' LIMIT 1",
            (hive_id, winner["candidate_cloud_id"]),
        ).fetchone()
        if existing:
            return
        hive = conn.execute("SELECT space_id, max_cells FROM hives WHERE id=?", (hive_id,)).fetchone()
        cells = conn.execute("SELECT id, hive_placement_id, retention FROM hive_cells WHERE hive_id=?", (hive_id,)).fetchall()
        if len(cells) >= int(hive["max_cells"]):
            weakest = min(cells, key=lambda item: float(item["retention"]))
            conn.execute("DELETE FROM hive_cells WHERE id=?", (weakest["id"],))
            conn.execute("DELETE FROM cloud_placements WHERE id=?", (weakest["hive_placement_id"],))
        index = len(cells)
        angle = index * 2.399963
        placement = self.repository.create_placement(
            conn, int(winner["candidate_cloud_id"]), int(hive["space_id"]),
            420.0 + math.cos(angle) * (80.0 + 42.0 * math.sqrt(index + 1)),
            280.0 + math.sin(angle) * (80.0 + 42.0 * math.sqrt(index + 1)),
            local_activation=winner["scores"]["semantic_total"], local_gravity=winner["scores"]["semantic_total"],
            metadata={"placement_kind": "candidate_bridge", "search_id": search["id"], "bridge_id": search["candidate_bridges"][0]["id"] if search["candidate_bridges"] else None},
        )
        global_source = conn.execute(
            """SELECT p.id, p.space_id FROM cloud_placements p JOIN spaces s ON s.id=p.space_id
            WHERE p.cloud_id=? AND s.space_type='global_field' LIMIT 1""", (winner["candidate_cloud_id"],)
        ).fetchone()
        cell_id, now = f"cell-{uuid.uuid4().hex}", utcnow()
        conn.execute(
            """INSERT INTO hive_cells(id, hive_id, dominant_cloud_id, hive_placement_id, source_cloud_id,
            source_placement_id, source_space_id, stored_strength, retention, local_activation, component_class,
            metadata_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'semantic_bridge', ?, ?, ?)""",
            (cell_id, hive_id, winner["candidate_cloud_id"], placement["id"], winner["candidate_cloud_id"],
             global_source["id"] if global_source else None, global_source["space_id"] if global_source else None,
             winner["scores"]["semantic_total"], winner["scores"]["semantic_total"], winner["scores"]["semantic_total"],
            encode({"search_id": search["id"], "bridge": search["candidate_bridges"][0] if search["candidate_bridges"] else {}, "conversation_id": search.get("conversation_id", ""), "message_id": search.get("message_id", ""), "query_frame_id": search.get("query_frame_id", ""), "created_for_surface": search.get("created_for_surface", "")}), now, now),
        )
        conn.execute(
            """INSERT INTO hive_cell_components(cell_id, cloud_id, composition_share, local_activation, role,
            effective_strength, component_class, source_cloud_id, source_placement_id, source_space_id, provenance_json)
            VALUES (?, ?, 1, ?, ?, ?, 'semantic_bridge', ?, ?, ?, ?)""",
            (cell_id, winner["candidate_cloud_id"], winner["scores"]["semantic_total"], search["query_role"] or "context",
             winner["scores"]["semantic_total"], winner["candidate_cloud_id"], global_source["id"] if global_source else None,
             global_source["space_id"] if global_source else None, encode({"search_id": search["id"], "source_cloud_id": winner["candidate_cloud_id"]})),
        )

    def _place_role_candidate(self, conn: Any, hive_id: str, candidate: Dict[str, Any]) -> Optional[str]:
        cloud = conn.execute("SELECT cloud_id FROM lexemes WHERE lemma=?", (candidate["lemma"],)).fetchone()
        if not cloud:
            return
        cloud_id = int(cloud["cloud_id"])
        exists = conn.execute(
            "SELECT id FROM hive_cells WHERE hive_id=? AND dominant_cloud_id=? AND component_class='role_candidate' LIMIT 1",
            (hive_id, cloud_id),
        ).fetchone()
        if exists:
            return str(exists["id"])
        hive = conn.execute("SELECT space_id, max_cells FROM hives WHERE id=?", (hive_id,)).fetchone()
        cells = conn.execute("SELECT id, hive_placement_id, retention FROM hive_cells WHERE hive_id=?", (hive_id,)).fetchall()
        if len(cells) >= int(hive["max_cells"]):
            weakest = min(cells, key=lambda item: float(item["retention"]))
            conn.execute("DELETE FROM hive_cells WHERE id=?", (weakest["id"],))
            conn.execute("DELETE FROM cloud_placements WHERE id=?", (weakest["hive_placement_id"],))
        index = len(cells)
        angle = index * 2.399963
        strength = candidate["scores"]["total"]
        placement = self.repository.create_placement(
            conn, cloud_id, int(hive["space_id"]),
            420.0 + math.cos(angle) * (80.0 + 42.0 * math.sqrt(index + 1)),
            280.0 + math.sin(angle) * (80.0 + 42.0 * math.sqrt(index + 1)),
            local_activation=strength, local_gravity=strength,
            metadata={"placement_kind": "role_candidate", "candidate_id": candidate["id"]},
        )
        source_scene = candidate["sources"][0].removeprefix("scene-") if candidate.get("sources") else None
        source = conn.execute(
            """SELECT p.id, p.space_id FROM cloud_placements p JOIN spaces s ON s.id=p.space_id
            WHERE p.cloud_id=? AND s.space_type='global_field' LIMIT 1""", (cloud_id,)
        ).fetchone()
        cell_id, now = f"cell-{uuid.uuid4().hex}", utcnow()
        metadata = {"candidate_id": candidate["id"], "query_session_id": candidate.get("query_session_id", ""), "competition_group_id": candidate.get("competition_group_id"), "target_role": candidate["target_role"], "source_scene_ids": candidate.get("sources", []), "preposition": candidate.get("preposition", ""), "conversation_id": candidate.get("conversation_id", ""), "message_id": candidate.get("message_id", ""), "query_frame_id": candidate.get("query_frame_id", ""), "created_for_surface": candidate.get("created_for_surface", "")}
        conn.execute(
            """INSERT INTO hive_cells(id, hive_id, dominant_cloud_id, hive_placement_id, source_cloud_id,
            source_placement_id, source_space_id, source_scene_cloud_id, stored_strength, retention, local_activation,
            component_class, metadata_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'role_candidate', ?, ?, ?)""",
            (cell_id, hive_id, cloud_id, placement["id"], cloud_id, source["id"] if source else None,
             source["space_id"] if source else None, int(source_scene) if source_scene and source_scene.isdigit() else None,
             strength, strength, strength, encode(metadata), now, now),
        )
        conn.execute(
            """INSERT INTO hive_cell_components(cell_id, cloud_id, composition_share, local_activation, role,
            effective_strength, component_class, source_cloud_id, source_placement_id, source_space_id, provenance_json)
            VALUES (?, ?, 1, ?, ?, ?, 'role_candidate', ?, ?, ?, ?)""",
            (cell_id, cloud_id, strength, candidate["target_role"], strength, cloud_id,
             source["id"] if source else None, source["space_id"] if source else None, encode(metadata)),
        )
        return cell_id

    def _set_pipeline(self, state: Dict[str, Any], search: Dict[str, Any], candidate_count: int) -> None:
        frame = state.get("query_frame", {})
        for token in frame.get("tokens", []):
            normalized = token.get("normalized")
            if normalized in {"где", "кто", "что", "чем", "когда"}:
                token.update({"component_type": "question_operator", "expected_role": search["query_role"] if normalized != "где" else "location", "resolution_state": "QUESTION_OPERATOR"})
            elif token.get("surface", "").casefold() == search["surface"]:
                token.update({"component_type": "query_object", "semantic_lemma": search["selected_candidate"]["candidate_lexeme"], "semantic_bridge_id": search["candidate_bridges"][0]["id"] if search["candidate_bridges"] else None, "resolution_state": "BRIDGED_PROBABLE"})
            elif token.get("part_of_speech") in {"VERB", "INFN"}:
                token.update({"component_type": "query_predicate", "resolution_state": "EXACT_FORM_MATCH" if token.get("word_form_cloud_id") else "PARSED_UNGROUNDED"})
            else:
                token.setdefault("resolution_state", "PARSED")
        pipeline = state["pipeline"] = {
            "query_parse": {"status": "RESOLVED"},
            "token_resolution": {"status": "PROBABLE_MATCH"},
            "memory_search": {"status": "ROLE_CANDIDATES_FOUND", "memory_source_count": len(state.get("memory_sources", [])), "candidate_count": candidate_count},
            "query_scene": {"status": "INCOMPLETE"},
            "vibration": {"status": "READY", "current_step": 0},
            "sentence_planning": {"status": "WAITING"},
            "morphology_generation": {"status": "WAITING"},
            "answer": {"status": "PENDING"},
        }
        state.setdefault("hive", {}).update({"status": "ACTIVE", "reasoning_step": 0, "pipeline": pipeline})
        state["query_scene"]["status"] = "INCOMPLETE"
        state["vibration"].update({"status": "ready", "current_step": 0})
        state["answer"] = {"status": "PENDING", "answer_mode": "pending", "confidence": 0.0, "resolved_role": frame.get("requested_role"), "resolved_value": None, "supporting_scenes": [], "surface_answer": None, "status_message": f"Найден {candidate_count} кандидат для роли «{frame.get('requested_role', '').upper()}». Требуется проверка вибрацией."}

    def _role_candidates_from_scenes(self, state: Dict[str, Any], requested: str, target: str, winner: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = []
        for scene in state.get("memory_scenes", []):
            roles = scene.get("roles", {})
            if not any(value.get("lemma") == target for value in roles.values() if isinstance(value, dict)):
                continue
            value = roles.get(requested)
            if not value:
                continue
            confidence = clamp(winner["scores"]["semantic_total"] * .72 + .22)
            result.append({
                "id": f"candidate-{value['lemma']}-{uuid.uuid5(uuid.NAMESPACE_URL, value['lemma']).hex[:8]}",
                "concept_id": f"concept-{value['lemma']}",
                "lemma": value["lemma"],
                "surface": value["surface"],
                "preposition": value.get("preposition", ""),
                "grammatical_features": deepcopy(value.get("grammatical_features", {})),
                "form_provenance": {
                    "source_type": "observed_training_form",
                    "scene_id": scene["id"],
                    "scene_text": scene.get("source_text", ""),
                    "observed_surface": value["surface"],
                    "generated": False,
                },
                "target_role": requested,
                "part_of_speech": value.get("part_of_speech", "NOUN"),
                "entity_type": "entity",
                "sources": [scene["id"]],
                "primary_source_id": scene["id"],
                "conversation_id": state.get("conversation_id", ""),
                "message_id": state.get("message_id", ""),
                "query_frame_id": state.get("query_frame_id", ""),
                "created_for_surface": state.get("created_for_surface", ""),
                "scores": {
                    "role_compatibility": .98,
                    "action_compatibility": .55,
                    "object_compatibility": winner["scores"]["semantic_total"],
                    "grammar_compatibility": .9,
                    "source_confidence": confidence,
                    "resonance": 0.0,
                    "retention": confidence,
                    "activation": confidence,
                    "evidence_confidence": confidence,
                    "semantic_confidence": winner["scores"]["semantic_total"],
                    "answer_confidence": min(confidence, winner["scores"]["semantic_total"]),
                    "contradiction": 0.0,
                    "total": confidence,
                },
                "status": "new",
                "weak_steps": 0,
            })
        return result

    @staticmethod
    def _edits(source: str, target: str) -> List[Dict[str, Any]]:
        operations = []
        matcher = difflib.SequenceMatcher(a=source, b=target)
        for operation, left_start, left_end, right_start, right_end in matcher.get_opcodes():
            if operation != "equal":
                operations.append({"type": operation, "from": source[left_start:left_end], "value": target[right_start:right_end], "position": left_start})
        return operations

    def _mission(self, search: Dict[str, Any], bee_type: str, source_level: str, route: List[str], result: str, strength: float) -> None:
        search["bee_missions"].append({"id": f"mission-{uuid.uuid4().hex[:10]}", "bee_type": bee_type, "source_level": source_level, "route": route, "result": result, "strength": round(strength, 4), "status": "completed"})

    def _evidence(self, search: Dict[str, Any], bee_type: str, source_level: str, candidate_cloud_id: int, evidence_type: str, value: str, strength: float, route: List[str], *, polarity: str = "positive", rejection_reasons: Optional[List[str]] = None) -> None:
        search["evidence"].append({"id": f"evidence-{uuid.uuid4().hex[:10]}", "bee_type": bee_type, "source_level": source_level, "source_cloud_id": candidate_cloud_id, "candidate_cloud_id": candidate_cloud_id, "target_unknown_token": search["surface"], "evidence_type": evidence_type, "value": value, "strength": round(strength, 4), "polarity": polarity, "rejection_reasons": rejection_reasons or [], "route": route})

    @staticmethod
    def _step_response(search: Dict[str, Any]) -> Dict[str, Any]:
        return {"mission": search["bee_missions"][-1] if search["bee_missions"] else None, "evidence": search["evidence"], "new_candidates": search["structural_candidates"], "search_status": search["current_mode"]}

    def _require_hive(self, conn: Any, hive_id: str) -> None:
        if not conn.execute("SELECT 1 FROM hives WHERE id=?", (hive_id,)).fetchone():
            raise KeyError(hive_id)

    def _find_search(self, conn: Any, hive_id: str, search_id: str) -> Dict[str, Any]:
        self._require_hive(conn, hive_id)
        state = self._working_state(conn, hive_id)
        for search in state.get("unknown_token_searches", []):
            if search["id"] == search_id:
                return search
        raise KeyError(search_id)

    @staticmethod
    def _working_state(conn: Any, hive_id: str) -> Dict[str, Any]:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        return decode(row["metadata_json"], {}).get("query_working_memory", {}) if row else {}

    def _store_search(self, conn: Any, hive_id: str, search: Dict[str, Any]) -> None:
        state = self._working_state(conn, hive_id)
        searches = [item for item in state.get("unknown_token_searches", []) if item["id"] != search["id"]]
        searches.append(search)
        self._store_searches(conn, hive_id, searches, state)

    def _store_searches(self, conn: Any, hive_id: str, searches: List[Dict[str, Any]], state: Dict[str, Any]) -> None:
        state["unknown_token_searches"] = searches
        self._save_working_state(conn, hive_id, state)

    @staticmethod
    def _save_working_state(conn: Any, hive_id: str, state: Dict[str, Any]) -> None:
        row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        metadata = decode(row["metadata_json"], {})
        metadata["query_working_memory"] = state
        conn.execute("UPDATE hives SET metadata_json=?, updated_at=? WHERE id=?", (encode(metadata), utcnow(), hive_id))
