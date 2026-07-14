"""Local hive routing, resonance and bounded V2 search."""

from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from server.tokenizer import tokenize_hierarchical
from .repository import V2Repository, encode, utcnow
from .training import Morphology, RoleResolver, RussianMorphology
from .migration import synchronize_legacy_field


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _parse_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.casefold() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class HiveLocalMemoryConfig:
    local_hit_threshold: float = float(os.getenv("HIVE_LOCAL_HIT_THRESHOLD", "0.72"))
    partial_hit_threshold: float = float(os.getenv("HIVE_PARTIAL_HIT_THRESHOLD", "0.35"))
    conflict_margin: float = float(os.getenv("HIVE_CONFLICT_MARGIN", "0.12"))
    stale_threshold: float = float(os.getenv("HIVE_STALE_THRESHOLD", "0.25"))
    background_max_support: float = float(os.getenv("HIVE_BACKGROUND_MAX_SUPPORT", "0.20"))
    activation_gain: float = float(os.getenv("ACTIVATION_GAIN", "0.25"))
    reinforcement_gain: float = float(os.getenv("REINFORCEMENT_GAIN", "0.03"))
    refractory_window_seconds: float = float(os.getenv("REFRACTORY_WINDOW_SECONDS", "8"))
    refractory_min_factor: float = float(os.getenv("REFRACTORY_MIN_FACTOR", "0.12"))
    max_mention_factor: float = float(os.getenv("MAX_MENTION_FACTOR", "2.5"))
    min_bees: int = int(os.getenv("HIVE_MIN_BEES", "2"))
    max_bees: int = int(os.getenv("HIVE_MAX_BEES", "12"))
    min_iterations: int = int(os.getenv("HIVE_MIN_ITERATIONS", "3"))
    max_iterations: int = int(os.getenv("HIVE_MAX_ITERATIONS", "12"))
    local_routing_enabled: bool = _parse_bool("HIVE_LOCAL_ROUTING_ENABLED", True)


@dataclass
class QueryComponent:
    id: str
    surface_form: str
    normalized_form: str
    lexeme: Optional[str]
    word_form_cloud_id: Optional[int]
    lexeme_cloud_id: Optional[int]
    concept_cloud_ids: List[int]
    expected_role: str
    token_index: int
    operator_type: Optional[str] = None
    mention_count: int = 1
    resolution_state: str = "MISS"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedMessage:
    original_text: str
    word_forms: List[str]
    lexemes: List[str]
    concept_candidates: List[int]
    scene_candidates: List[int]
    grammatical_roles: List[str]
    operators: List[str]
    modality: str
    tense: Optional[str]
    question_type: Optional[str]
    negation: bool
    mention_counts: Dict[str, int]
    components: List[QueryComponent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["components"] = [component.to_dict() for component in self.components]
        return value


@dataclass
class HiveMatch:
    component_id: str
    cell_id: str
    match_type: str
    local_support: float
    component_share: float
    role_compatibility: float
    matched_cloud_id: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExternalSearchRequest:
    unresolved_components: List[Dict[str, Any]]
    local_anchors: List[Dict[str, Any]]
    expected_roles: List[str]
    excluded_known_components: List[str]
    search_budget: float
    max_iterations: int
    max_bees: int
    novelty_required: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HiveQueryDecision:
    decision: str
    external_search_required: bool
    matches: List[HiveMatch]
    unresolved_components: List[Dict[str, Any]]
    local_anchors: List[Dict[str, Any]]
    external_request: Optional[ExternalSearchRequest]
    reasons: List[str]
    parsed_message: ParsedMessage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "external_search_required": self.external_search_required,
            "matches": [item.to_dict() for item in self.matches],
            "unresolved_components": self.unresolved_components,
            "local_anchors": self.local_anchors,
            "external_request": self.external_request.to_dict() if self.external_request else None,
            "reasons": self.reasons,
            "parsed_message": self.parsed_message.to_dict(),
        }


@dataclass
class HiveMergeResult:
    action: str
    cell_id: Optional[str]
    created_cells: int = 0
    merged_cells: int = 0
    discarded: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _seconds_since(value: Optional[str]) -> float:
    if not value:
        return 10_000.0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
    except (TypeError, ValueError):
        return 10_000.0


class HiveMessageParser:
    def __init__(self) -> None:
        self.morphology = RussianMorphology()
        self.roles = RoleResolver()

    def parse(self, text: str, conn: Any) -> ParsedMessage:
        tokenization = tokenize_hierarchical(text)
        tokens = tokenization.all_tokens
        morphologies = [self.morphology.parse(token.normalized) for token in tokens]
        roles = self.roles.resolve(tokens, morphologies)
        operators: List[str] = []
        negation = False
        for token in tokens:
            if token.normalized == "не":
                negation, _ = True, operators.append("negation")
            elif token.normalized == "ли" or "?" in text:
                if "question" not in operators:
                    operators.append("question")
            elif token.normalized in {"если", "при"}:
                operators.append("condition")
            elif token.normalized in {"когда", "пока", "уже", "будет", "был"}:
                operators.append("time")
        word_forms = [token.normalized for token in tokens]
        mention_counts: Dict[str, int] = {}
        for word in word_forms:
            mention_counts[word] = mention_counts.get(word, 0) + 1
        components: List[QueryComponent] = []
        concept_ids: List[int] = []
        lexemes: List[str] = []
        for index, (token, morph, role) in enumerate(zip(tokens, morphologies, roles)):
            word = conn.execute(
                "SELECT c.id, w.lexeme_cloud_id FROM v2_word_forms w JOIN v2_clouds c ON c.id = w.cloud_id WHERE w.normalized_form = ? LIMIT 1",
                (token.normalized,),
            ).fetchone()
            lexeme_id = int(word["lexeme_cloud_id"]) if word else None
            word_id = int(word["id"]) if word else None
            lexeme = morph.lemma
            if word:
                lexeme_row = conn.execute("SELECT canonical_name FROM v2_clouds WHERE id = ?", (lexeme_id,)).fetchone()
                lexeme = lexeme_row[0] if lexeme_row else morph.lemma
            concepts = [int(row["concept_cloud_id"]) for row in conn.execute(
                "SELECT concept_cloud_id FROM v2_semantic_memberships WHERE lexeme_cloud_id = ? ORDER BY confidence DESC",
                (lexeme_id,),
            ).fetchall()] if lexeme_id else []
            concept_ids.extend(concepts)
            lexemes.append(lexeme)
            components.append(QueryComponent(
                id=f"component-{index}-{token.normalized}", surface_form=token.text,
                normalized_form=token.normalized, lexeme=lexeme, word_form_cloud_id=word_id,
                lexeme_cloud_id=lexeme_id, concept_cloud_ids=concepts,
                expected_role=role["role"], token_index=index,
                operator_type="negation" if token.normalized == "не" else None,
                mention_count=mention_counts[token.normalized],
            ))
        scene_candidates: List[int] = []
        if components:
            query_words = {component.normalized_form for component in components}
            for row in conn.execute(
                "SELECT cloud_id, canonical_text FROM v2_scenes JOIN v2_clouds ON v2_clouds.id = v2_scenes.cloud_id"
            ).fetchall():
                scene_words = set(str(row["canonical_text"]).split())
                if query_words and query_words.issubset(scene_words):
                    scene_candidates.append(int(row["cloud_id"]))
        tense = next((m.features.get("tense") for m in morphologies if m.features.get("tense")), None)
        modality = "question" if "question" in operators else "assertion"
        return ParsedMessage(
            original_text=text, word_forms=word_forms, lexemes=lexemes,
            concept_candidates=list(dict.fromkeys(concept_ids)), scene_candidates=scene_candidates,
            grammatical_roles=[role["role"] for role in roles], operators=list(dict.fromkeys(operators)),
            modality=modality, tense=tense, question_type="yes_no" if modality == "question" else None,
            negation=negation, mention_counts=mention_counts, components=components,
        )


class HiveLookupService:
    QUALITY = {"exact_word_form": 1.0, "normalized_word_form": 0.95, "lexeme": 0.9, "concept": 0.85, "scene_role": 0.9, "composition": 0.75, "background": 0.2}

    def __init__(self, repository: Optional[V2Repository] = None, config: Optional[HiveLocalMemoryConfig] = None) -> None:
        self.repository = repository or V2Repository()
        self.config = config or HiveLocalMemoryConfig()
        self.parser = HiveMessageParser()

    def _cells(self, conn: Any, hive_id: str) -> List[Dict[str, Any]]:
        cells = [dict(row) for row in conn.execute("SELECT * FROM v2_hive_cells WHERE hive_id = ? ORDER BY retention DESC", (hive_id,)).fetchall()]
        for cell in cells:
            cell["components"] = [dict(row) for row in conn.execute(
                "SELECT hc.*, c.canonical_name, c.cloud_type FROM v2_hive_cell_components hc JOIN v2_clouds c ON c.id = hc.cloud_id WHERE hc.cell_id = ?",
                (cell["id"],),
            ).fetchall()]
        return cells

    def lookup(self, hive_id: str, message: ParsedMessage, conn: Any) -> Tuple[List[HiveMatch], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        matches: List[HiveMatch] = []
        anchors: List[Dict[str, Any]] = []
        unresolved: List[Dict[str, Any]] = []
        reasons: List[str] = []
        cells = self._cells(conn, hive_id)
        for component in message.components:
            best: Optional[HiveMatch] = None
            candidate_pool: List[HiveMatch] = []
            for cell in cells:
                for item in cell["components"]:
                    quality = 0.0
                    match_type = "background"
                    if component.word_form_cloud_id == item["cloud_id"]:
                        quality, match_type = self.QUALITY["exact_word_form"], "exact_word_form"
                    elif component.lexeme_cloud_id == item["cloud_id"]:
                        quality, match_type = self.QUALITY["lexeme"], "lexeme"
                    elif item["cloud_id"] in component.concept_cloud_ids:
                        quality, match_type = self.QUALITY["concept"], "concept"
                    if quality <= 0:
                        continue
                    share = clamp(float(item["composition_share"]))
                    role = 1.0
                    metadata = json.loads(cell.get("metadata_json") or "{}")
                    roles = metadata.get("roles", {})
                    if roles.get(str(item["cloud_id"])) and roles[str(item["cloud_id"])] != component.expected_role:
                        role = 0.45
                    cell_support = clamp(float(cell["stored_strength"]) * share * quality * role *
                                         max(0.05, float(cell["local_activation"])) *
                                         max(0.05, float(cell["retention"])) *
                                         max(0.05, 1.0 - _seconds_since(cell.get("last_activated_at")) / 3600.0))
                    candidate = HiveMatch(component.id, cell["id"], match_type, cell_support, share, role, int(item["cloud_id"]), "component share included")
                    candidate_pool.append(candidate)
                    if best is None or candidate.local_support > best.local_support:
                        best = candidate
            ranked_candidates = sorted(candidate_pool, key=lambda item: item.local_support, reverse=True)
            comparable_meanings = len(ranked_candidates) > 1 and (ranked_candidates[0].match_type == ranked_candidates[1].match_type or {ranked_candidates[0].match_type, ranked_candidates[1].match_type} <= {"concept", "scene_role"})
            if len(ranked_candidates) > 1 and abs(ranked_candidates[0].local_support - ranked_candidates[1].local_support) <= self.config.conflict_margin and ranked_candidates[0].cell_id != ranked_candidates[1].cell_id and ranked_candidates[0].matched_cloud_id != ranked_candidates[1].matched_cloud_id and comparable_meanings:
                component.resolution_state = "AMBIGUOUS"
                matches.extend(ranked_candidates[:2])
                unresolved.append(component.to_dict())
                reasons.append(f"{component.normalized_form}: competing local meanings")
                continue
            if best and best.local_support >= self.config.partial_hit_threshold and best.component_share > self.config.background_max_support:
                matches.append(best)
                if best.local_support >= self.config.local_hit_threshold:
                    component.resolution_state = "LOCAL_HIT"
                    anchors.append({"component_id": component.id, "cloud_id": best.matched_cloud_id, "cell_id": best.cell_id, "role": component.expected_role, "strength": best.local_support})
                else:
                    component.resolution_state = "PARTIAL_HIT"
                    unresolved.append(component.to_dict())
            else:
                component.resolution_state = "MISS"
                unresolved.append(component.to_dict())
                reasons.append(f"{component.normalized_form}: local support below threshold")
        for scene_id in message.scene_candidates:
            for cell in cells:
                if int(cell.get("dominant_cloud_id") or -1) != int(scene_id):
                    continue
                support = clamp(float(cell["stored_strength"]) * max(.05, float(cell["local_activation"])) * max(.05, float(cell["retention"])))
                if support >= self.config.partial_hit_threshold:
                    matches.append(HiveMatch(f"scene-{scene_id}", cell["id"], "scene_role", support, 1.0, 1.0, int(scene_id), "ordered scene candidate"))
                    if support >= self.config.local_hit_threshold:
                        anchors.append({"component_id": f"scene-{scene_id}", "cloud_id": int(scene_id), "cell_id": cell["id"], "role": "scene", "strength": support})
        if message.negation:
            for cell in cells:
                metadata = json.loads(cell.get("metadata_json") or "{}")
                if metadata.get("negation") is not None and bool(metadata.get("negation")) != message.negation:
                    if any(match.cell_id == cell["id"] for match in matches):
                        reasons.append("incompatible polarity in active scene")
        return matches, anchors, unresolved, reasons


class HiveQueryRouter:
    def __init__(self, repository: Optional[V2Repository] = None, config: Optional[HiveLocalMemoryConfig] = None) -> None:
        self.repository = repository or V2Repository()
        self.config = config or HiveLocalMemoryConfig()
        self.lookup_service = HiveLookupService(self.repository, self.config)
        self.parser = self.lookup_service.parser

    def route(self, hive_id: str, text: str, conn: Any) -> HiveQueryDecision:
        parsed = self.parser.parse(text, conn)
        matches, anchors, unresolved, reasons = self.lookup_service.lookup(hive_id, parsed, conn)
        local = [match for match in matches if match.local_support >= self.config.local_hit_threshold]
        conflict = any("polarity" in reason for reason in reasons)
        ambiguous = any(component.resolution_state == "AMBIGUOUS" or (len(component.concept_cloud_ids) > 1 and not anchors) for component in parsed.components)
        if conflict:
            decision = "CONFLICT"
        elif ambiguous:
            decision = "AMBIGUOUS"
        elif not unresolved and local:
            decision = "LOCAL_HIT"
        elif local:
            decision = "PARTIAL_HIT"
        else:
            decision = "MISS"
        stale = any(match.local_support < self.config.local_hit_threshold and match.local_support >= self.config.partial_hit_threshold for match in matches)
        matched_component_ids = {match.component_id for match in matches}
        if stale and len(matched_component_ids.intersection({component.id for component in parsed.components})) == len(parsed.components) and decision not in {"CONFLICT", "AMBIGUOUS"}:
            decision = "STALE_HIT"
        need = clamp((len(unresolved) / max(1, len(parsed.components))) * (1.0 if unresolved else 0.0) + (0.35 if conflict or ambiguous else 0.0) + (0.25 if stale else 0.0))
        external = decision != "LOCAL_HIT" and self.config.local_routing_enabled
        if not self.config.local_routing_enabled:
            external, decision = True, "MISS"
            unresolved = [component.to_dict() for component in parsed.components]
        bees = round(self.config.min_bees + (self.config.max_bees - self.config.min_bees) * need)
        iterations = round(self.config.min_iterations + (self.config.max_iterations - self.config.min_iterations) * need)
        unresolved_ids = {item["id"] for item in unresolved}
        request = ExternalSearchRequest(unresolved, anchors, [item["expected_role"] for item in unresolved], [item.normalized_form for item in parsed.components if item.id not in unresolved_ids], need, iterations, bees, need, "; ".join(reasons) or decision)
        return HiveQueryDecision(decision, external, matches, unresolved, anchors, request if external else None, reasons or [decision], parsed)


class HiveResonanceService:
    def __init__(self, repository: Optional[V2Repository] = None, config: Optional[HiveLocalMemoryConfig] = None) -> None:
        self.repository = repository or V2Repository()
        self.config = config or HiveLocalMemoryConfig()

    def resonate(self, hive_id: str, message_id: str, decision: HiveQueryDecision, conn: Any) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        for match in decision.matches:
            if match.local_support < self.config.partial_hit_threshold:
                continue
            key = (match.cell_id, match.component_id)
            if key in seen:
                continue
            seen.add(key)
            component = next((item for item in decision.parsed_message.components if item.id == match.component_id), None)
            if component is None:
                class _SceneComponent:
                    mention_count = 1
                component = _SceneComponent()
            cell = conn.execute("SELECT * FROM v2_hive_cells WHERE id = ?", (match.cell_id,)).fetchone()
            if not cell:
                continue
            refractory = clamp(_seconds_since(cell["last_activated_at"]) / self.config.refractory_window_seconds, self.config.refractory_min_factor, 1.0)
            mention_factor = min(self.config.max_mention_factor, 1.0 + math.log1p(component.mention_count))
            direct = 1.0 if match.match_type in {"exact_word_form", "lexeme"} else 0.5
            gain = self.config.activation_gain * match.local_support * direct * refractory * mention_factor * (1 - float(cell["local_activation"]))
            reinforce = self.config.reinforcement_gain * match.local_support * refractory * (1 - float(cell["stored_strength"]))
            activation = clamp(float(cell["local_activation"]) + gain)
            strength = clamp(float(cell["stored_strength"]) + reinforce)
            retention = clamp(strength * max(0.05, float(cell["composition_cohesion"])) * activation * max(0.05, float(cell["conversation_focus"]) + 0.35))
            now = utcnow()
            conn.execute("UPDATE v2_hive_cells SET local_activation = ?, stored_strength = ?, retention = ?, conversation_focus = ?, activation_count = activation_count + 1, last_activated_at = ?, updated_at = ? WHERE id = ?", (activation, strength, retention, clamp(float(cell["conversation_focus"]) * .85 + .15), now, now, match.cell_id))
            component_row = conn.execute("SELECT * FROM v2_hive_cell_components WHERE cell_id = ? AND cloud_id = ? LIMIT 1", (match.cell_id, match.matched_cloud_id)).fetchone()
            if component_row:
                component_activation = clamp(float(component_row["local_activation"]) + gain * (1.0 if direct else .15) * match.component_share)
                conn.execute("UPDATE v2_hive_cell_components SET local_activation = ?, activation_count = activation_count + 1, last_activated_at = ? WHERE id = ?", (component_activation, now, component_row["id"]))
            event = {"id": f"resonance-{uuid.uuid4().hex[:12]}", "hive_id": hive_id, "message_id": message_id, "cell_id": match.cell_id, "component_cloud_id": match.matched_cloud_id, "reason": "local_match", "payload": {"match": match.to_dict(), "activation_delta": gain, "reinforcement_delta": reinforce, "mention_factor": mention_factor, "refractory_factor": refractory}}
            conn.execute("INSERT INTO v2_hive_resonance_events(id, hive_id, message_id, cell_id, component_cloud_id, reason, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (event["id"], hive_id, message_id, match.cell_id, match.matched_cloud_id, event["reason"], encode(event["payload"]), now))
            events.append(event)
        return events


class V2BoundedSwarm:
    """A bounded global-field sampler; local anchors are never copied globally."""
    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()

    def search(self, request: ExternalSearchRequest, conn: Any) -> Dict[str, Any]:
        terms = {str(item.get("normalized_form", "")).casefold() for item in request.unresolved_components}
        terms.discard("")
        anchors = {str(item.get("cloud_id")) for item in request.local_anchors}
        excluded = {str(item).casefold() for item in request.excluded_known_components}
        global_space = conn.execute("SELECT id FROM v2_spaces WHERE space_type = 'global_field' ORDER BY id LIMIT 1").fetchone()
        if not global_space:
            return {"sources": [], "bees": [], "iterations": 0, "reason": "global field is empty"}
        rows = conn.execute("SELECT p.*, c.cloud_type, c.canonical_name, c.mass, c.stability FROM v2_cloud_placements p JOIN v2_clouds c ON c.id = p.cloud_id WHERE p.space_id = ?", (global_space[0],)).fetchall()
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            canonical = str(row["canonical_name"]).casefold()
            words = set(canonical.split())
            overlap = len(words & terms) / max(1, len(terms)) if terms else 0.0
            anchor_bonus = .12 if str(row["cloud_id"]) in anchors else 0.0
            score = overlap * .75 + min(1.0, float(row["mass"]) / 10.0) * .15 + float(row["stability"]) * .1 + anchor_bonus
            if canonical in excluded or (overlap <= 0 and not anchor_bonus):
                continue
            if overlap > 0 or anchor_bonus:
                ranked.append((score, dict(row)))
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        selected = ranked[: max(1, min(len(ranked), int(request.max_bees)))]
        sources = [{"id": f"source-{row['id']}", "placement_id": row["id"], "cloud_id": row["cloud_id"], "label": row["canonical_name"], "x": row["x"], "y": row["y"], "fitness": score, "state": "ACTIVE"} for score, row in selected]
        return {"sources": sources, "bees": [{"id": f"bee-{index}", "role": "scout", "status": "completed"} for index in range(len(selected))], "iterations": request.max_iterations if selected else 0, "anchors": list(request.local_anchors), "excluded": request.excluded_known_components}


class HiveMergeService:
    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()

    def merge(self, hive_id: str, search: Dict[str, Any], message: ParsedMessage, conn: Any) -> List[HiveMergeResult]:
        results: List[HiveMergeResult] = []
        for source in search.get("sources", []):
            source_row = conn.execute("SELECT * FROM v2_cloud_placements WHERE id = ?", (source["placement_id"],)).fetchone()
            if not source_row:
                continue
            composition: Dict[int, float] = {int(source["cloud_id"]): 1.0}
            cloud = conn.execute("SELECT cloud_type FROM v2_clouds WHERE id = ?", (source["cloud_id"],)).fetchone()
            if cloud and cloud[0] == "scene":
                for row in conn.execute("SELECT word_form_cloud_id FROM v2_scene_components WHERE scene_cloud_id = ?", (source["cloud_id"],)).fetchall():
                    composition[int(row[0])] = composition.get(int(row[0]), 0.0) + 1.0
            total = sum(composition.values())
            composition = {key: value / total for key, value in composition.items()}
            existing_cells = [dict(row) for row in conn.execute("SELECT * FROM v2_hive_cells WHERE hive_id = ?", (hive_id,)).fetchall()]
            best = None
            best_overlap = 0.0
            for cell in existing_cells:
                current = {int(row["cloud_id"]): float(row["composition_share"]) for row in conn.execute("SELECT cloud_id, composition_share FROM v2_hive_cell_components WHERE cell_id = ?", (cell["id"],)).fetchall()}
                overlap = sum(min(current.get(key, 0), value) for key, value in composition.items())
                if overlap > best_overlap:
                    best_overlap, best = overlap, cell
            if best and best_overlap >= .82:
                conn.execute("UPDATE v2_hive_cells SET stored_strength = MIN(1, stored_strength * .97 + ? * .08), retention = MIN(1, retention + ? * .06), updated_at = ? WHERE id = ?", (float(source["fitness"]), float(source["fitness"]), utcnow(), best["id"]))
                results.append(HiveMergeResult("MERGE_EXISTING", best["id"], merged_cells=1, reason="composition similarity >= 0.82"))
                continue
            hive = conn.execute("SELECT max_cells FROM v2_hives WHERE id = ?", (hive_id,)).fetchone()
            max_cells = int(hive["max_cells"]) if hive else 24
            if len(existing_cells) >= max_cells:
                weakest = min(existing_cells, key=lambda row: float(row["retention"]))
                conn.execute("DELETE FROM v2_hive_cells WHERE id = ?", (weakest["id"],))
            index = len(existing_cells)
            angle = index * 2.399963
            cell_id = f"cell-{uuid.uuid4().hex}"
            metadata = {"negation": message.negation, "roles": {str(component.word_form_cloud_id): component.expected_role for component in message.components if component.word_form_cloud_id}}
            created_at = utcnow()
            initial_strength = clamp(max(float(source["fitness"]), .95))
            conn.execute("INSERT INTO v2_hive_cells(id, hive_id, dominant_cloud_id, source_placement_id, source_scene_cloud_id, x, y, stored_strength, query_relevance, composition_cohesion, retention, local_activation, component_activation, conversation_focus, activation_count, last_activated_at, metadata_json, component_class, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 'context', ?, ?)", (cell_id, hive_id, source["cloud_id"], source["placement_id"], source["cloud_id"] if cloud and cloud[0] == "scene" else None, 80 + math.cos(angle) * (30 + 26 * math.sqrt(max(1, index))), 80 + math.sin(angle) * (30 + 26 * math.sqrt(max(1, index))), initial_strength, clamp(float(source["fitness"])), 1.0, 1.0, .95, .95, 1.0, created_at, encode(metadata), created_at, created_at))
            for cloud_id, share in composition.items():
                conn.execute("INSERT INTO v2_hive_cell_components(cell_id, cloud_id, composition_share, local_activation, activation_count, last_activated_at, source_placement_id, provenance_json) VALUES (?, ?, ?, .9, 0, ?, ?, ?)", (cell_id, cloud_id, share, created_at, source["placement_id"], encode({"source_placement_id": source["placement_id"], "message": message.original_text})))
            results.append(HiveMergeResult("CREATE_NEW", cell_id, created_cells=1, reason="new composition"))
        return results


class HiveDecayService:
    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()

    def decay(self, hive_id: str, conn: Any) -> None:
        for row in conn.execute("SELECT id, local_activation, stored_strength, retention, conversation_focus FROM v2_hive_cells WHERE hive_id = ?", (hive_id,)).fetchall():
            current = conn.execute("SELECT last_activated_at FROM v2_hive_cells WHERE id = ?", (row["id"],)).fetchone()
            if current and _seconds_since(current[0]) < 1.0:
                continue
            activation = clamp(float(row["local_activation"]) * .86)
            focus = clamp(float(row["conversation_focus"]) * .96)
            retention = clamp(float(row["stored_strength"]) * activation * max(.05, focus) * .9)
            conn.execute("UPDATE v2_hive_cells SET local_activation = ?, conversation_focus = ?, retention = ?, updated_at = ? WHERE id = ?", (activation, focus, retention, utcnow(), row["id"]))


class V2LocalMemoryService:
    def __init__(self, repository: Optional[V2Repository] = None, config: Optional[HiveLocalMemoryConfig] = None) -> None:
        self.repository = repository or V2Repository()
        self.config = config or HiveLocalMemoryConfig()
        self.router = HiveQueryRouter(self.repository, self.config)
        self.resonance = HiveResonanceService(self.repository, self.config)
        self.swarm = V2BoundedSwarm(self.repository)
        self.merge = HiveMergeService(self.repository)
        self.decay = HiveDecayService(self.repository)

    def _synchronize_legacy_field(self) -> None:
        synchronize_legacy_field(self.repository)

    def create_hive(self, max_cells: int = 24) -> Dict[str, Any]:
        self._synchronize_legacy_field()
        hive_id = f"hive-{uuid.uuid4().hex}"
        with self.repository.transaction() as conn:
            global_space = self.repository.ensure_space(conn, "global_field", seed=1337)
            hive_space = self.repository.create_space(conn, "hive_space", parent_space_id=int(global_space["id"]), seed=int(uuid.uuid4().hex[:8], 16))
            now = utcnow()
            conn.execute("INSERT INTO v2_hives(id, space_id, query_text, query_json, max_cells, created_at, updated_at) VALUES (?, ?, '', '{}', ?, ?, ?)", (hive_id, hive_space["id"], max_cells, now, now))
        return self.get_hive(hive_id)

    def preview(self, hive_id: str, text: str) -> Dict[str, Any]:
        self._synchronize_legacy_field()
        with self.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM v2_hives WHERE id = ?", (hive_id,)).fetchone():
                raise KeyError(hive_id)
            decision = self.router.route(hive_id, text, conn)
            return decision.to_dict()

    def query(self, hive_id: str, text: str) -> Dict[str, Any]:
        self._synchronize_legacy_field()
        with self.repository.transaction() as conn:
            hive = conn.execute("SELECT * FROM v2_hives WHERE id = ?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            turn = int(conn.execute("SELECT COALESCE(MAX(turn_index), 0) FROM v2_hive_messages WHERE hive_id = ?", (hive_id,)).fetchone()[0]) + 1
            message_id = f"message-{uuid.uuid4().hex[:12]}"
            now = utcnow()
            started = time.perf_counter()
            decision = self.router.route(hive_id, text, conn)
            route_ms = round((time.perf_counter() - started) * 1000, 3)
            conn.execute("INSERT INTO v2_hive_messages(id, hive_id, turn_index, text, parsed_json, created_at) VALUES (?, ?, ?, ?, ?, ?)", (message_id, hive_id, turn, text, encode(decision.parsed_message.to_dict()), now))
            events = self.resonance.resonate(hive_id, message_id, decision, conn)
            search_started = time.perf_counter()
            search = self.swarm.search(decision.external_request, conn) if decision.external_search_required and decision.external_request else {"sources": [], "bees": [], "iterations": 0}
            search_ms = round((time.perf_counter() - search_started) * 1000, 3)
            merges = self.merge.merge(hive_id, search, decision.parsed_message, conn) if search.get("sources") else []
            self.decay.decay(hive_id, conn)
            component_count = max(1, len(decision.parsed_message.components))
            local_count = sum(1 for c in decision.parsed_message.components if c.resolution_state == "LOCAL_HIT")
            partial_count = sum(1 for c in decision.parsed_message.components if c.resolution_state == "PARTIAL_HIT")
            misses = sum(1 for c in decision.parsed_message.components if c.resolution_state == "MISS")
            created_cells = sum(item.created_cells for item in merges)
            merged_cells = sum(item.merged_cells for item in merges)
            metrics = {"query_components": len(decision.parsed_message.components), "local_hits": local_count, "partial_hits": partial_count, "misses": misses, "external_search": bool(search.get("sources")), "bees": len(search.get("bees", [])), "iterations": search.get("iterations", 0), "activated_cells": len(events), "created_cells": created_cells, "merged_cells": merged_cells, "local_lookup_ms": route_ms, "external_search_ms": search_ms, "total_ms": round((time.perf_counter() - started) * 1000, 3), "hive_local_hit_rate": local_count / component_count, "external_search_skip_rate": 0.0 if search.get("sources") else 1.0, "partial_search_rate": 1.0 if decision.decision == "PARTIAL_HIT" else 0.0, "duplicate_nectar_rate": merged_cells / max(1, created_cells + merged_cells), "hive_merge_rate": merged_cells / max(1, len(search.get("sources", [])))}
            decision_id = f"decision-{uuid.uuid4().hex[:12]}"
            conn.execute("INSERT INTO v2_hive_query_decisions(id, hive_id, message_id, decision, external_search_required, search_budget_json, anchors_json, unresolved_json, reasons_json, metrics_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (decision_id, hive_id, message_id, decision.decision, int(decision.external_search_required), encode(decision.external_request.to_dict() if decision.external_request else {}), encode(decision.local_anchors), encode(decision.unresolved_components), encode(decision.reasons), encode(metrics), now))
            for match in decision.matches:
                conn.execute("INSERT INTO v2_hive_cell_matches(decision_id, cell_id, component_id, match_type, local_support, role_compatibility, component_share, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (decision_id, match.cell_id, match.component_id, match.match_type, match.local_support, match.role_compatibility, match.component_share, encode({"reason": match.reason})))
            conn.execute("UPDATE v2_hives SET query_text = ?, query_json = ?, updated_at = ? WHERE id = ?", (text, encode(decision.parsed_message.to_dict()), now, hive_id))
            response = self.get_hive(hive_id, conn)
            response.update({"message_id": message_id, "decision": decision.to_dict(), "resonance_events": events, "external_search": search, "merge_results": [item.to_dict() for item in merges], "metrics": metrics})
            return response

    def get_hive(self, hive_id: str, conn: Any = None) -> Dict[str, Any]:
        def read(connection: Any) -> Dict[str, Any]:
            hive = connection.execute("SELECT * FROM v2_hives WHERE id = ?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            cells = [dict(row) for row in connection.execute("SELECT * FROM v2_hive_cells WHERE hive_id = ? ORDER BY retention DESC", (hive_id,)).fetchall()]
            for cell in cells:
                dominant = connection.execute("SELECT canonical_name FROM v2_clouds WHERE id = ?", (cell["dominant_cloud_id"],)).fetchone()
                cell["label"] = dominant[0] if dominant else str(cell["dominant_cloud_id"])
                cell["gravity"] = cell["retention"]
                cell["components"] = [dict(row) for row in connection.execute("SELECT hc.*, c.canonical_name, c.cloud_type FROM v2_hive_cell_components hc JOIN v2_clouds c ON c.id = hc.cloud_id WHERE hc.cell_id = ? ORDER BY hc.composition_share DESC", (cell["id"],)).fetchall()]
            messages = [dict(row) for row in connection.execute("SELECT id, turn_index, text, created_at FROM v2_hive_messages WHERE hive_id = ? ORDER BY turn_index", (hive_id,)).fetchall()]
            return {"hive": dict(hive), "cells": cells, "messages": messages}
        if conn is not None:
            return read(conn)
        with self.repository.transaction() as connection:
            return read(connection)

    def events(self, hive_id: str) -> List[Dict[str, Any]]:
        with self.repository.transaction() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM v2_hive_resonance_events WHERE hive_id = ? ORDER BY created_at DESC", (hive_id,)).fetchall()]

    def decisions(self, hive_id: str) -> List[Dict[str, Any]]:
        with self.repository.transaction() as conn:
            rows = [dict(row) for row in conn.execute("SELECT * FROM v2_hive_query_decisions WHERE hive_id = ? ORDER BY created_at DESC", (hive_id,)).fetchall()]
            for row in rows:
                for key in ("search_budget_json", "anchors_json", "unresolved_json", "reasons_json", "metrics_json"):
                    row[key[:-5] if key.endswith("_json") else key] = json.loads(row.pop(key) or "{}")
            return rows

    def matches(self, hive_id: str, cell_id: str) -> List[Dict[str, Any]]:
        with self.repository.transaction() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM v2_hive_cell_matches WHERE cell_id = ? AND decision_id IN (SELECT id FROM v2_hive_query_decisions WHERE hive_id = ?) ORDER BY id DESC", (cell_id, hive_id)).fetchall()]


local_memory_service = V2LocalMemoryService()
