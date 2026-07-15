"""Local-first hive memory backed by hive-space placements."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from server.tokenizer import tokenize_hierarchical
from .repository import V2Repository, encode, utcnow
from .training import RoleResolver, RussianMorphology


SEARCH_STOP_WORDS = {
    "а", "бы", "в", "во", "где", "да", "же", "и", "из", "к", "как", "когда",
    "кто", "ли", "на", "над", "не", "но", "о", "об", "от", "откуда", "по",
    "под", "при", "про", "с", "со", "там", "то", "у", "что", "чем", "это",
}

ROLE_ALIASES = {
    "subject": {"subject", "agent"},
    "agent": {"subject", "agent"},
    "predicate": {"predicate", "action"},
    "action": {"predicate", "action"},
    "location": {"location", "destination", "source"},
    "destination": {"location", "destination"},
    "source": {"location", "source"},
}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class HiveLocalMemoryConfig:
    local_hit_threshold: float = 0.72
    partial_hit_threshold: float = 0.35
    max_sources: int = 8


@dataclass
class QueryComponent:
    id: str
    surface_form: str
    normalized_form: str
    lexeme: str
    word_form_cloud_id: Optional[int]
    lexeme_cloud_id: Optional[int]
    expected_role: str
    token_index: int
    resolution_state: str = "MISS"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HiveMessageParser:
    def __init__(self) -> None:
        self.morphology = RussianMorphology()
        self.roles = RoleResolver()

    def parse(self, text: str, conn: Any) -> Dict[str, Any]:
        tokenization = tokenize_hierarchical(text)
        tokens = tokenization.all_tokens
        morphologies = [self.morphology.parse(token.normalized) for token in tokens]
        roles = self.roles.resolve(tokens, morphologies)
        components: List[QueryComponent] = []
        for index, (token, morphology, role) in enumerate(zip(tokens, morphologies, roles)):
            row = conn.execute(
                """SELECT wf.cloud_id, wf.lexeme_cloud_id FROM word_forms wf
                WHERE wf.normalized_form = ? LIMIT 1""",
                (token.normalized,),
            ).fetchone()
            components.append(QueryComponent(
                id=f"component-{index}-{token.normalized}",
                surface_form=token.text,
                normalized_form=token.normalized,
                lexeme=morphology.lemma,
                word_form_cloud_id=int(row["cloud_id"]) if row else None,
                lexeme_cloud_id=int(row["lexeme_cloud_id"]) if row and row["lexeme_cloud_id"] else None,
                expected_role=role["role"],
                token_index=index,
            ))
        return {
            "original_text": text,
            "word_forms": [token.normalized for token in tokens],
            "lexemes": [item.lemma for item in morphologies],
            "grammatical_roles": [item["role"] for item in roles],
            "operators": ["negation"] if any(token.normalized == "не" for token in tokens) else [],
            "negation": any(token.normalized == "не" for token in tokens),
            "components": components,
        }


class V2LocalMemoryService:
    def __init__(
        self,
        repository: Optional[V2Repository] = None,
        config: Optional[HiveLocalMemoryConfig] = None,
    ) -> None:
        self.repository = repository or V2Repository()
        self.config = config or HiveLocalMemoryConfig()
        self.parser = HiveMessageParser()

    def create_hive(self, max_cells: int = 24, conversation_id: str = "") -> Dict[str, Any]:
        hive_id = f"hive-{uuid.uuid4().hex}"
        with self.repository.transaction() as conn:
            if conversation_id:
                existing = conn.execute("SELECT id FROM hives WHERE conversation_id = ? AND status = 'ACTIVE' LIMIT 1", (conversation_id,)).fetchone()
                if existing:
                    return self.get_hive(str(existing["id"]), conn)
            global_space, _ = self.repository.get_or_create_space(conn, "global_field", seed=1337)
            random_seed = int(uuid.uuid4().hex[:8], 16)
            hive_space = self.repository.create_space(
                conn,
                "hive_space",
                parent_space_id=int(global_space["id"]),
                seed=random_seed,
            )
            now = utcnow()
            conn.execute(
                """INSERT INTO hives
                (id, space_id, hive_space_id, conversation_id, query_text, query_json, max_cells,
                 capacity, random_seed, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, '', '{}', ?, ?, ?, '{}', ?, ?)""",
                (hive_id, hive_space["id"], hive_space["id"], conversation_id, max_cells, max_cells, random_seed, now, now),
            )
        return self.get_hive(hive_id)

    def _cell_components(self, conn: Any, hive_id: str) -> Dict[int, List[Dict[str, Any]]]:
        result: Dict[int, List[Dict[str, Any]]] = {}
        for row in conn.execute(
            """SELECT hcc.*, hc.id cell_id, hc.stored_strength, hc.retention, hc.local_activation
            FROM hive_cell_components hcc JOIN hive_cells hc ON hc.id = hcc.cell_id
            WHERE hc.hive_id = ?""",
            (hive_id,),
        ):
            result.setdefault(int(row["cloud_id"]), []).append(dict(row))
        return result

    def _route(self, hive_id: str, parsed: Dict[str, Any], conn: Any) -> Dict[str, Any]:
        component_index = self._cell_components(conn, hive_id)
        matches: List[Dict[str, Any]] = []
        anchors: List[Dict[str, Any]] = []
        unresolved: List[Dict[str, Any]] = []
        for component in parsed["components"]:
            candidates = component_index.get(int(component.word_form_cloud_id), []) if component.word_form_cloud_id else []
            best = max(
                candidates,
                key=lambda item: float(item["composition_share"]) * float(item["stored_strength"]) * float(item["retention"]),
                default=None,
            )
            if best:
                support = clamp(max(0.75, float(best["stored_strength"]) * float(best["retention"])))
                component.resolution_state = "LOCAL_HIT" if support >= self.config.local_hit_threshold else "PARTIAL_HIT"
                match = {
                    "component_id": component.id,
                    "cell_id": best["cell_id"],
                    "match_type": "exact_word_form",
                    "local_support": support,
                    "component_share": float(best["composition_share"]),
                    "role_compatibility": 1.0,
                    "matched_cloud_id": component.word_form_cloud_id,
                }
                matches.append(match)
                anchors.append({
                    "component_id": component.id,
                    "cloud_id": component.word_form_cloud_id,
                    "cell_id": best["cell_id"],
                    "role": component.expected_role,
                    "strength": support,
                })
            else:
                component.resolution_state = "MISS"
                unresolved.append(component.to_dict())

        conflict = False
        if parsed["negation"] and matches:
            matched_cells = {item["cell_id"] for item in matches}
            for cell_id in matched_cells:
                row = conn.execute("SELECT metadata_json FROM hive_cells WHERE id = ?", (cell_id,)).fetchone()
                if row and not json.loads(row["metadata_json"] or "{}").get("negation", False):
                    conflict = True
                    break
        if conflict:
            decision = "CONFLICT"
        elif not unresolved and matches:
            decision = "LOCAL_HIT"
        elif matches:
            decision = "PARTIAL_HIT"
        else:
            decision = "MISS"
        external = decision != "LOCAL_HIT"
        return {
            "decision": decision,
            "external_search_required": external,
            "matches": matches,
            "unresolved_components": unresolved,
            "local_anchors": anchors,
            "external_request": {
                "unresolved_components": unresolved,
                "local_anchors": anchors,
                "excluded_known_components": [
                    item.normalized_form for item in parsed["components"] if item.resolution_state == "LOCAL_HIT"
                ],
                "max_bees": min(12, max(2, len(unresolved) * 3)),
                "max_iterations": min(12, max(3, len(unresolved) * 2)),
                "reason": decision,
            } if external else None,
            "reasons": [decision],
            "parsed_message": {
                **{key: value for key, value in parsed.items() if key != "components"},
                "components": [item.to_dict() for item in parsed["components"]],
            },
        }

    def _search(self, decision: Dict[str, Any], conn: Any) -> Dict[str, Any]:
        request = decision.get("external_request")
        if not request:
            return {"sources": [], "bees": [], "iterations": 0, "anchors": decision["local_anchors"]}
        query_components = [
            item for item in decision["parsed_message"]["components"]
            if item.get("lexeme_cloud_id")
            and str(item.get("normalized_form") or "").casefold() not in SEARCH_STOP_WORDS
            and str(item.get("expected_role") or "") not in {"preposition", "conjunction", "particle", "question"}
        ]
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        scene_rows = conn.execute(
            """SELECT p.*, c.cloud_type, c.canonical_name, c.mass, c.stability
               FROM cloud_placements p JOIN spaces s ON s.id = p.space_id
               JOIN clouds c ON c.id = p.cloud_id
               WHERE s.space_type = 'global_field' AND c.cloud_type = 'scene'"""
        ).fetchall()
        for row in scene_rows:
            components = conn.execute(
                """SELECT lexeme_cloud_id, grammatical_role
                   FROM scene_components WHERE scene_cloud_id = ?""",
                (row["cloud_id"],),
            ).fetchall()
            matched = 0.0
            for query in query_components:
                aliases = ROLE_ALIASES.get(str(query.get("expected_role") or ""), {str(query.get("expected_role") or "")})
                compatible = [
                    component for component in components
                    if component["lexeme_cloud_id"] is not None
                    and int(component["lexeme_cloud_id"]) == int(query["lexeme_cloud_id"])
                ]
                if not compatible:
                    continue
                role_score = max(
                    1.0 if str(component["grammatical_role"]) in aliases else 0.85
                    for component in compatible
                )
                matched += role_score
            overlap = matched / max(1, len(query_components))
            if overlap <= 0:
                continue
            score = clamp(
                overlap * 0.85
                + min(1.0, float(row["mass"]) / 10.0) * 0.05
                + float(row["stability"]) * 0.10
            )
            candidate = dict(row)
            candidate["_query_overlap"] = overlap
            ranked.append((score, candidate))

        if not ranked:
            terms = {
                str(item.get("normalized_form") or "").casefold()
                for item in request["unresolved_components"]
                if item.get("normalized_form")
                and str(item.get("normalized_form") or "").casefold() not in SEARCH_STOP_WORDS
            }
            rows = conn.execute(
                """SELECT p.*, c.cloud_type, c.canonical_name, c.mass, c.stability
                   FROM cloud_placements p JOIN spaces s ON s.id = p.space_id
                   JOIN clouds c ON c.id = p.cloud_id
                   WHERE s.space_type = 'global_field'
                     AND c.cloud_type IN ('word_form','lexeme','concept_candidate')"""
            ).fetchall()
            for row in rows:
                words = set(str(row["canonical_name"]).casefold().split())
                overlap = len(words & terms) / max(1, len(terms))
                if overlap <= 0:
                    continue
                score = clamp(
                    overlap * 0.8
                    + min(1.0, float(row["mass"]) / 10.0) * 0.1
                    + float(row["stability"]) * 0.1
                )
                candidate = dict(row)
                candidate["_query_overlap"] = overlap
                ranked.append((score, candidate))
        ranked.sort(key=lambda item: (-item[0], int(item[1]["id"])))
        exact_scenes = [
            item for item in ranked
            if item[1]["cloud_type"] == "scene" and float(item[1]["_query_overlap"]) >= 0.999
        ]
        selected = (exact_scenes or ranked)[: self.config.max_sources]
        sources = [{
            "id": f"source-{row['id']}",
            "placement_id": int(row["id"]),
            "space_id": int(row["space_id"]),
            "cloud_id": int(row["cloud_id"]),
            "label": row["canonical_name"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "fitness": score,
            "state": "ACTIVE",
        } for score, row in selected]
        return {
            "sources": sources,
            "bees": [{"id": f"bee-{index}", "role": "scout", "status": "completed"} for index in range(len(sources))],
            "iterations": request["max_iterations"] if sources else 0,
            "anchors": decision["local_anchors"],
            "excluded": request["excluded_known_components"],
        }

    def _composition(self, conn: Any, source: Dict[str, Any]) -> Dict[int, float]:
        composition: Dict[int, float] = {}
        cloud = conn.execute("SELECT cloud_type FROM clouds WHERE id = ?", (source["cloud_id"],)).fetchone()
        if cloud and cloud["cloud_type"] == "scene":
            for row in conn.execute(
                "SELECT word_form_cloud_id FROM scene_components WHERE scene_cloud_id = ?",
                (source["cloud_id"],),
            ):
                cloud_id = int(row["word_form_cloud_id"])
                composition[cloud_id] = composition.get(cloud_id, 0.0) + 1.0
        else:
            composition[int(source["cloud_id"])] = 1.0
        total = sum(composition.values())
        return {cloud_id: value / total for cloud_id, value in composition.items()}

    def _merge(self, hive: Dict[str, Any], search: Dict[str, Any], parsed: Dict[str, Any], conn: Any) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for source in search["sources"]:
            composition = self._composition(conn, source)
            existing = conn.execute(
                "SELECT * FROM hive_cells WHERE hive_id = ? AND source_cloud_id = ? LIMIT 1",
                (hive["id"], source["cloud_id"]),
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE hive_cells SET stored_strength = MIN(1, stored_strength + 0.05),
                    retention = MIN(1, retention + 0.08), local_activation = 1, updated_at = ? WHERE id = ?""",
                    (utcnow(), existing["id"]),
                )
                results.append({"action": "MERGE_EXISTING", "cell_id": existing["id"], "created_cells": 0, "merged_cells": 1})
                continue

            cells = conn.execute("SELECT id, hive_placement_id, retention FROM hive_cells WHERE hive_id = ?", (hive["id"],)).fetchall()
            if len(cells) >= int(hive["max_cells"]):
                weakest = min(cells, key=lambda item: float(item["retention"]))
                conn.execute("DELETE FROM hive_cells WHERE id = ?", (weakest["id"],))
                conn.execute("DELETE FROM cloud_placements WHERE id = ?", (weakest["hive_placement_id"],))

            index = len(cells)
            angle = index * 2.399963
            x = 420.0 + math.cos(angle) * (80.0 + 42.0 * math.sqrt(index + 1))
            y = 280.0 + math.sin(angle) * (80.0 + 42.0 * math.sqrt(index + 1))
            hive_placement = self.repository.create_placement(
                conn,
                int(source["cloud_id"]),
                int(hive["space_id"]),
                x,
                y,
                local_activation=0.95,
                local_gravity=clamp(source["fitness"]),
                metadata={
                    "placement_kind": "hive",
                    "source_cloud_id": source["cloud_id"],
                    "source_placement_id": source["placement_id"],
                    "source_space_id": source["space_id"],
                },
            )
            cell_id = f"cell-{uuid.uuid4().hex}"
            now = utcnow()
            source_cloud = conn.execute("SELECT cloud_type FROM clouds WHERE id = ?", (source["cloud_id"],)).fetchone()
            conn.execute(
                """INSERT INTO hive_cells
                (id, hive_id, dominant_cloud_id, hive_placement_id, source_cloud_id,
                 source_placement_id, source_space_id, source_scene_cloud_id,
                 stored_strength, retention, local_activation, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, .95, ?, ?, ?)""",
                (
                    cell_id,
                    hive["id"],
                    source["cloud_id"],
                    hive_placement["id"],
                    source["cloud_id"],
                    source["placement_id"],
                    source["space_id"],
                    source["cloud_id"] if source_cloud and source_cloud["cloud_type"] == "scene" else None,
                    clamp(max(0.75, source["fitness"])),
                    encode({"negation": parsed["negation"], "message": parsed["original_text"]}),
                    now,
                    now,
                ),
            )
            for cloud_id, share in composition.items():
                role = "context"
                if source_cloud and source_cloud["cloud_type"] == "scene":
                    role_row = conn.execute("SELECT grammatical_role FROM scene_components WHERE scene_cloud_id=? AND word_form_cloud_id=? LIMIT 1", (source["cloud_id"], cloud_id)).fetchone()
                    role = str(role_row["grammatical_role"]) if role_row else role
                component_class = "core" if role in {"subject", "predicate", "object"} else "context"
                conn.execute(
                    """INSERT INTO hive_cell_components
                    (cell_id, cloud_id, composition_share, local_activation, role,
                     effective_strength, component_class, source_cloud_id, source_placement_id,
                     source_space_id, provenance_json)
                    VALUES (?, ?, ?, .9, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cell_id,
                        cloud_id,
                        share,
                        role,
                        clamp(source["fitness"] * share),
                        component_class,
                        source["cloud_id"],
                        source["placement_id"],
                        source["space_id"],
                        encode({
                            "source_cloud_id": source["cloud_id"],
                            "source_placement_id": source["placement_id"],
                            "source_space_id": source["space_id"],
                        }),
                    ),
                )
            results.append({"action": "CREATE_NEW", "cell_id": cell_id, "created_cells": 1, "merged_cells": 0})
        return results

    def preview(self, hive_id: str, text: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM hives WHERE id = ?", (hive_id,)).fetchone():
                raise KeyError(hive_id)
            return self._route(hive_id, self.parser.parse(text, conn), conn)

    def query(self, hive_id: str, text: str) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            hive_row = conn.execute("SELECT * FROM hives WHERE id = ?", (hive_id,)).fetchone()
            if not hive_row:
                raise KeyError(hive_id)
            hive = dict(hive_row)
            started = time.perf_counter()
            parsed = self.parser.parse(text, conn)
            decision = self._route(hive_id, parsed, conn)
            search = self._search(decision, conn) if decision["external_search_required"] else {
                "sources": [], "bees": [], "iterations": 0, "anchors": decision["local_anchors"]
            }
            merges = self._merge(hive, search, parsed, conn) if search["sources"] else []
            turn = int(conn.execute(
                "SELECT COALESCE(MAX(turn_index), 0) FROM hive_messages WHERE hive_id = ?", (hive_id,)
            ).fetchone()[0]) + 1
            message_id = f"message-{uuid.uuid4().hex[:12]}"
            now = utcnow()
            parsed_json = {
                **{key: value for key, value in parsed.items() if key != "components"},
                "components": [item.to_dict() for item in parsed["components"]],
            }
            conn.execute(
                "INSERT INTO hive_messages(id, hive_id, turn_index, role, text, parsed_json, created_at) VALUES (?, ?, ?, 'user', ?, ?, ?)",
                (message_id, hive_id, turn, text, encode(parsed_json), now),
            )
            for match in decision["matches"]:
                event_id = f"resonance-{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """INSERT INTO hive_resonance_events
                    (id, hive_id, message_id, cell_id, component_cloud_id, reason, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, 'local match', ?, ?)""",
                    (event_id, hive_id, message_id, match["cell_id"], match["matched_cloud_id"], encode(match), now),
                )
                conn.execute(
                    """UPDATE hive_cells SET local_activation = 1, retention = MIN(1, retention + .03),
                    updated_at = ? WHERE id = ?""",
                    (now, match["cell_id"]),
                )
            metrics = {
                "query_components": len(parsed["components"]),
                "local_hits": sum(item.resolution_state == "LOCAL_HIT" for item in parsed["components"]),
                "partial_hits": sum(item.resolution_state == "PARTIAL_HIT" for item in parsed["components"]),
                "misses": sum(item.resolution_state == "MISS" for item in parsed["components"]),
                "external_search": bool(search["sources"]),
                "bees": len(search["bees"]),
                "iterations": search["iterations"],
                "activated_cells": len(decision["matches"]),
                "created_cells": sum(item["created_cells"] for item in merges),
                "merged_cells": sum(item["merged_cells"] for item in merges),
                "total_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            decision_id = f"decision-{uuid.uuid4().hex[:12]}"
            conn.execute(
                """INSERT INTO hive_query_decisions
                (id, hive_id, message_id, decision, external_search_required, anchors_json,
                 unresolved_json, reasons_json, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    decision_id,
                    hive_id,
                    message_id,
                    decision["decision"],
                    int(decision["external_search_required"]),
                    encode(decision["local_anchors"]),
                    encode(decision["unresolved_components"]),
                    encode(decision["reasons"]),
                    encode(metrics),
                    now,
                ),
            )
            for match in decision["matches"]:
                conn.execute(
                    """INSERT INTO hive_cell_matches
                    (decision_id, cell_id, component_id, match_type, local_support, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (decision_id, match["cell_id"], match["component_id"], match["match_type"], match["local_support"], encode(match)),
                )
            conn.execute(
                "UPDATE hives SET query_text = ?, query_json = ?, updated_at = ? WHERE id = ?",
                (text, encode(parsed_json), now, hive_id),
            )
            response = self.get_hive(hive_id, conn)
            response.update({
                "message_id": message_id,
                "decision": decision,
                "resonance_events": decision["matches"],
                "external_search": search,
                "merge_results": merges,
                "metrics": metrics,
            })
            return response

    def get_hive(self, hive_id: str, conn: Any = None) -> Dict[str, Any]:
        def read(connection: Any) -> Dict[str, Any]:
            hive = connection.execute("SELECT * FROM hives WHERE id = ?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            cells = [dict(row) for row in connection.execute(
                """SELECT hc.*, p.x, p.y, p.space_id hive_space_id, p.local_gravity,
                hns.energy, hns.local_stability, hns.eviction_status, hns.velocity_x, hns.velocity_y,
                hns.age_steps, hns.weakening_steps
                FROM hive_cells hc JOIN cloud_placements p ON p.id = hc.hive_placement_id
                LEFT JOIN hive_node_states hns ON hns.hive_id = hc.hive_id AND hns.placement_id = hc.hive_placement_id
                WHERE hc.hive_id = ? ORDER BY hc.retention DESC""",
                (hive_id,),
            )]
            for cell in cells:
                dominant = connection.execute(
                    "SELECT canonical_name FROM clouds WHERE id = ?", (cell["dominant_cloud_id"],)
                ).fetchone()
                cell["label"] = dominant["canonical_name"] if dominant else str(cell["dominant_cloud_id"])
                cell["gravity"] = cell["local_gravity"]
                cell["components"] = [dict(row) for row in connection.execute(
                    """SELECT hcc.*, c.canonical_name, c.cloud_type FROM hive_cell_components hcc
                    JOIN clouds c ON c.id = hcc.cloud_id WHERE hcc.cell_id = ?
                    ORDER BY hcc.composition_share DESC""",
                    (cell["id"],),
                )]
            messages = [dict(row) for row in connection.execute(
                "SELECT id, turn_index, role, text, parsed_json, created_at FROM hive_messages WHERE hive_id = ? ORDER BY turn_index",
                (hive_id,),
            )]
            return {"hive": dict(hive), "cells": cells, "messages": messages}

        if conn is not None:
            return read(conn)
        with self.repository.transaction() as connection:
            return read(connection)

    def events(self, hive_id: str) -> List[Dict[str, Any]]:
        with self.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM hives WHERE id = ?", (hive_id,)).fetchone():
                raise KeyError(hive_id)
            return [dict(row) for row in conn.execute(
                "SELECT * FROM hive_resonance_events WHERE hive_id = ? ORDER BY created_at DESC", (hive_id,)
            )]

    def decisions(self, hive_id: str) -> List[Dict[str, Any]]:
        with self.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM hives WHERE id = ?", (hive_id,)).fetchone():
                raise KeyError(hive_id)
            return [dict(row) for row in conn.execute(
                "SELECT * FROM hive_query_decisions WHERE hive_id = ? ORDER BY created_at DESC", (hive_id,)
            )]

    def matches(self, hive_id: str, cell_id: str) -> List[Dict[str, Any]]:
        with self.repository.transaction() as conn:
            return [dict(row) for row in conn.execute(
                """SELECT hcm.* FROM hive_cell_matches hcm
                JOIN hive_query_decisions hqd ON hqd.id = hcm.decision_id
                WHERE hqd.hive_id = ? AND hcm.cell_id = ? ORDER BY hcm.id DESC""",
                (hive_id, cell_id),
            )]
local_memory_service = V2LocalMemoryService()
