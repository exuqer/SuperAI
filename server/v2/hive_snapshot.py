"""Canonical, display-ready projection of persistent hive memory."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from .repository import V2Repository, decode
from .capacity import get_working_occupancy


ROLE_OFFSETS = {
    "agent": (-0.13, 0.0), "subject": (-0.13, 0.0),
    "action": (0.0, 0.0), "predicate": (0.0, 0.0),
    "object": (0.13, 0.0), "location": (0.0, 0.14),
    "destination": (0.13, 0.14), "instrument": (0.0, -0.14),
    "property": (0.13, -0.12),
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _position_seed(value: int) -> tuple[float, float]:
    digest = hashlib.sha256(str(value).encode()).digest()
    angle = int.from_bytes(digest[:4], "big") / 2**32 * math.tau
    return math.cos(angle), math.sin(angle)


class HiveSnapshotProjector:
    """Joins persistent memory, query state and transient dynamics in one DTO."""

    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()

    def project(
        self,
        hive_id: str,
        *,
        include_scenes: bool = True,
        include_words: bool = True,
        include_query: bool = True,
        include_resonance: bool = True,
        include_history: bool = True,
        aggregation: str = "lexeme",
        resonance_step: str | int = "current",
    ) -> Dict[str, Any]:
        aggregation = "word_form" if str(aggregation).lower() in {"word_form", "form", "forms"} else "lexeme"
        with self.repository.transaction() as conn:
            hive = conn.execute("SELECT * FROM hives WHERE id=?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            working = decode(hive["metadata_json"], {}).get("query_working_memory", {})
            cells = [dict(row) for row in conn.execute(
                """SELECT hc.*, hp.x, hp.y, hp.local_gravity, hp.local_activation AS placement_activation,
                          c.canonical_name AS cell_label
                   FROM hive_cells hc JOIN cloud_placements hp ON hp.id=hc.hive_placement_id
                   JOIN clouds c ON c.id=hc.dominant_cloud_id
                   WHERE hc.hive_id=? ORDER BY hc.created_at, hc.id""", (hive_id,)
            )]
            scene_rows = self._scene_rows(conn, hive_id, cells)
            dynamics = self._dynamics(working, hive, resonance_step)
            dynamic_nodes = {str(node.get("cell_id")): node for node in dynamics.get("nodes", [])}
            scenes = self._scenes(scene_rows, cells, dynamic_nodes, working)
            words = self._words(scene_rows, scenes, dynamic_nodes, aggregation)
            projected_cells = self._cells(cells, dynamic_nodes)
            capacity = get_working_occupancy(cells, int(hive["capacity"] or hive["max_cells"]))
            query_overlay = self._query_overlay(working)
            timeline = self._timeline(dynamics) if include_history else []
            resonance = self._resonance(dynamics, working) if include_resonance else {}
            projected_energy = sum(_number(scene["physics"].get("energy")) for scene in scenes) / max(1, len(scenes))
            active_words = sum(1 for word in words if _number(word["local"].get("activation")) >= .5)
            warnings = []
            working_cells_total = len(working.get("working_cells") or [])
            if cells and not projected_cells:
                warnings.append({"code": "WORKING_CELLS_EMPTY", "message": "Cells exist but working_cells projection is empty"})
            if not dynamics.get("nodes"):
                warnings.append({"code": "DYNAMICS_NOT_INITIALIZED", "message": "Static projection is used"})
            center = self._center_of_mass(words)
            status = "IDLE" if not dynamics.get("nodes") else str(dynamics.get("status") or "READY")
            return {
                "schema_version": 1,
                "hive": {
                    "id": str(hive["id"]), "status": str(hive["status"] or "ACTIVE"),
                    "capacity": capacity["capacity"], "occupied_cells": capacity["active_total"], "occupancy": capacity["occupancy"], "pressure": capacity["pressure"],
                    "temperature": _number(hive["current_temperature"], .35),
                    "reasoning_step": int(hive["reasoning_step"] or 0), "energy": round(projected_energy, 6),
                },
                "summary": {
                    "scene_count": len(scenes), "word_count": len(words),
                    "concept_count": 0, "active_word_count": active_words,
                    "query_anchor_count": len(query_overlay.get("anchors", [])),
                    "candidate_scene_count": sum(1 for scene in scenes if scene["status"].get("candidate_status") in {"CANDIDATE", "WINNER"}),
                    "rejected_scene_count": sum(1 for scene in scenes if scene["status"].get("candidate_status") in {"REJECTED", "NO_HIT", "SELF_MATCH"}),
                    "resonance_status": status,
                },
                "cells": projected_cells,
                "scenes": scenes if include_scenes else [],
                "words": words if include_words else [],
                "query_overlay": query_overlay if include_query else {},
                "resonance": resonance,
                "timeline": timeline,
                "field": {"center_of_mass": center, "zones": dynamics.get("zones") or {}},
                "diagnostics": {"warnings": warnings, "counts": {
                    "cells": len(cells), "placements": len(cells),
                    "working_cells": working_cells_total,
                    "dynamic_nodes": len(dynamics.get("nodes") or []), "projected_words": len(words),
                    "cells_total": len(cells), "working_cells_total": working_cells_total,
                    "projected_cells_total": len(projected_cells),
                    "filtered_cells_total": max(0, len(cells) - len(projected_cells)),
                    "projection_error": None,
                }},
            }

    @staticmethod
    def _cells(cells: list[Dict[str, Any]], dynamic_nodes: Dict[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
        projected = []
        for row in cells:
            node = dynamic_nodes.get(str(row["id"]), {})
            position = node.get("position") or {}
            projected.append({
                "id": str(row["id"]), "label": str(row.get("cell_label") or row["id"]),
                "component_class": str(row.get("component_class") or "context"),
                "source_scene_id": int(row["source_scene_cloud_id"]) if row.get("source_scene_cloud_id") is not None else None,
                "position": {
                    "x": _clamp(_number(position.get("x"), _number(row.get("x")) / 1000 if _number(row.get("x")) > 1 else _number(row.get("x")))),
                    "y": _clamp(_number(position.get("y"), _number(row.get("y")) / 700 if _number(row.get("y")) > 1 else _number(row.get("y")))),
                },
                "physics": {
                    "local_activation": _number(node.get("activation"), _number(row.get("local_activation"))),
                    "local_gravity": _number(node.get("gravity"), _number(row.get("local_gravity"))),
                    "stored_strength": _number(row.get("stored_strength")),
                    "retention": _number(node.get("retention"), _number(row.get("retention"))),
                    "energy": _number(node.get("energy"), _number(row.get("placement_activation"), _number(row.get("local_activation")))),
                },
                "projection_status": "PROJECTED",
            })
        return projected

    def _scene_rows(self, conn: Any, hive_id: str, cells: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
        cell_by_scene = {int(cell["source_scene_cloud_id"]): cell for cell in cells if cell.get("source_scene_cloud_id") is not None}
        if not cell_by_scene:
            return []
        marks = ",".join("?" for _ in cell_by_scene)
        rows = conn.execute(
            f"""SELECT s.cloud_id, s.sentence_text, hc.id AS cell_id, hc.hive_placement_id,
                       hc.stored_strength, hc.retention, hc.local_activation, hp.local_gravity,
                       hp.x, hp.y
                FROM scenes s JOIN hive_cells hc ON hc.source_scene_cloud_id=s.cloud_id
                JOIN cloud_placements hp ON hp.id=hc.hive_placement_id
                WHERE hc.hive_id=? AND s.cloud_id IN ({marks})
                  AND s.knowledge_status<>'RETRACTED'
                ORDER BY hc.created_at, s.cloud_id""",
            (hive_id, *cell_by_scene),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["components"] = [dict(component) for component in conn.execute(
                """SELECT sc.token_index, sc.grammatical_role, sc.word_form_cloud_id, sc.lexeme_cloud_id,
                           wf.normalized_form AS surface, COALESCE(l.lemma, wf.normalized_form) AS lemma,
                           c.mass, c.density, c.stability, c.observation_count
                    FROM scene_components sc JOIN word_forms wf ON wf.cloud_id=sc.word_form_cloud_id
                    LEFT JOIN lexemes l ON l.cloud_id=sc.lexeme_cloud_id
                    JOIN clouds c ON c.id=COALESCE(sc.lexeme_cloud_id, sc.word_form_cloud_id)
                    WHERE sc.scene_cloud_id=? ORDER BY sc.token_index""", (row["cloud_id"],)
            )]
            result.append(item)
        return result

    def _scenes(self, rows: list[Dict[str, Any]], cells: list[Dict[str, Any]], dynamic_nodes: Dict[str, Dict[str, Any]], working: Dict[str, Any]) -> list[Dict[str, Any]]:
        query_scenes = {}
        evaluated = {str(item.get("id")): item for item in working.get("memory_scenes") or []}
        for row in rows:
            node = dynamic_nodes.get(str(row["cell_id"]), {})
            physics = {
                "mass": sum(_number(component.get("mass"), 1) for component in row["components"]) / max(1, len(row["components"])),
                "local_activation": _number(node.get("activation"), _number(row["local_activation"])),
                "local_gravity": _number(node.get("gravity"), _number(row["local_gravity"])),
                "stored_strength": _number(row["stored_strength"]), "retention": _number(node.get("retention"), _number(row["retention"])),
                "energy": _number(node.get("energy"), _number(row["local_activation"])),
            }
            roles: Dict[str, Any] = {}
            for component in row["components"]:
                role = str(component.get("grammatical_role") or "context").lower()
                roles.setdefault(role, {"lemma": component["lemma"], "surface": component["surface"], "cloud_id": component.get("lexeme_cloud_id"), "word_form_cloud_id": component.get("word_form_cloud_id")})
            evaluated_scene = evaluated.get(f"scene-{row['cloud_id']}", {})
            scores = evaluated_scene.get("scores") or {}
            candidate_status = str(evaluated_scene.get("candidate_status") or evaluated_scene.get("result_type") or "NOT_EVALUATED")
            result = {
                "id": f"scene-{row['cloud_id']}", "cloud_id": int(row["cloud_id"]), "text": row["sentence_text"], "source": "LOCAL",
                "cell_id": row["cell_id"], "placement_id": int(row["hive_placement_id"]),
                "position": {"x": _clamp(_number(row["x"]) / 1000 if _number(row["x"]) > 1 else _number(row["x"])), "y": _clamp(_number(row["y"]) / 700 if _number(row["y"]) > 1 else _number(row["y"]))},
                "physics": physics,
                "status": {"cell_status": "ACTIVE", "retrieval_status": str(evaluated_scene.get("retrieval_scope") or "NOT_EVALUATED"), "candidate_status": candidate_status, "eviction_status": node.get("eviction_status", "ACTIVE")},
                "roles": roles, "match": {"total_score": _number(scores.get("total_score"), _number(evaluated_scene.get("score"))), "matched_roles": list(evaluated_scene.get("matched_roles") or scores.get("matched_roles") or []), "mismatched_roles": list(evaluated_scene.get("mismatched_roles") or scores.get("mismatched_roles") or []), "selection_reason": str(evaluated_scene.get("selection_reason") or "not evaluated")},
            }
            query_scenes[result["id"]] = result
        return list(query_scenes.values())

    def _words(self, rows: list[Dict[str, Any]], scenes: list[Dict[str, Any]], dynamic_nodes: Dict[str, Dict[str, Any]], aggregation: str) -> list[Dict[str, Any]]:
        scene_by_cloud = {scene["cloud_id"]: scene for scene in scenes}
        groups: Dict[str, list[tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
        for row in rows:
            scene = scene_by_cloud.get(int(row["cloud_id"]))
            if not scene:
                continue
            for component in row["components"]:
                key = str(component["word_form_cloud_id"] if aggregation == "word_form" else component.get("lexeme_cloud_id") or component["word_form_cloud_id"])
                groups[key].append((row, scene, component))
        words = []
        for key, items in groups.items():
            row, scene, first = items[0]
            contributions, surface_counts, role_set = [], defaultdict(int), set()
            total_weight = base_x = base_y = activation = gravity = strength = retention = energy = 0.0
            displacement_x = displacement_y = velocity_x = velocity_y = emitted = received = support = suppression = 0.0
            for source_row, source_scene, component in items:
                role = str(component.get("grammatical_role") or "context").lower()
                dx, dy = ROLE_OFFSETS.get(role, (0.0, 0.0))
                weight = max(.0001, _number(source_scene["physics"]["local_activation"]) * _number(source_scene["physics"]["stored_strength"]) * _number(source_scene["physics"]["retention"]))
                base_x += (source_scene["position"]["x"] + dx) * weight
                base_y += (source_scene["position"]["y"] + dy) * weight
                total_weight += weight
                activation += _number(source_scene["physics"]["local_activation"]) * weight
                gravity += _number(source_scene["physics"]["local_gravity"]) * weight
                strength += _number(source_scene["physics"]["stored_strength"]) * weight
                retention += _number(source_scene["physics"]["retention"]) * weight
                energy += _number(source_scene["physics"]["energy"]) * weight
                dynamic = dynamic_nodes.get(str(source_scene["cell_id"]), {})
                position = dynamic.get("position") or {}
                displacement_x += (_number(position.get("x"), source_scene["position"]["x"]) - source_scene["position"]["x"]) * weight
                displacement_y += (_number(position.get("y"), source_scene["position"]["y"]) - source_scene["position"]["y"]) * weight
                velocity = dynamic.get("velocity") or {}
                velocity_x += _number(velocity.get("x")) * weight
                velocity_y += _number(velocity.get("y")) * weight
                support += sum(max(0.0, _number(force.get("magnitude"), _number(force.get("value")))) for force in dynamic.get("force_breakdown", []) if str(force.get("type") or force.get("name") or "").lower() not in {"competition", "suppression"}) * weight
                suppression += sum(abs(min(0.0, _number(force.get("magnitude"), _number(force.get("value"))))) for force in dynamic.get("force_breakdown", []) if "supp" in str(force.get("type") or force.get("name") or "").lower()) * weight
                emitted += _number(dynamic.get("emitted_energy")) * weight
                received += _number(dynamic.get("received_energy")) * weight
                surface_counts[(component["surface"], component["word_form_cloud_id"])] += 1
                role_set.add(role)
                contributions.append({"scene_id": source_scene["id"], "role": role, "surface": component["surface"], "scene_activation": source_scene["physics"]["local_activation"], "scene_gravity": source_scene["physics"]["local_gravity"], "stored_strength": source_scene["physics"]["stored_strength"]})
            seed_x, seed_y = _position_seed(int(key))
            base_x, base_y = base_x / total_weight, base_y / total_weight
            base_x = _clamp(base_x + seed_x * .008)
            base_y = _clamp(base_y + seed_y * .008)
            local = {"activation": activation / total_weight, "gravity": gravity / total_weight, "stored_strength": strength / total_weight, "retention": retention / total_weight, "energy": energy / total_weight}
            words.append({
                "id": f"hive-word:{key}", "node_type": "word", "lemma_cloud_id": first.get("lexeme_cloud_id"), "word_form_cloud_id": first["word_form_cloud_id"],
                "lemma": first["surface"] if aggregation == "word_form" else first["lemma"],
                "surface_forms": [{"surface": surface, "word_form_cloud_id": form_id, "count": count} for (surface, form_id), count in sorted(surface_counts.items())],
                "global": {"mass": _number(first.get("mass"), 1), "density": _number(first.get("density"), 1), "stability": _number(first.get("stability")), "observation_count": int(first.get("observation_count") or 0)},
                "local": local,
                "position": {"base_x": base_x, "base_y": base_y, "render_x": _clamp(base_x + displacement_x / total_weight), "render_y": _clamp(base_y + displacement_y / total_weight)},
                "roles": sorted(role_set), "scene_support_count": len({item[1]["id"] for item in items}), "contributions": contributions,
                "resonance": {"active": bool(dynamic_nodes), "displacement": [displacement_x / total_weight, displacement_y / total_weight], "velocity": [velocity_x / total_weight, velocity_y / total_weight], "emitted_energy": emitted / total_weight, "received_energy": received / total_weight, "support": support / total_weight, "suppression": suppression / total_weight, "temperature_noise": 0.0},
            })
        return sorted(words, key=lambda word: (-word["global"]["mass"], word["lemma"]))

    def _dynamics(self, working: Dict[str, Any], hive: Any, step: str | int) -> Dict[str, Any]:
        dynamics = dict(working.get("dynamics") or {})
        if step != "current" and isinstance(step, int):
            history = dynamics.get("history") or []
            selected = next((item for item in history if int(item.get("step", -1)) == step), None)
            if selected:
                dynamics = {**dynamics, **selected}
        return dynamics

    @staticmethod
    def _query_overlay(working: Dict[str, Any]) -> Dict[str, Any]:
        frame = working.get("query_frame") or {}
        scene = working.get("query_scene") or {}
        roles = frame.get("roles") or {}
        anchors = [dict(value, role=role) for role, value in roles.items() if isinstance(value, dict) and value.get("status") in {"fixed", "inherited", "resolved"}]
        return {"source_text": frame.get("source_text") or frame.get("original_text") or "", "reconstructed_text": frame.get("reconstructed_query") or "", "requested_role": frame.get("requested_role") or scene.get("requested_role"), "roles": roles, "slots": scene.get("slots") or [], "anchors": anchors, "exclusions": frame.get("exclusions") or []}

    @staticmethod
    def _resonance(dynamics: Dict[str, Any], working: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": dynamics.get("status", "NOT_STARTED"), "step": dynamics.get("step", 0), "temperature": (dynamics.get("temperature") or {}).get("current"), "nodes": dynamics.get("nodes") or [], "session": working.get("resonance_session") or working.get("active_resonance_session")}

    @staticmethod
    def _timeline(dynamics: Dict[str, Any]) -> list[Dict[str, Any]]:
        history = dynamics.get("history") or []
        return [{"step": int(item.get("step", index)), "temperature": item.get("temperature") or item.get("temperature_value"), "center_of_mass": item.get("center_of_mass"), "nodes": item.get("nodes") or [], "evicted": item.get("evicted") or [], "promoted": item.get("promoted") or []} for index, item in enumerate(history)]

    @staticmethod
    def _center_of_mass(words: list[Dict[str, Any]]) -> Dict[str, float]:
        if not words:
            return {"x": .5, "y": .5}
        weighted = [(word, max(.0001, _number(word["global"].get("mass")) * _number(word["local"].get("activation")) * _number(word["local"].get("retention")))) for word in words]
        total = sum(weight for _, weight in weighted)
        return {"x": sum(word["position"]["render_x"] * weight for word, weight in weighted) / total, "y": sum(word["position"]["render_y"] * weight for word, weight in weighted) / total}
