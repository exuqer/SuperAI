"""Read-only, explainable analytics for hive reasoning runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .repository import V2Repository, decode


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


ROLE_LABELS = {
    "subject": "подлежащее",
    "predicate": "сказуемое",
    "object": "дополнение",
    "location": "место",
    "attribute": "определение",
}

VIABILITY = {
    "ACTIVE": 1.0,
    "WEAKENING": 0.65,
    "EVICTION_CANDIDATE": 0.35,
    "EVICTED": 0.10,
}


class HiveAnalyticsService:
    """Build an immutable analytics view from already persisted reasoning data."""

    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()

    @staticmethod
    def _json_row(row: Any) -> Dict[str, Any]:
        item = dict(row)
        for key in list(item):
            if key.endswith("_json"):
                item[key[:-5]] = decode(item.pop(key), {})
        return item

    @staticmethod
    def _run_summary(run: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: run.get(key)
            for key in (
                "id",
                "hive_id",
                "status",
                "reasoning_steps",
                "completed_steps",
                "stop_reason",
                "random_seed",
                "created_at",
                "completed_at",
                "query",
                "config",
            )
        }

    @staticmethod
    def _query_components(conn: Any, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        persisted = query.get("components", [])
        if isinstance(persisted, list) and persisted:
            return [{
                "term": str(item.get("normalized_form") or item.get("surface_form") or "").casefold(),
                "role": str(item.get("expected_role") or "unknown"),
                "word_form_cloud_id": item.get("word_form_cloud_id"),
            } for item in persisted]
        terms = [str(term).casefold() for term in query.get("terms", [])]
        roles = [str(role) for role in query.get("roles", [])]
        components: List[Dict[str, Any]] = []
        for index, term in enumerate(terms):
            row = conn.execute(
                "SELECT cloud_id FROM word_forms WHERE normalized_form = ? LIMIT 1", (term,)
            ).fetchone()
            components.append({
                "term": term,
                "role": roles[index] if index < len(roles) else "unknown",
                "word_form_cloud_id": int(row["cloud_id"]) if row else None,
            })
        return components

    @staticmethod
    def _snapshot_payload(
        nodes: List[Dict[str, Any]],
        clouds: Dict[int, Dict[str, Any]],
        scene_components: Dict[int, List[Dict[str, Any]]],
        cells: Dict[int, str],
        query_components: List[Dict[str, Any]],
        *,
        step: int,
        phase: str,
        created_at: str,
        temperature: float,
        delta: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        displayed_nodes = []
        candidates = []
        for node in nodes:
            cloud = clouds.get(int(node["cloud_id"]), {})
            descriptor = {
                **node,
                "label": cloud.get("canonical_name", str(node["cloud_id"])),
                "cell_id": cells.get(int(node["placement_id"])),
            }
            displayed_nodes.append(descriptor)
            if cloud.get("cloud_type") == "scene":
                candidates.append(HiveAnalyticsService._candidate(
                    node,
                    scene_components.get(int(node["cloud_id"]), []),
                    query_components,
                    descriptor["label"],
                    descriptor["cell_id"],
                ))
        candidates.sort(key=lambda item: (
            -item["candidate_score"],
            -item["semantic_score"],
            -item["dynamic_score"],
            item["scene_label"],
        ))
        active = sum(node.get("eviction_status") == "ACTIVE" for node in nodes)
        weakening = sum(node.get("eviction_status") in {"WEAKENING", "EVICTION_CANDIDATE"} for node in nodes)
        evicted = sum(node.get("eviction_status") == "EVICTED" for node in nodes)
        count = max(1, len(nodes))
        return {
            "step": step,
            "phase": phase,
            "created_at": created_at,
            "temperature": temperature,
            "metrics": {
                "average_activation": sum(float(node.get("local_activation", 0)) for node in nodes) / count,
                "average_retention": sum(float(node.get("retention", 0)) for node in nodes) / count,
                "total_energy": sum(float(node.get("energy", 0)) for node in nodes),
                "active_nodes": active,
                "weakening_nodes": weakening,
                "evicted_nodes": evicted,
            },
            "nodes": displayed_nodes,
            "candidates": candidates,
            "delta": delta or {},
            "events": events or [],
        }

    @staticmethod
    def _catalog(conn: Any, hive_id: str) -> tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]], Dict[int, str]]:
        clouds = {
            int(row["id"]): dict(row)
            for row in conn.execute("SELECT id, cloud_type, canonical_name FROM clouds")
        }
        scene_components: Dict[int, List[Dict[str, Any]]] = {}
        for row in conn.execute(
            """SELECT sc.scene_cloud_id, sc.word_form_cloud_id, sc.grammatical_role,
                      sc.token_index, c.canonical_name
               FROM scene_components sc JOIN clouds c ON c.id = sc.word_form_cloud_id
               ORDER BY sc.scene_cloud_id, sc.token_index"""
        ):
            scene_components.setdefault(int(row["scene_cloud_id"]), []).append(dict(row))
        cells = {
            int(row["hive_placement_id"]): str(row["id"])
            for row in conn.execute(
                "SELECT id, hive_placement_id FROM hive_cells WHERE hive_id = ?", (hive_id,)
            )
        }
        return clouds, scene_components, cells

    @staticmethod
    def _candidate(
        node: Dict[str, Any],
        scene_components: Iterable[Dict[str, Any]],
        query_components: List[Dict[str, Any]],
        label: str,
        cell_id: Optional[str],
    ) -> Dict[str, Any]:
        components = list(scene_components)
        known = [item for item in query_components if item["word_form_cloud_id"] is not None]
        unknown = [item for item in query_components if item["word_form_cloud_id"] is None]
        matches: List[Dict[str, Any]] = []
        for item in known:
            matching = next((component for component in components if (
                int(component["word_form_cloud_id"]) == int(item["word_form_cloud_id"])
                and component["grammatical_role"] == item["role"]
            )), None)
            if matching:
                matches.append({
                    "term": item["term"],
                    "role": item["role"],
                    "label": matching["canonical_name"],
                })
        answers: List[Dict[str, Any]] = []
        for item in unknown:
            matching = next((component for component in components if component["grammatical_role"] == item["role"]), None)
            if matching:
                answers.append({
                    "answer": matching["canonical_name"],
                    "role": item["role"],
                    "question_term": item["term"],
                })
        semantic = len(matches) / len(known) if known else 0.0
        dynamic = (
            clamp(node.get("local_activation", 0.0)) * 0.40
            + clamp(node.get("retention", 0.0)) * 0.35
            + clamp(node.get("local_gravity", 0.0)) * 0.25
        )
        status = str(node.get("eviction_status") or "ACTIVE")
        viability = VIABILITY.get(status, 1.0)
        score = (semantic * 0.70 + dynamic * 0.30) * viability
        matched_text = ", ".join(
            f"{item['term']} ({ROLE_LABELS.get(item['role'], item['role'])})" for item in matches
        )
        answer_text = ", ".join(item["answer"] for item in answers)
        explanation_parts = []
        if matched_text:
            explanation_parts.append(f"совпали {matched_text}")
        if answers:
            explanation_parts.append(
                "ответ извлечён из роли " + ", ".join(
                    ROLE_LABELS.get(item["role"], item["role"]) for item in answers
                )
            )
        if not explanation_parts:
            explanation_parts.append("совпадений ролей не найдено; показана сцена для сравнения")
        return {
            "placement_id": int(node["placement_id"]),
            "cell_id": cell_id,
            "scene_cloud_id": int(node["cloud_id"]),
            "scene_label": label,
            "answer": answer_text or None,
            "matched_components": matches,
            "answer_components": answers,
            "semantic_score": round(semantic, 6),
            "dynamic_score": round(dynamic, 6),
            "viability": viability,
            "candidate_score": round(score, 6),
            "eviction_status": status,
            "explanation": "; ".join(explanation_parts),
        }

    def _analysis(self, conn: Any, hive_id: str, run: Dict[str, Any]) -> Dict[str, Any]:
        clouds, scene_components, cells = self._catalog(conn, hive_id)
        query_components = self._query_components(conn, run.get("query") or {})
        snapshot_rows = [
            self._json_row(row)
            for row in conn.execute(
                "SELECT * FROM hive_reasoning_snapshots WHERE run_id = ? ORDER BY step, id", (run["id"],)
            )
        ]
        snapshots: List[Dict[str, Any]] = []
        for snapshot in snapshot_rows:
            state = snapshot.pop("state", {})
            nodes = state.get("nodes", [])
            snapshots.append(self._snapshot_payload(
                nodes, clouds, scene_components, cells, query_components,
                step=int(snapshot["step"]), phase=str(snapshot["phase"]),
                created_at=str(snapshot["created_at"]), temperature=float(state.get("temperature", 0.0)),
                delta=snapshot.get("delta", {}), events=snapshot.get("events", []),
            ))
        events = [
            self._json_row(row)
            for row in conn.execute(
                "SELECT * FROM hive_reasoning_events WHERE run_id = ? ORDER BY step, created_at", (run["id"],)
            )
        ]
        clusters = [
            self._json_row(row)
            for row in conn.execute(
                "SELECT * FROM hive_resonance_clusters WHERE run_id = ? ORDER BY reasoning_step, id", (run["id"],)
            )
        ]
        return {
            "run": self._run_summary(run),
            "query_components": query_components,
            "snapshots": snapshots,
            "events": events,
            "clusters": clusters,
        }

    def _current(self, conn: Any, hive: Dict[str, Any]) -> Dict[str, Any]:
        hive_id = str(hive["id"])
        clouds, scene_components, cells = self._catalog(conn, hive_id)
        query_components = self._query_components(conn, hive.get("query") or {})
        nodes = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM hive_node_states WHERE hive_id = ? ORDER BY placement_id", (hive_id,)
            )
        ]
        if not nodes:
            nodes = [
                dict(row)
                for row in conn.execute(
                    """SELECT hc.hive_placement_id AS placement_id, hc.dominant_cloud_id AS cloud_id,
                              c.cloud_type AS node_type, p.x, p.y, p.z, 0 AS velocity_x, 0 AS velocity_y,
                              0 AS velocity_z, hc.local_activation, p.local_gravity, hc.stored_strength,
                              hc.retention AS local_stability, hc.retention,
                              hc.local_activation AS energy, 0 AS phase, 1 AS frequency,
                              1 AS temperature_response, 0 AS age_steps, 0 AS activation_count,
                              0 AS last_activated_step, 0 AS weakening_steps, 'ACTIVE' AS eviction_status
                       FROM hive_cells hc JOIN cloud_placements p ON p.id = hc.hive_placement_id
                       JOIN clouds c ON c.id = hc.dominant_cloud_id WHERE hc.hive_id = ?
                       ORDER BY hc.hive_placement_id""",
                    (hive_id,),
                )
            ]
        snapshot = self._snapshot_payload(
            nodes, clouds, scene_components, cells, query_components,
            step=int(hive.get("reasoning_step") or 0), phase="CURRENT",
            created_at=str(hive.get("updated_at") or ""),
            temperature=float(hive.get("current_temperature") or 0.0),
        )
        return {
            "query_components": query_components,
            "snapshot": snapshot,
            "updated_at": hive.get("updated_at"),
        }

    def get(
        self, hive_id: str, run_id: Optional[str] = None, compare_run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            hive_row = conn.execute("SELECT * FROM hives WHERE id = ?", (hive_id,)).fetchone()
            if not hive_row:
                raise KeyError(hive_id)
            hive = self._json_row(hive_row)
            runs = [
                self._json_row(row)
                for row in conn.execute(
                    "SELECT * FROM hive_reasoning_runs WHERE hive_id = ? ORDER BY created_at DESC", (hive_id,)
                )
            ]
            by_id = {str(run["id"]): run for run in runs}
            primary_id = run_id or (str(runs[0]["id"]) if runs else None)
            if primary_id and primary_id not in by_id:
                raise KeyError(primary_id)
            comparison_id = compare_run_id
            if comparison_id is None and primary_id:
                comparison_id = next((str(run["id"]) for run in runs if str(run["id"]) != primary_id), None)
            if comparison_id and comparison_id not in by_id:
                raise KeyError(comparison_id)
            return {
                "hive_id": hive_id,
                "current": self._current(conn, hive),
                "runs": [self._run_summary(run) for run in runs],
                "primary": self._analysis(conn, hive_id, by_id[primary_id]) if primary_id else None,
                "comparison": self._analysis(conn, hive_id, by_id[comparison_id]) if comparison_id else None,
            }
