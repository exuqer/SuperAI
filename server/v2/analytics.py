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
        working = decode(conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()[0], {}).get("query_working_memory", {})
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
            if working.get("candidates") and not snapshots[-1].get("candidates"):
                snapshots[-1]["candidates"] = [{
                    "placement_id": node.get("placement_id"), "cell_id": node.get("cell_id"), "scene_cloud_id": 0,
                    "scene_label": ", ".join(candidate.get("sources") or []), "answer": candidate.get("surface") or candidate.get("lemma"),
                    "matched_components": [], "answer_components": [], "semantic_score": float(candidate.get("scores", {}).get("object_compatibility", 0.0)),
                    "dynamic_score": float(node.get("local_activation", 0.0)), "viability": 1.0,
                    "candidate_score": float(node.get("local_activation", 0.0)), "eviction_status": node.get("eviction_status", "ACTIVE"),
                    "explanation": "кандидат роли из сцены памяти",
                } for node, candidate in zip(snapshots[-1].get("nodes", []), sorted(working["candidates"], key=lambda item: -float(item.get("scores", {}).get("total", 0.0))))]
            if working.get("candidates") and snapshots[-1].get("candidates"):
                ranked = sorted(working["candidates"], key=lambda item: -float(item.get("scores", {}).get("total", 0.0)))
                for index, card in enumerate(snapshots[-1]["candidates"]):
                    if card.get("answer") is None and index < len(ranked):
                        card["answer"] = ranked[index].get("surface") or ranked[index].get("lemma")
                    if index < len(ranked) and card.get("answer") == (ranked[index].get("surface") or ranked[index].get("lemma")):
                        card["semantic_score"] = max(float(card.get("semantic_score", 0.0)), float(ranked[index].get("scores", {}).get("object_compatibility", 0.0)))
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
                    """SELECT hc.hive_placement_id AS placement_id, hc.dominant_cloud_id AS cloud_id, hc.component_class,
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
            if hive.get("metadata", {}).get("query_working_memory", {}).get("candidates"):
                nodes = [node for node in nodes if node.get("component_class") in {"role_candidate", "semantic_bridge"}]
        snapshot = self._snapshot_payload(
            nodes, clouds, scene_components, cells, query_components,
            step=int(hive.get("reasoning_step") or 0), phase="CURRENT",
            created_at=str(hive.get("updated_at") or ""),
            temperature=float(hive.get("current_temperature") or 0.0),
        )
        working = (hive.get("metadata") or {}).get("query_working_memory") or {}
        if working.get("candidates") and not snapshot.get("candidates"):
            snapshot["candidates"] = [{
                "placement_id": node.get("placement_id"), "cell_id": node.get("cell_id"), "scene_cloud_id": 0,
                "scene_label": ", ".join(candidate.get("sources") or []), "answer": candidate.get("surface") or candidate.get("lemma"),
                "matched_components": [], "answer_components": [], "semantic_score": float(candidate.get("scores", {}).get("object_compatibility", 0.0)),
                "dynamic_score": float(candidate.get("scores", {}).get("total", 0.0)), "viability": 1.0,
                "candidate_score": float(candidate.get("scores", {}).get("total", 0.0)), "eviction_status": "ACTIVE",
                "explanation": "кандидат роли из сцены памяти",
            } for node, candidate in zip(nodes, sorted(working["candidates"], key=lambda item: -float(item.get("scores", {}).get("total", 0.0))))]
        return {
            "query_components": query_components,
            "snapshot": snapshot,
            "updated_at": hive.get("updated_at"),
        }

    def _query_vibration_analysis(self, conn: Any, hive: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Expose query-scene vibration as a first-class analytics timeline.

        Query-scene vibration deliberately does not invoke the global physics
        engine, so it has no ``hive_reasoning_runs`` rows.  Its persisted
        working-memory history is nevertheless a real sequence of candidate
        state changes and must remain visible in the laboratory.
        """
        working = (hive.get("metadata") or {}).get("query_working_memory") or {}
        vibration = working.get("vibration") or {}
        history = vibration.get("history") or []
        candidates = working.get("candidates") or []
        if not history or not candidates:
            return None

        hive_id = str(hive["id"])
        run_id = f"query-vibration-{hive_id}"
        query_frame = working.get("query_frame") or {}
        query_components = []
        for token in query_frame.get("tokens") or []:
            role = next((name for name, value in (query_frame.get("roles") or {}).items() if value.get("index") == token.get("index")), "unknown")
            row = conn.execute("SELECT cloud_id FROM word_forms WHERE normalized_form=? LIMIT 1", (str(token.get("normalized") or "").casefold(),)).fetchone()
            query_components.append({"term": str(token.get("normalized") or "").casefold(), "role": role, "word_form_cloud_id": int(row["cloud_id"]) if row else None})

        cell_rows = [dict(row) for row in conn.execute(
            """SELECT hc.id AS cell_id, hc.hive_placement_id, hc.dominant_cloud_id,
                      c.canonical_name FROM hive_cells hc JOIN clouds c ON c.id=hc.dominant_cloud_id
               WHERE hc.hive_id=?""", (hive_id,)
        )]
        cells_by_cloud = {int(row["dominant_cloud_id"]): row for row in cell_rows}
        lexeme_ids = {
            str(row["lemma"]): int(row["cloud_id"])
            for row in conn.execute("SELECT lemma, cloud_id FROM lexemes")
        }
        first_changes = history[0].get("candidate_changes") or []
        scores = {str(item.get("candidate_id")): float(item.get("before", 0.0)) for item in first_changes}
        for candidate in candidates:
            scores.setdefault(str(candidate.get("id")), float((candidate.get("scores") or {}).get("total", 0.0)))

        def nodes_for(values: Dict[str, float]) -> List[Dict[str, Any]]:
            nodes: List[Dict[str, Any]] = []
            for index, candidate in enumerate(candidates):
                candidate_id = str(candidate.get("id"))
                lemma = str(candidate.get("lemma") or candidate.get("surface") or f"candidate-{index}")
                cloud_id = lexeme_ids.get(lemma, 0)
                cell = cells_by_cloud.get(cloud_id, {})
                score = clamp(values.get(candidate_id, float((candidate.get("scores") or {}).get("total", 0.0))))
                status = str(candidate.get("status") or "new")
                nodes.append({
                    "placement_id": int(cell.get("hive_placement_id") or -(index + 1)),
                    "cloud_id": cloud_id or -(index + 1), "node_type": "role_candidate",
                    "x": 0.0, "y": 0.0, "local_activation": score,
                    "local_gravity": score, "stored_strength": score,
                    "local_stability": score, "retention": score, "energy": score,
                    "eviction_status": "EVICTED" if status == "evicted" else "ACTIVE",
                    "label": lemma, "cell_id": cell.get("cell_id"), "candidate": candidate,
                })
            return nodes

        def snapshot(step: int, phase: str, values: Dict[str, float], event_items: List[Dict[str, Any]]) -> Dict[str, Any]:
            nodes = nodes_for(values)
            count = max(1, len(nodes))
            cards = []
            for node in nodes:
                candidate = node.pop("candidate")
                cards.append({
                    "placement_id": node["placement_id"], "cell_id": node["cell_id"], "scene_cloud_id": 0,
                    "scene_label": ", ".join(candidate.get("sources") or []) or "сцена запроса",
                    "answer": candidate.get("surface") or candidate.get("lemma"),
                    "matched_components": [], "answer_components": [],
                    "semantic_score": round(float((candidate.get("scores") or {}).get("object_compatibility", 0.0)), 6),
                    "dynamic_score": round(float(node["local_activation"]), 6), "viability": 1.0,
                    "candidate_score": round(float(node["local_activation"]), 6),
                    "eviction_status": node["eviction_status"],
                    "explanation": "; ".join((candidate.get("sources") or [])[:2]) or "кандидат роли из сцены памяти",
                })
            cards.sort(key=lambda item: -item["candidate_score"])
            return {
                "step": step, "phase": phase, "created_at": str(hive.get("updated_at") or ""),
                "temperature": 1.0 / max(1, step + 1),
                "metrics": {
                    "average_activation": sum(node["local_activation"] for node in nodes) / count,
                    "average_retention": sum(node["retention"] for node in nodes) / count,
                    "total_energy": sum(node["energy"] for node in nodes),
                    "active_nodes": sum(node["eviction_status"] == "ACTIVE" for node in nodes),
                    "weakening_nodes": 0, "evicted_nodes": sum(node["eviction_status"] == "EVICTED" for node in nodes),
                },
                "nodes": nodes, "candidates": cards, "delta": {}, "events": event_items,
            }

        snapshots = [snapshot(0, "INITIAL", dict(scores), [])]
        events = []
        for event in history:
            step = int(event.get("step") or len(snapshots))
            changes = event.get("candidate_changes") or []
            for change in changes:
                scores[str(change.get("candidate_id"))] = float(change.get("after", 0.0))
            event_items = [{"id": f"query-vibration-{step}-{index}", "step": step, "event_type": "QUERY_VIBRATION", "payload": change} for index, change in enumerate(changes)]
            events.extend(event_items)
            snapshots.append(snapshot(step, "AFTER_SETTLE", dict(scores), event_items))
        run = {
            "id": run_id, "hive_id": hive_id, "status": "COMPLETED",
            "reasoning_steps": int(vibration.get("max_steps") or len(history)), "completed_steps": len(history),
            "stop_reason": "COMPLETED" if vibration.get("status") == "finished" else "IN_PROGRESS",
            "random_seed": 0, "created_at": str(hive.get("updated_at") or ""),
            "completed_at": str(hive.get("updated_at") or ""), "query": {"terms": [item["term"] for item in query_components], "roles": [item["role"] for item in query_components]}, "config": {"engine": "query_scene_vibration"},
        }
        return {"run": self._run_summary(run), "query_components": query_components, "snapshots": snapshots, "events": events, "clusters": []}

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
            query_vibration = self._query_vibration_analysis(conn, hive)
            query_run = query_vibration["run"] if query_vibration else None
            displayed_runs = ([query_run] if query_run else []) + [self._run_summary(run) for run in runs]
            by_id = {str(run["id"]): run for run in runs}
            primary_id = run_id or (str(query_run["id"]) if query_run else (str(runs[0]["id"]) if runs else None))
            if query_run and primary_id == str(query_run["id"]):
                primary = query_vibration
            else:
                primary = self._analysis(conn, hive_id, by_id[primary_id]) if primary_id else None
            if primary_id and primary_id not in by_id:
                if not query_run or primary_id != str(query_run["id"]):
                    raise KeyError(primary_id)
            comparison_id = compare_run_id
            if comparison_id is None and primary_id:
                comparison_id = next((str(run["id"]) for run in displayed_runs if str(run["id"]) != primary_id), None)
            if comparison_id and comparison_id not in by_id:
                if not query_run or comparison_id != str(query_run["id"]):
                    raise KeyError(comparison_id)
            comparison = query_vibration if query_run and comparison_id == str(query_run["id"]) else (self._analysis(conn, hive_id, by_id[comparison_id]) if comparison_id else None)
            return {
                "hive_id": hive_id,
                "current": self._current(conn, hive),
                "runs": displayed_runs,
                "primary": primary,
                "comparison": comparison,
            }
