"""Read-only, explainable analytics for hive reasoning runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .repository import V2Repository, decode


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


ROLE_LABELS = {
    "agent": "деятель",
    "action": "действие",
    "subject": "подлежащее",
    "predicate": "сказуемое",
    "object": "дополнение",
    "location": "место",
    "destination": "направление",
    "source": "источник",
    "instrument": "инструмент",
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
    def _working_query_components(working: Dict[str, Any]) -> List[Dict[str, Any]]:
        frame = working.get("query_frame") or {}
        roles = frame.get("roles") or {}
        requested_role = frame.get("requested_role")
        components: List[Dict[str, Any]] = []
        for role, value in roles.items():
            if role == requested_role or not isinstance(value, dict) or value.get("status") != "fixed":
                continue
            term = str(value.get("normalized") or value.get("surface") or value.get("lemma") or "").casefold()
            if not term:
                continue
            components.append({
                "term": term,
                "role": role,
                "word_form_cloud_id": value.get("word_form_cloud_id"),
            })
        if requested_role:
            components.append({
                "term": str(frame.get("question_word") or "?").casefold(),
                "role": str(requested_role),
                "word_form_cloud_id": None,
            })
        return components

    def _working_candidate_cards(
        self,
        conn: Any,
        hive_id: str,
        working: Dict[str, Any],
        score_overrides: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        candidates = working.get("candidates") or []
        scenes = {
            str(scene.get("id")): scene
            for scene in working.get("memory_scenes") or []
        }
        cells = {
            str(row["cell_id"]): dict(row)
            for row in conn.execute(
                """SELECT hc.id AS cell_id, hc.hive_placement_id, hc.source_scene_cloud_id,
                          hc.dominant_cloud_id, hc.local_activation, hc.retention,
                          p.local_gravity
                   FROM hive_cells hc JOIN cloud_placements p ON p.id=hc.hive_placement_id
                   WHERE hc.hive_id=?""",
                (hive_id,),
            )
        }
        cards: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            scores = candidate.get("scores") or {}
            source_ids = [candidate.get("primary_source_id"), *(candidate.get("sources") or [])]
            source = next((scenes[str(source_id)] for source_id in source_ids if str(source_id) in scenes), {})
            source_roles = source.get("roles") or {}
            matched_roles = source.get("matched_roles") or []
            cell = cells.get(str(candidate.get("cell_id")), {})
            semantic = clamp(scores.get("semantic_total", scores.get("total", 0.0)))
            dynamic = clamp(
                (
                    .20 * clamp(scores.get("gravity", cell.get("local_gravity", 0.0)))
                    + .10 * clamp(scores.get("resonance", 0.0))
                    + .05 * clamp(scores.get("retention", cell.get("retention", 0.0)))
                ) / .35
            )
            decision = clamp(
                (score_overrides or {}).get(
                    str(candidate.get("id")),
                    scores.get("decision_score", scores.get("total", semantic)),
                )
            )
            matched = [{
                "term": str((source_roles.get(role) or {}).get("surface") or (source_roles.get(role) or {}).get("lemma") or role),
                "role": role,
                "label": str((source_roles.get(role) or {}).get("surface") or (source_roles.get(role) or {}).get("lemma") or role),
            } for role in matched_roles]
            answer = str(candidate.get("surface") or candidate.get("lemma") or "") or None
            requested_role = str(candidate.get("target_role") or "unknown")
            answer_components = [{
                "answer": answer,
                "role": requested_role,
                "question_term": str((working.get("query_frame") or {}).get("question_word") or "?"),
            }] if answer else []
            source_text = str(source.get("source_text") or "")
            explanation = str(candidate.get("selection_reason") or "").strip()
            if not explanation:
                explanation = "семантические опоры: " + ", ".join(matched_roles) if matched_roles else "кандидат поддержан сценой памяти"
            cards.append({
                "placement_id": int(cell.get("hive_placement_id") or -(index + 1)),
                "cell_id": candidate.get("cell_id"),
                "scene_cloud_id": int(source.get("cloud_id") or cell.get("source_scene_cloud_id") or 0),
                "scene_label": source_text or ", ".join(str(item) for item in candidate.get("sources") or []) or "сцена запроса",
                "answer": answer,
                "matched_components": matched,
                "answer_components": answer_components,
                "semantic_score": round(semantic, 6),
                "dynamic_score": round(dynamic, 6),
                "viability": 0.1 if str(candidate.get("status")) in {"evicted", "conflict"} else 1.0,
                "candidate_score": round(decision, 6),
                "eviction_status": "EVICTED" if str(candidate.get("status")) in {"evicted", "conflict"} else "ACTIVE",
                "explanation": explanation,
            })
        cards.sort(key=lambda item: (-item["candidate_score"], -item["semantic_score"], item["scene_label"]))
        return cards

    @staticmethod
    def _overlay_working_candidates(snapshot: Dict[str, Any], working: Dict[str, Any]) -> None:
        scenes = {
            str(scene.get("id")): scene
            for scene in working.get("memory_scenes") or []
        }
        scene_ids_by_cloud = {
            int(scene.get("cloud_id")): scene_id
            for scene_id, scene in scenes.items()
            if scene.get("cloud_id") is not None
        }
        candidates = working.get("candidates") or []
        for card in snapshot.get("candidates") or []:
            scene_id = scene_ids_by_cloud.get(int(card.get("scene_cloud_id") or 0))
            matching = [
                candidate for candidate in candidates
                if scene_id and scene_id in {str(source_id) for source_id in candidate.get("sources") or []}
            ]
            if not matching:
                continue
            candidate = max(
                matching,
                key=lambda item: float((item.get("scores") or {}).get("decision_score", (item.get("scores") or {}).get("total", 0.0))),
            )
            scores = candidate.get("scores") or {}
            answer = candidate.get("surface") or candidate.get("lemma")
            semantic = max(
                float(card.get("semantic_score", 0.0)),
                float(scores.get("object_compatibility", 0.0)),
                float(scores.get("semantic_total", 0.0)),
            )
            card["answer"] = answer
            card["answer_components"] = [{
                "answer": answer,
                "role": candidate.get("target_role") or "unknown",
                "question_term": (working.get("query_frame") or {}).get("question_word") or "?",
            }]
            card["semantic_score"] = round(clamp(semantic), 6)
            card["candidate_score"] = round(
                (clamp(semantic) * .70 + float(card.get("dynamic_score", 0.0)) * .30)
                * float(card.get("viability", 1.0)),
                6,
            )
            card["explanation"] = candidate.get("selection_reason") or "кандидат роли из сцены памяти"
        snapshot["candidates"].sort(key=lambda item: (
            -float(item.get("candidate_score", 0.0)),
            -float(item.get("semantic_score", 0.0)),
            str(item.get("scene_label") or ""),
        ))

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
        metadata_row = conn.execute("SELECT metadata_json FROM hives WHERE id=?", (hive_id,)).fetchone()
        working = decode(metadata_row["metadata_json"], {}).get("query_working_memory", {}) if metadata_row else {}
        run_terms = [str(term).casefold() for term in (run.get("query") or {}).get("terms", [])]
        working_terms = [
            str(token.get("normalized") or "").casefold()
            for token in (working.get("query_frame") or {}).get("tokens") or []
        ]
        same_query = bool(run_terms and run_terms == working_terms)
        query_components = self._working_query_components(working) if same_query else self._query_components(conn, run.get("query") or {})
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
            if same_query and working.get("candidates"):
                self._overlay_working_candidates(snapshots[-1], working)
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
        working = (hive.get("metadata") or {}).get("query_working_memory") or {}
        query_components = self._working_query_components(working) or self._query_components(conn, hive.get("query") or {})
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
                    """SELECT hc.id AS cell_id, hc.hive_placement_id AS placement_id, hc.dominant_cloud_id AS cloud_id, hc.component_class,
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
        candidate_cell_ids = {
            str(candidate.get("cell_id"))
            for candidate in working.get("candidates") or []
            if candidate.get("cell_id")
        }
        if candidate_cell_ids:
            candidate_nodes = [
                node for node in nodes
                if str(node.get("cell_id") or cells.get(int(node.get("placement_id") or 0))) in candidate_cell_ids
            ]
            if candidate_nodes:
                nodes = candidate_nodes
        snapshot = self._snapshot_payload(
            nodes, clouds, scene_components, cells, query_components,
            step=int(hive.get("reasoning_step") or 0), phase="CURRENT",
            created_at=str(hive.get("updated_at") or ""),
            temperature=float(hive.get("current_temperature") or 0.0),
        )
        if working.get("query_frame"):
            snapshot["candidates"] = self._working_candidate_cards(conn, hive_id, working)
        return {
            "query_components": query_components,
            "snapshot": snapshot,
            "updated_at": hive.get("updated_at"),
        }

    def _query_vibration_analysis(self, conn: Any, hive: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Expose query-scene vibration as a first-class analytics timeline."""
        working = (hive.get("metadata") or {}).get("query_working_memory") or {}
        vibration = working.get("vibration") or {}
        history = vibration.get("history") or []
        candidates = working.get("candidates") or []
        if not history or not candidates:
            return None

        hive_id = str(hive["id"])
        run_id = f"query-vibration-{hive_id}"
        query_components = self._working_query_components(working)

        cell_rows = [dict(row) for row in conn.execute(
            """SELECT hc.id AS cell_id, hc.hive_placement_id, hc.dominant_cloud_id,
                      hc.local_activation, hc.retention, p.local_gravity, c.canonical_name
               FROM hive_cells hc JOIN clouds c ON c.id=hc.dominant_cloud_id
               JOIN cloud_placements p ON p.id=hc.hive_placement_id
               WHERE hc.hive_id=?""", (hive_id,)
        )]
        cells_by_id = {str(row["cell_id"]): row for row in cell_rows}
        first_changes = history[0].get("candidate_changes") or []
        scores = {str(item.get("candidate_id")): float(item.get("before", 0.0)) for item in first_changes}
        for candidate in candidates:
            scores.setdefault(str(candidate.get("id")), float((candidate.get("scores") or {}).get("total", 0.0)))

        def nodes_for(values: Dict[str, float]) -> List[Dict[str, Any]]:
            nodes: List[Dict[str, Any]] = []
            for index, candidate in enumerate(candidates):
                candidate_id = str(candidate.get("id"))
                lemma = str(candidate.get("lemma") or candidate.get("surface") or f"candidate-{index}")
                cell = cells_by_id.get(str(candidate.get("cell_id")), {})
                cloud_id = int(cell.get("dominant_cloud_id") or 0)
                score = clamp(values.get(candidate_id, float((candidate.get("scores") or {}).get("total", 0.0))))
                status = str(candidate.get("status") or "new")
                nodes.append({
                    "placement_id": int(cell.get("hive_placement_id") or -(index + 1)),
                    "cloud_id": cloud_id or -(index + 1), "node_type": "role_candidate",
                    "x": 0.0, "y": 0.0, "local_activation": score,
                    "local_gravity": score, "stored_strength": score,
                    "local_stability": score, "retention": score, "energy": score,
                    "eviction_status": "EVICTED" if status == "evicted" else "ACTIVE",
                    "label": lemma, "cell_id": candidate.get("cell_id"), "candidate_id": candidate_id,
                })
            return nodes

        def snapshot(step: int, phase: str, values: Dict[str, float], event_items: List[Dict[str, Any]]) -> Dict[str, Any]:
            nodes = nodes_for(values)
            count = max(1, len(nodes))
            cards = self._working_candidate_cards(conn, hive_id, working, values)
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
