"""Stable JSON DTOs for hive inspection and reasoning traces."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .repository import V2Repository, decode


class HiveExportService:
    def __init__(self, repository: Optional[V2Repository] = None) -> None:
        self.repository = repository or V2Repository()

    @staticmethod
    def _json_row(row: Any) -> Dict[str, Any]:
        item = dict(row)
        for key in list(item):
            if key.endswith("_json"):
                item[key[:-5]] = decode(item.pop(key), {})
        return item

    def current(self, hive_id: str, detail: str = "full") -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            hive = conn.execute("SELECT * FROM hives WHERE id=?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            space = conn.execute("SELECT * FROM spaces WHERE id=?", (hive["space_id"],)).fetchone()
            nodes = [
                self._json_row(row)
                for row in conn.execute(
                    "SELECT * FROM hive_node_states WHERE hive_id=? ORDER BY placement_id",
                    (hive_id,),
                )
            ]
            cells = [
                self._json_row(row)
                for row in conn.execute(
                    "SELECT * FROM hive_cells WHERE hive_id=? ORDER BY id", (hive_id,)
                )
            ]
            if not nodes:
                nodes = [
                    {
                        "hive_id": hive_id,
                        "placement_id": cell["hive_placement_id"],
                        "cloud_id": cell["dominant_cloud_id"],
                        "x": cell["x"],
                        "y": cell["y"],
                        "local_activation": cell["local_activation"],
                        "local_gravity": cell["local_gravity"],
                        "stored_strength": cell["stored_strength"],
                        "retention": cell["retention"],
                        "energy": cell["local_activation"],
                        "eviction_status": "ACTIVE",
                    }
                    for cell in [
                        dict(row)
                        for row in conn.execute(
                            "SELECT hc.*, p.x, p.y, p.local_gravity FROM hive_cells hc JOIN cloud_placements p ON p.id=hc.hive_placement_id WHERE hc.hive_id=?",
                            (hive_id,),
                        )
                    ]
                ]
            components = [
                self._json_row(row)
                for row in conn.execute(
                    "SELECT hcc.* FROM hive_cell_components hcc JOIN hive_cells hc ON hc.id=hcc.cell_id WHERE hc.hive_id=? ORDER BY hcc.cell_id, hcc.id",
                    (hive_id,),
                )
            ]
            subspaces = [
                self._json_row(row) for row in conn.execute(
                    "SELECT * FROM hive_subspaces WHERE hive_id=? ORDER BY depth, id", (hive_id,)
                )
            ]
            generation_candidates = [
                self._json_row(row) for row in conn.execute(
                    "SELECT * FROM hive_generation_candidates WHERE hive_id=? ORDER BY score_total DESC, id", (hive_id,)
                )
            ]
            payload = {
                "schema_version": 2,
                "export_type": "current",
                "hive": self._json_row(hive),
                "space": self._json_row(space),
                "nodes": nodes,
                "cells": cells,
                "components": components,
                "subspaces": subspaces,
                "generation_candidates": generation_candidates,
                "sentence_plan": decode(hive["query_json"], {}).get("sentence_plan"),
                "selected_surface": decode(hive["metadata_json"], {}).get("selected_surface"),
                "reverse_validation": decode(hive["metadata_json"], {}).get("reverse_validation"),
                "morphology_trace": decode(hive["metadata_json"], {}).get("morphology_trace", []),
            }
            if detail != "compact":
                payload["stats"] = {
                    "nodes": len(nodes),
                    "cells": len(cells),
                    "components": len(components),
                    "total_energy": sum(float(node.get("energy", 0)) for node in nodes),
                }
            return payload

    def snapshot(self, run_id: str, step: Optional[int] = None) -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            query = "SELECT * FROM hive_reasoning_snapshots WHERE run_id=?"
            args = [run_id]
            if step is not None:
                query += " AND step=? AND phase=?"
                args.append(step)
                args.append("INITIAL" if step == 0 else "AFTER_SETTLE")
            query += " ORDER BY step DESC LIMIT 1"
            row = conn.execute(query, args).fetchone()
            if not row:
                raise KeyError(run_id)
            item = self._json_row(row)
            return {"schema_version": 2, "export_type": "snapshot", **item}

    def trace(self, run_id: str, detail: str = "full") -> Dict[str, Any]:
        with self.repository.transaction() as conn:
            run = conn.execute("SELECT * FROM hive_reasoning_runs WHERE id=?", (run_id,)).fetchone()
            if not run:
                raise KeyError(run_id)
            snapshots = [
                self._json_row(row)
                for row in conn.execute(
                    "SELECT * FROM hive_reasoning_snapshots WHERE run_id=? ORDER BY step, id",
                    (run_id,),
                )
            ]
            events = [
                self._json_row(row)
                for row in conn.execute(
                    "SELECT * FROM hive_reasoning_events WHERE run_id=? ORDER BY step, created_at",
                    (run_id,),
                )
            ]
            clusters = [
                self._json_row(row)
                for row in conn.execute(
                    "SELECT * FROM hive_resonance_clusters WHERE run_id=? ORDER BY reasoning_step, id",
                    (run_id,),
                )
            ]
            result = {
                "schema_version": 2,
                "export_type": "trace",
                "run": self._json_row(run),
                "snapshots": snapshots,
                "events": events,
                "clusters": clusters,
            }
            if detail == "compact":
                result["snapshots"] = [
                    {"step": item["step"], "phase": item["phase"], "state_hash": item["state_hash"]}
                    for item in snapshots
                ]
            return result

    def diff(self, run_id: str, from_step: int, to_step: int) -> Dict[str, Any]:
        before, after = self.snapshot(run_id, from_step), self.snapshot(run_id, to_step)
        left = {int(node["placement_id"]): node for node in before["state"]["nodes"]}
        right = {int(node["placement_id"]): node for node in after["state"]["nodes"]}
        activated, weakened, moved, evicted = [], [], [], []
        for placement_id in sorted(set(left) | set(right)):
            old, new = left.get(placement_id), right.get(placement_id)
            if not new:
                continue
            if old and new["local_activation"] > old["local_activation"] + 0.001:
                activated.append(placement_id)
            if old and new["retention"] < old["retention"] - 0.001:
                weakened.append(placement_id)
            if old and abs(new["x"] - old["x"]) + abs(new["y"] - old["y"]) > 0.001:
                moved.append(placement_id)
            if new.get("eviction_status") == "EVICTED":
                evicted.append(placement_id)
        return {
            "schema_version": 2,
            "from_step": from_step,
            "to_step": to_step,
            "nodes": {
                "activated": activated,
                "weakened": weakened,
                "moved": moved,
                "evicted": evicted,
            },
        }
