"""Public V2 hive facade."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .local_memory import V2LocalMemoryService, HiveLocalMemoryConfig
from .repository import V2Repository
from .vibration import HiveVibrationEngine, QueryActivation, VibrationConfig
from .export import HiveExportService
from .analytics import HiveAnalyticsService


class V2HiveService:
    def __init__(self, repository: Optional[V2Repository] = None, config: Optional[HiveLocalMemoryConfig] = None) -> None:
        self.service = V2LocalMemoryService(repository, config)

    def create(self, max_cells: int = 24, conversation_id: str = "") -> Dict[str, Any]:
        return self.service.create_hive(max_cells, conversation_id)

    def preview(self, hive_id: str, text: str) -> Dict[str, Any]:
        return self.service.preview(hive_id, text)

    def query(self, hive_id: str, text: str) -> Dict[str, Any]:
        return self.service.query(hive_id, text)

    def forage(self, query: str, max_cells: int = 24) -> Dict[str, Any]:
        """Compatibility helper: create a hive and process its first query."""
        hive = self.create(max_cells)
        return self.query(hive["hive"]["id"], query)

    def get_hive(self, hive_id: str) -> Dict[str, Any]:
        return self.service.get_hive(hive_id)

    def reason(self, hive_id: str, text: str = "", config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.service.repository.transaction() as conn:
            hive = conn.execute("SELECT query_json FROM hives WHERE id=?", (hive_id,)).fetchone()
            if not hive:
                raise KeyError(hive_id)
            parsed = self.service.parser.parse(text, conn) if text else json.loads(hive["query_json"] or "{}")
        components = [
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in parsed.get("components", [])
        ]
        activation = QueryActivation(
            tuple(sorted({int(item.get("word_form_cloud_id")) for item in components if item.get("word_form_cloud_id") is not None})),
            tuple(item.get("normalized_form", "") for item in components),
            tuple(item.get("expected_role", "") for item in components),
        )
        result = HiveVibrationEngine(self.service.repository).reason(hive_id, activation, VibrationConfig(**(config or {})))
        return result.to_dict() | {"hive": self.get_hive(hive_id)}

    def export(self, hive_id: str, mode: str = "current", run_id: Optional[str] = None, step: Optional[int] = None, detail: str = "full") -> Dict[str, Any]:
        exporter = HiveExportService(self.service.repository)
        if mode == "current":
            return exporter.current(hive_id, detail)
        if mode == "snapshot":
            if not run_id:
                raise KeyError("run_id")
            return exporter.snapshot(run_id, step)
        if mode == "trace":
            if not run_id:
                raise KeyError("run_id")
            return exporter.trace(run_id, detail)
        if mode == "initial":
            if not run_id:
                raise KeyError("run_id")
            return exporter.snapshot(run_id, 0)
        raise ValueError("unsupported export mode")

    def diff(self, run_id: str, from_step: int, to_step: int) -> Dict[str, Any]:
        return HiveExportService(self.service.repository).diff(run_id, from_step, to_step)

    def restore(self, hive_id: str, run_id: str, step: int) -> Dict[str, Any]:
        restored = HiveVibrationEngine(self.service.repository).restore(hive_id, run_id, step)
        restored["hive"] = self.get_hive(hive_id)
        return restored

    def runs(self, hive_id: str) -> List[Dict[str, Any]]:
        with self.service.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM hives WHERE id=?", (hive_id,)).fetchone():
                raise KeyError(hive_id)
            return [dict(row) for row in conn.execute("SELECT * FROM hive_reasoning_runs WHERE hive_id=? ORDER BY created_at DESC", (hive_id,))]

    def snapshots(self, hive_id: str, run_id: str) -> List[Dict[str, Any]]:
        with self.service.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM hives WHERE id=? AND id IN (SELECT hive_id FROM hive_reasoning_runs WHERE id=?)", (hive_id, run_id)).fetchone():
                raise KeyError(run_id)
            return [dict(row) for row in conn.execute("SELECT id, run_id, hive_id, step, phase, state_hash, delta_json, clusters_json, events_json, created_at FROM hive_reasoning_snapshots WHERE run_id=? ORDER BY step, id", (run_id,))]

    def analytics(
        self, hive_id: str, run_id: Optional[str] = None, compare_run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return HiveAnalyticsService(self.service.repository).get(hive_id, run_id, compare_run_id)
