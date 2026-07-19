"""Application boundary for the V2.7 graph dialogue service."""

from __future__ import annotations

from typing import Any, Optional

from server.core.exceptions import NotFoundError, ValidationError
from server.modules.hive.infrastructure.repository import HiveRepository
from server.v2.hive import V2HiveService


class HiveService:
    def __init__(
        self,
        repository: Optional[HiveRepository] = None,
        local_memory_config: Any = None,
    ) -> None:
        self.facade = V2HiveService(
            repository or HiveRepository(),
            local_memory_config,
        )

    def create(
        self,
        max_cells: int = 24,
        conversation_id: str = "",
    ) -> dict[str, Any]:
        return self.facade.create(max_cells, conversation_id)

    def get_hive(self, hive_id: str) -> dict[str, Any]:
        try:
            return self.facade.get_hive(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def delete(self, hive_id: str) -> dict[str, Any]:
        try:
            return self.facade.delete(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def preview(self, hive_id: str, text: str) -> dict[str, Any]:
        try:
            return self.facade.preview(hive_id, text)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def query(
        self,
        hive_id: str,
        text: str,
        resolved_mode: Optional[str] = None,
        resonance_scope: str = "LOCAL_ONLY",
    ) -> dict[str, Any]:
        try:
            return self.facade.query(
                hive_id,
                text,
                resolved_mode,
                resonance_scope,
            )
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error
        except ValueError as error:
            raise ValidationError(str(error)) from error

    def parse_query(self, text: str) -> dict[str, Any]:
        return self.facade.parse_query(text)

    def activate_query(
        self,
        hive_id: str,
        text: str,
        resolved_mode: str = "NEW_QUERY",
    ) -> dict[str, Any]:
        try:
            return self.facade.activate_query(hive_id, text, resolved_mode)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def query_working_state(self, hive_id: str) -> dict[str, Any]:
        return self.get_hive(hive_id)

    def vibration_step(
        self,
        hive_id: str,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self.facade.vibration_step(hive_id, config)

    def vibration_run(
        self,
        hive_id: str,
        steps: int = 3,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self.facade.vibration_run(hive_id, steps, config)

    def vibration_stop(self, hive_id: str) -> dict[str, Any]:
        return self.facade.vibration_stop(hive_id)

    def dynamics_state(
        self,
        hive_id: str,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self.facade.dynamics_state(hive_id, config)

    def dynamics_step(
        self,
        hive_id: str,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self.facade.dynamics_step(hive_id, config)

    def dynamics_history(self, hive_id: str) -> list[dict[str, Any]]:
        return self.facade.dynamics_history(hive_id)

    def dynamics_reset(self, hive_id: str) -> dict[str, Any]:
        return self.facade.dynamics_reset(hive_id)

    def dynamics_node(self, hive_id: str, node_id: str) -> dict[str, Any]:
        return self.facade.dynamics_node(hive_id, node_id)

    def dynamics_evictions(self, hive_id: str) -> list[dict[str, Any]]:
        return self.facade.dynamics_evictions(hive_id)

    def snapshot(self, hive_id: str, **options: Any) -> dict[str, Any]:
        return self.facade.snapshot(hive_id, **options)

    def export(
        self,
        hive_id: str,
        mode: str = "current",
        run_id: Optional[str] = None,
        step: Optional[int] = None,
        detail: str = "full",
    ) -> dict[str, Any]:
        return self.facade.export(hive_id, mode, run_id, step, detail)

    def analytics(
        self,
        hive_id: str,
        run_id: Optional[str] = None,
        compare_run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return self.facade.analytics(hive_id, run_id, compare_run_id)

    def events(self, hive_id: str) -> list[dict[str, Any]]:
        return self.facade.events(hive_id)

    def decisions(self, hive_id: str) -> list[dict[str, Any]]:
        return self.facade.decisions(hive_id)

    def matches(self, hive_id: str, node_id: str) -> list[dict[str, Any]]:
        return self.facade.matches(hive_id, node_id)

    def forage(self, query: str, max_cells: int = 24) -> dict[str, Any]:
        return self.facade.forage(query, max_cells)
