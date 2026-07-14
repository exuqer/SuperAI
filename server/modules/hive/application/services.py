"""Hive application service."""

from __future__ import annotations

from typing import Any

from server.core.exceptions import NotFoundError, ValidationError
from server.modules.hive.infrastructure.repository import HiveRepository
from server.v2.hive import V2HiveService
from server.v2.local_memory import HiveLocalMemoryConfig
from server.v2.morphology import MorphologyService
from server.v2.repository import decode


class HiveService:
    def __init__(
        self,
        repository: HiveRepository | None = None,
        local_memory_config: HiveLocalMemoryConfig | None = None,
    ) -> None:
        self.facade = V2HiveService(repository or HiveRepository(), local_memory_config)
        self.morphology = MorphologyService(self.facade.service.repository)

    def create(self, max_cells: int = 24, conversation_id: str = "") -> dict[str, Any]:
        return self.facade.create(max_cells, conversation_id)

    def get_hive(self, hive_id: str) -> dict[str, Any]:
        try:
            return self.facade.get_hive(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def preview(self, hive_id: str, text: str) -> dict[str, Any]:
        try:
            return self.facade.preview(hive_id, text)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def query(self, hive_id: str, text: str) -> dict[str, Any]:
        try:
            return self.facade.query(hive_id, text)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def forage(self, query: str, max_cells: int = 24) -> dict[str, Any]:
        return self.facade.forage(query, max_cells)

    def events(self, hive_id: str) -> list[dict[str, Any]]:
        try:
            return self.facade.service.events(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def decisions(self, hive_id: str) -> list[dict[str, Any]]:
        try:
            return self.facade.service.decisions(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def matches(self, hive_id: str, cell_id: str) -> list[dict[str, Any]]:
        return self.facade.service.matches(hive_id, cell_id)

    def reason(
        self, hive_id: str, text: str = "", config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            return self.facade.reason(hive_id, text, config)
        except KeyError as error:
            raise NotFoundError("hive or query", hive_id) from error
        except (TypeError, ValueError) as error:
            raise ValidationError(str(error)) from error

    def export(
        self,
        hive_id: str,
        mode: str = "current",
        run_id: str | None = None,
        step: int | None = None,
        detail: str = "full",
    ) -> dict[str, Any]:
        try:
            return self.facade.export(hive_id, mode, run_id, step, detail)
        except KeyError as error:
            raise NotFoundError("hive or reasoning run", hive_id) from error
        except ValueError as error:
            raise ValidationError(str(error)) from error

    def diff(self, run_id: str, from_step: int, to_step: int) -> dict[str, Any]:
        try:
            return self.facade.diff(run_id, from_step, to_step)
        except KeyError as error:
            raise NotFoundError("reasoning run", run_id) from error

    def restore(self, hive_id: str, run_id: str, step: int) -> dict[str, Any]:
        try:
            return self.facade.restore(hive_id, run_id, step)
        except KeyError as error:
            raise NotFoundError("reasoning snapshot", run_id) from error

    def runs(self, hive_id: str) -> list[dict[str, Any]]:
        try:
            return self.facade.runs(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def snapshots(self, hive_id: str, run_id: str) -> list[dict[str, Any]]:
        try:
            return self.facade.snapshots(hive_id, run_id)
        except KeyError as error:
            raise NotFoundError("reasoning run", run_id) from error

    def analytics(
        self, hive_id: str, run_id: str | None = None, compare_run_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return self.facade.analytics(hive_id, run_id, compare_run_id)
        except KeyError as error:
            raise NotFoundError("hive or reasoning run", str(error)) from error

    def hierarchy(self, hive_id: str) -> dict[str, Any]:
        try:
            return self.morphology.hierarchy(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def expand(self, hive_id: str, cell_id: str, target_level: str, reason: str, max_candidates: int) -> dict[str, Any]:
        try:
            return self.morphology.expand_cell(hive_id, cell_id, target_level, reason, max_candidates)
        except KeyError as error:
            raise NotFoundError("hive cell", cell_id) from error
        except ValueError as error:
            raise ValidationError(str(error)) from error

    def collapse(self, hive_id: str, subspace_id: int) -> dict[str, Any]:
        try:
            return self.morphology.collapse(hive_id, subspace_id)
        except KeyError as error:
            raise NotFoundError("hive subspace", str(subspace_id)) from error

    def candidates(self, hive_id: str, candidate_id: int | None = None) -> list[dict[str, Any]]:
        try:
            return self.morphology.candidates(hive_id, candidate_id)
        except KeyError as error:
            raise NotFoundError("generation candidate", str(candidate_id)) from error

    def select_candidate(self, hive_id: str, candidate_id: int) -> dict[str, Any]:
        try:
            return self.morphology.select_candidate(hive_id, candidate_id)
        except KeyError as error:
            raise NotFoundError("generation candidate", str(candidate_id)) from error

    def generate(self, hive_id: str, sentence_plan: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.morphology.generate_sentence(hive_id, sentence_plan)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error
        except ValueError as error:
            raise ValidationError(str(error)) from error

    def validate_surface(self, hive_id: str, surface: str) -> dict[str, Any]:
        if not surface.strip():
            raise ValidationError("surface must not be empty")
        hierarchy = self.morphology.hierarchy(hive_id)
        hive = hierarchy["hive"]
        metadata = decode(hive.get("metadata_json"), {})
        sentence_plan = decode(hive.get("query_json"), {}).get("sentence_plan")
        expected = metadata.get("selected_surface", "")
        normalized_surface = surface.strip().casefold().rstrip(".?!")
        normalized_expected = str(expected).strip().casefold().rstrip(".?!")
        valid = bool(sentence_plan and normalized_expected and normalized_surface == normalized_expected)
        return {
            "surface": surface,
            "valid": valid,
            "score": 1.0 if valid else 0.0,
            "sentence_plan": sentence_plan,
            "reason": "surface retained its role plan" if valid else "surface does not match the generated role plan",
        }
