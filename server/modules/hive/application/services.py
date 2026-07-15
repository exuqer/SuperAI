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

    def query(
        self,
        hive_id: str,
        text: str,
        resolved_mode: str | None = None,
        resonance_scope: str = "LOCAL_ONLY",
    ) -> dict[str, Any]:
        try:
            return self.facade.query(hive_id, text, resolved_mode, resonance_scope)
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

    def parse_query(self, text: str) -> dict[str, Any]:
        return self.facade.parse_query(text)

    def classify_resonance(self, text: str) -> dict[str, str]:
        return self.facade.classify_resonance(text)

    def lexical_candidates(self, hive_id: str, text: str, use_global_memory: bool = True) -> dict[str, Any]:
        return self.facade.lexical_candidates(hive_id, text, use_global_memory)

    def resonance_create(self, hive_id: str, text: str, scope: str = "LOCAL_THEN_GLOBAL", **options: Any) -> dict[str, Any]:
        try:
            return self.facade.resonance_create(hive_id, text, scope, **options)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def resonance_step(self, hive_id: str, probe_id: str) -> dict[str, Any]:
        return self.facade.resonance_step(hive_id, probe_id)

    def resonance_run(self, hive_id: str, probe_id: str) -> dict[str, Any]:
        return self.facade.resonance_run(hive_id, probe_id)

    def resonance_get(self, hive_id: str, probe_id: str) -> dict[str, Any]:
        return self.facade.resonance_get(hive_id, probe_id)

    def resonance_import(self, hive_id: str, probe_id: str, match_id: str, include_scenes: bool = False) -> dict[str, Any]:
        return self.facade.resonance_import(hive_id, probe_id, match_id, include_scenes)

    def resonance_related_scenes(self, hive_id: str, probe_id: str, match_id: str = "") -> dict[str, Any]:
        return self.facade.resonance_related_scenes(hive_id, probe_id, match_id)

    def resonance_stop(self, session_id: str) -> dict[str, Any]:
        return self.facade.resonance_stop(session_id)

    def resonance_snapshots(self, session_id: str) -> list[dict[str, Any]]:
        return self.facade.resonance_snapshots(session_id)

    def import_resonance_concept(self, session_id: str, concept_id: str) -> dict[str, Any]:
        return self.facade.import_resonance_concept(session_id, concept_id)

    def activate_query(self, hive_id: str, text: str, resolved_mode: str = "NEW_QUERY") -> dict[str, Any]:
        try:
            return self.facade.activate_query(hive_id, text, resolved_mode)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def vibration_step(self, hive_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return self.facade.vibration_step(hive_id, config)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def vibration_run(self, hive_id: str, steps: int = 3, config: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return self.facade.vibration_run(hive_id, steps, config)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def vibration_stop(self, hive_id: str) -> dict[str, Any]:
        try:
            return self.facade.vibration_stop(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def dynamics_state(self, hive_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return self.facade.dynamics_state(hive_id, config)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def dynamics_step(self, hive_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return self.facade.dynamics_step(hive_id, config)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def dynamics_history(self, hive_id: str) -> list[dict[str, Any]]:
        try:
            return self.facade.dynamics_history(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def dynamics_reset(self, hive_id: str) -> dict[str, Any]:
        try:
            return self.facade.dynamics_reset(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def dynamics_node(self, hive_id: str, cell_id: str) -> dict[str, Any]:
        try:
            return self.facade.dynamics_node(hive_id, cell_id)
        except KeyError as error:
            raise NotFoundError("dynamics node", cell_id) from error

    def dynamics_evictions(self, hive_id: str) -> list[dict[str, Any]]:
        try:
            return self.facade.dynamics_evictions(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def query_working_state(self, hive_id: str) -> dict[str, Any]:
        try:
            return self.facade.query_working_state(hive_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def unknown_search_start(self, hive_id: str, surface: str, token_index: int, query_role: str = "", query_scene_id: str = "") -> dict[str, Any]:
        try:
            return self.facade.unknown_search_start(hive_id, surface, token_index, query_role, query_scene_id)
        except KeyError as error:
            raise NotFoundError("hive", hive_id) from error

    def unknown_search_step(self, hive_id: str, search_id: str) -> dict[str, Any]:
        return self.facade.unknown_search_step(hive_id, search_id)

    def unknown_search_run(self, hive_id: str, search_id: str) -> dict[str, Any]:
        return self.facade.unknown_search_run(hive_id, search_id)

    def unknown_search_vibrate(self, hive_id: str, search_id: str) -> dict[str, Any]:
        return self.facade.unknown_search_vibrate(hive_id, search_id)

    def unknown_search_get(self, hive_id: str, search_id: str) -> dict[str, Any]:
        return self.facade.unknown_search_get(hive_id, search_id)

    def unknown_search_evidence(self, hive_id: str, search_id: str) -> list[dict[str, Any]]:
        return self.facade.unknown_search_evidence(hive_id, search_id)

    def unknown_search_routes(self, hive_id: str, search_id: str) -> list[dict[str, Any]]:
        return self.facade.unknown_search_routes(hive_id, search_id)

    def unknown_search_confirm(self, hive_id: str, search_id: str) -> dict[str, Any]:
        return self.facade.unknown_search_confirm(hive_id, search_id)

    def view(self, hive_id: str, view_id: int | None = None) -> dict[str, Any]:
        try:
            return self.morphology.view(hive_id, view_id)
        except KeyError as error:
            raise NotFoundError("hive projection", str(error)) from error

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
