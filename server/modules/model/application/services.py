"""Application services for the canonical model."""

from __future__ import annotations

from typing import Any

from server.core.exceptions import ConflictError, NotFoundError
from server.modules.model.infrastructure.repository import ModelRepository
from server.v2.physics import PlacementPhysicsV2
from server.v2.validation import ModelInvariantValidator


class ModelService:
    def __init__(self, repository: ModelRepository | None = None) -> None:
        self.repository = repository or ModelRepository()

    def get_field(self) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            space, _ = self.repository.get_or_create_space(conn, "global_field", seed=1337)
        return self.repository.normalized_space(int(space["id"]))

    def get_stats(self) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            return self.repository.stats(conn)

    def clear_model(self) -> dict[str, Any]:
        self.repository.clear()
        with self.repository.transaction() as conn:
            return {"success": True, "stats": self.repository.stats(conn)}

    def get_trained_model_snapshot(self) -> dict[str, Any]:
        return self.repository.trained_model_snapshot()

    def get_cloud(self, cloud_id: int) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            cloud = conn.execute("SELECT * FROM clouds WHERE id = ?", (cloud_id,)).fetchone()
            if not cloud:
                raise NotFoundError("cloud", cloud_id)
            owned_spaces = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM spaces WHERE owner_cloud_id=? ORDER BY id", (cloud_id,)
                )
            ]
            return {"cloud": dict(cloud), "owned_spaces": owned_spaces}

    def get_structure(self, cloud_id: int) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            cloud = conn.execute("SELECT * FROM clouds WHERE id = ?", (cloud_id,)).fetchone()
            if not cloud:
                raise NotFoundError("cloud", cloud_id)
            space = conn.execute(
                "SELECT * FROM spaces WHERE owner_cloud_id = ? AND space_type = 'word_structure_space'",
                (cloud_id,),
            ).fetchone()
            components = [
                dict(row)
                for row in conn.execute(
                    """SELECT sc.* FROM structural_components sc
                    WHERE sc.parent_cloud_id = ? ORDER BY sc.component_index""",
                    (cloud_id,),
                )
            ]
            child_ids = sorted({int(item["child_cloud_id"]) for item in components})
            children: dict[str, Any] = {}
            if child_ids:
                marks = ",".join("?" for _ in child_ids)
                children = {
                    str(row["id"]): dict(row)
                    for row in conn.execute(
                        f"SELECT * FROM clouds WHERE id IN ({marks})", child_ids
                    )
                }
            return {
                "cloud": dict(cloud),
                "structure_space": dict(space) if space else None,
                "components": components,
                "clouds": children,
            }

    def get_placement(self, placement_id: int) -> dict[str, Any]:
        placement = self.repository.get_placement(placement_id)
        if not placement:
            raise NotFoundError("placement", placement_id)
        return {
            "placement": placement,
            "cloud": self.repository.get_cloud(int(placement["cloud_id"])),
        }

    def get_space(self, space_id: int) -> dict[str, Any]:
        try:
            return self.repository.normalized_space(space_id)
        except KeyError as error:
            raise NotFoundError("space", space_id) from error

    def physics_tick(self, space_id: int) -> dict[str, Any]:
        try:
            return {"space_id": space_id, "updates": PlacementPhysicsV2(space_id).tick()}
        except KeyError as error:
            raise NotFoundError("space", space_id) from error
        except ValueError as error:
            raise ConflictError(str(error)) from error

    def get_scene(self, scene_id: int) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            scene = conn.execute(
                "SELECT * FROM scenes WHERE cloud_id = ?", (scene_id,)
            ).fetchone()
            if not scene:
                raise NotFoundError("scene", scene_id)
            components = [
                {
                    "id": row["id"],
                    "placement_id": row["placement_id"],
                    "cloud_id": row["word_form_cloud_id"],
                    "lexeme_cloud_id": row["lexeme_cloud_id"],
                    "token_index": row["token_index"],
                    "grammatical_role": row["grammatical_role"],
                    "dependency_role": row["dependency_role"],
                    "head_component_id": row["head_component_id"],
                    "confidence": row["confidence"],
                    "morphology_json": row["morphology_json"],
                }
                for row in conn.execute(
                    "SELECT * FROM scene_components WHERE scene_cloud_id = ? ORDER BY token_index",
                    (scene_id,),
                )
            ]
            scene_dto = dict(scene)
            scene_dto["components"] = components
            return {"scene": scene_dto}

    def debug_invariants(self) -> dict[str, Any]:
        return ModelInvariantValidator(self.repository).validate()
