"""Cloud repository - data access layer for clouds and related entities."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from server.models.cloud import Cloud, CloudPlacement, StructuralComponent
from server.models.space import Space, Layer


DB_PATH = Path(".superai/state.sqlite")


def get_db_path() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def get_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class CloudRepository:
    """Repository for Cloud entities."""
    
    def create(self, cloud: Cloud) -> Cloud:
        with get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO clouds 
                (layer_id, cloud_type, canonical_name, mass, density, radius, 
                 stability, activation, observation_count, created_at, updated_at, 
                 last_activated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cloud.layer_id, cloud.cloud_type, cloud.canonical_name, cloud.mass,
                 cloud.density, cloud.radius, cloud.stability, cloud.activation,
                 cloud.observation_count, cloud.created_at, cloud.updated_at,
                 cloud.last_activated_at, cloud.metadata_json)
            )
            cloud.id = cursor.lastrowid
            conn.commit()
        return cloud
    
    def get_by_id(self, cloud_id: int) -> Optional[Cloud]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM clouds WHERE id = ?", (cloud_id,)).fetchone()
        return self._row_to_cloud(row) if row else None
    
    def get_by_canonical_name(self, layer_id: int, canonical_name: str) -> Optional[Cloud]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM clouds WHERE layer_id = ? AND canonical_name = ?",
                (layer_id, canonical_name)
            ).fetchone()
        return self._row_to_cloud(row) if row else None
    
    def get_by_layer(self, layer_id: int, limit: int = 1000) -> List[Cloud]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM clouds WHERE layer_id = ? ORDER BY mass DESC LIMIT ?",
                (layer_id, limit)
            ).fetchall()
        return [self._row_to_cloud(row) for row in rows]
    
    def get_recently_activated(self, layer_id: int, since: datetime, limit: int = 100) -> List[Cloud]:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM clouds 
                WHERE layer_id = ? AND last_activated_at >= ?
                ORDER BY last_activated_at DESC LIMIT ?""",
                (layer_id, since.isoformat(), limit)
            ).fetchall()
        return [self._row_to_cloud(row) for row in rows]
    
    def update(self, cloud: Cloud) -> Cloud:
        cloud.updated_at = datetime.utcnow()
        with get_connection() as conn:
            conn.execute(
                """UPDATE clouds SET
                layer_id = ?, cloud_type = ?, canonical_name = ?, mass = ?,
                density = ?, radius = ?, stability = ?, activation = ?,
                observation_count = ?, updated_at = ?, last_activated_at = ?,
                metadata_json = ?
                WHERE id = ?""",
                (cloud.layer_id, cloud.cloud_type, cloud.canonical_name, cloud.mass,
                 cloud.density, cloud.radius, cloud.stability, cloud.activation,
                 cloud.observation_count, cloud.updated_at, cloud.last_activated_at,
                 cloud.metadata_json, cloud.id)
            )
            conn.commit()
        return cloud
    
    def increment_observation(self, cloud_id: int, mass_delta: float = 0.1, 
                               stability_delta: float = 0.01) -> None:
        with get_connection() as conn:
            conn.execute(
                """UPDATE clouds SET
                observation_count = observation_count + 1,
                mass = mass + ?,
                stability = MIN(1.0, stability + ?),
                updated_at = ?,
                last_activated_at = ?
                WHERE id = ?""",
                (mass_delta, stability_delta, datetime.utcnow(), datetime.utcnow(), cloud_id)
            )
            conn.commit()
    
    def _row_to_cloud(self, row: sqlite3.Row) -> Cloud:
        return Cloud(
            id=row["id"],
            layer_id=row["layer_id"],
            cloud_type=row["cloud_type"],
            canonical_name=row["canonical_name"],
            mass=row["mass"],
            density=row["density"],
            radius=row["radius"],
            stability=row["stability"],
            activation=row["activation"],
            observation_count=row["observation_count"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            last_activated_at=datetime.fromisoformat(row["last_activated_at"]) if row["last_activated_at"] else None,
            metadata_json=row["metadata_json"] or "{}",
        )


class CloudPlacementRepository:
    """Repository for CloudPlacement entities."""
    
    def create(self, placement: CloudPlacement) -> CloudPlacement:
        with get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO cloud_placements
                (space_id, cloud_id, x, y, z, radius, density, mass, activation,
                 velocity_x, velocity_y, velocity_z, fixed, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (placement.space_id, placement.cloud_id, placement.x, placement.y,
                 placement.z, placement.radius, placement.density, placement.mass,
                 placement.activation, placement.velocity_x, placement.velocity_y,
                 placement.velocity_z, placement.fixed, placement.created_at,
                 placement.updated_at)
            )
            placement.id = cursor.lastrowid
            conn.commit()
        return placement
    
    def get_by_id(self, placement_id: int) -> Optional[CloudPlacement]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM cloud_placements WHERE id = ?", (placement_id,)).fetchone()
        return self._row_to_placement(row) if row else None
    
    def get_by_space(self, space_id: int) -> List[CloudPlacement]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM cloud_placements WHERE space_id = ?", (space_id,)
            ).fetchall()
        return [self._row_to_placement(row) for row in rows]
    
    def get_by_cloud(self, cloud_id: int) -> List[CloudPlacement]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM cloud_placements WHERE cloud_id = ?", (cloud_id,)
            ).fetchall()
        return [self._row_to_placement(row) for row in rows]
    
    def get_in_viewport(self, space_id: int, min_x: float, min_y: float, 
                        max_x: float, max_y: float, min_density: float = 0.0) -> List[CloudPlacement]:
        """Get placements visible in viewport with density filter."""
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM cloud_placements 
                WHERE space_id = ? AND density >= ?
                AND x + radius >= ? AND x - radius <= ?
                AND y + radius >= ? AND y - radius <= ?""",
                (space_id, min_density, min_x, max_x, min_y, max_y)
            ).fetchall()
        return [self._row_to_placement(row) for row in rows]
    
    def update(self, placement: CloudPlacement) -> CloudPlacement:
        placement.updated_at = datetime.utcnow()
        with get_connection() as conn:
            conn.execute(
                """UPDATE cloud_placements SET
                x = ?, y = ?, z = ?, radius = ?, density = ?, mass = ?,
                activation = ?, velocity_x = ?, velocity_y = ?, velocity_z = ?,
                fixed = ?, updated_at = ?
                WHERE id = ?""",
                (placement.x, placement.y, placement.z, placement.radius,
                 placement.density, placement.mass, placement.activation,
                 placement.velocity_x, placement.velocity_y, placement.velocity_z,
                 placement.fixed, placement.updated_at, placement.id)
            )
            conn.commit()
        return placement
    
    def get_structural_space(self, host_cloud_id: int) -> Optional[Space]:
        """Get structural space for a host cloud."""
        from server.repositories.cloud_repository import SpaceRepository
        space_repo = SpaceRepository()
        return space_repo.get_structural_space(host_cloud_id)
    
    def get_semantic_space(self, host_cloud_id: int) -> Optional[Space]:
        """Get semantic space for a host cloud."""
        from server.repositories.cloud_repository import SpaceRepository
        space_repo = SpaceRepository()
        return space_repo.get_semantic_space(host_cloud_id)
    
    def update_positions_batch(self, placements: List[CloudPlacement]) -> None:
        """Batch update positions for physics simulation."""
        if not placements:
            return
        with get_connection() as conn:
            for p in placements:
                conn.execute(
                    """UPDATE cloud_placements SET
                    x = ?, y = ?, z = ?, velocity_x = ?, velocity_y = ?, velocity_z = ?,
                    updated_at = ?
                    WHERE id = ?""",
                    (p.x, p.y, getattr(p, "z", 0.0),
                     getattr(p, "velocity_x", 0.0), getattr(p, "velocity_y", 0.0),
                     getattr(p, "velocity_z", 0.0),
                     datetime.utcnow(), p.id)
                )
            conn.commit()
    
    def _row_to_placement(self, row: sqlite3.Row) -> CloudPlacement:
        return CloudPlacement(
            id=row["id"],
            space_id=row["space_id"],
            cloud_id=row["cloud_id"],
            x=row["x"],
            y=row["y"],
            z=row["z"],
            radius=row["radius"],
            density=row["density"],
            mass=row["mass"],
            activation=row["activation"],
            velocity_x=row["velocity_x"],
            velocity_y=row["velocity_y"],
            velocity_z=row["velocity_z"],
            fixed=bool(row["fixed"]),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )


class StructuralComponentRepository:
    """Repository for StructuralComponent entities."""
    
    def create(self, component: StructuralComponent) -> StructuralComponent:
        with get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO structural_components
                (parent_cloud_id, child_cloud_id, child_placement_id, position_index,
                 phase, weight, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (component.parent_cloud_id, component.child_cloud_id, component.child_placement_id,
                 component.position_index, component.phase, component.weight,
                 component.role, component.created_at)
            )
            component.id = cursor.lastrowid
            conn.commit()
        return component
    
    def get_children(self, parent_cloud_id: int) -> List[StructuralComponent]:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM structural_components 
                WHERE parent_cloud_id = ? ORDER BY position_index""",
                (parent_cloud_id,)
            ).fetchall()
        return [self._row_to_component(row) for row in rows]
    
    def get_by_parent_and_child(self, parent_cloud_id: int, child_cloud_id: int) -> Optional[StructuralComponent]:
        with get_connection() as conn:
            row = conn.execute(
                """SELECT * FROM structural_components 
                WHERE parent_cloud_id = ? AND child_cloud_id = ?""",
                (parent_cloud_id, child_cloud_id)
            ).fetchone()
        return self._row_to_component(row) if row else None
    
    def _row_to_component(self, row: sqlite3.Row) -> StructuralComponent:
        return StructuralComponent(
            id=row["id"],
            parent_cloud_id=row["parent_cloud_id"],
            child_cloud_id=row["child_cloud_id"],
            child_placement_id=row["child_placement_id"],
            position_index=row["position_index"],
            phase=row["phase"],
            weight=row["weight"],
            role=row["role"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )


class SpaceRepository:
    """Repository for Space entities."""
    
    def create(self, space: Space) -> Space:
        with get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO spaces
                (host_cloud_id, layer_id, mode, coordinate_dimensions, scale, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (space.host_cloud_id, space.layer_id, space.mode,
                 space.coordinate_dimensions, space.scale, space.config_json,
                 space.created_at, space.updated_at)
            )
            space.id = cursor.lastrowid
            conn.commit()
        return space
    
    def get_by_id(self, space_id: int) -> Optional[Space]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM spaces WHERE id = ?", (space_id,)).fetchone()
        return self._row_to_space(row) if row else None
    
    def get_by_host_cloud(self, host_cloud_id: int, mode: Optional[str] = None) -> List[Space]:
        with get_connection() as conn:
            if mode:
                rows = conn.execute(
                    "SELECT * FROM spaces WHERE host_cloud_id = ? AND mode = ?",
                    (host_cloud_id, mode)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM spaces WHERE host_cloud_id = ?", (host_cloud_id,)
                ).fetchall()
        return [self._row_to_space(row) for row in rows]
    
    def get_structural_space(self, host_cloud_id: int) -> Optional[Space]:
        return self.get_by_host_cloud(host_cloud_id, mode="structural")[0] if self.get_by_host_cloud(host_cloud_id, mode="structural") else None
    
    def get_semantic_space(self, host_cloud_id: int) -> Optional[Space]:
        return self.get_by_host_cloud(host_cloud_id, mode="semantic")[0] if self.get_by_host_cloud(host_cloud_id, mode="semantic") else None

    def get_global_space(self, layer_id: int) -> Optional[Space]:
        for space in self.get_by_host_cloud(0, mode="global"):
            if space.layer_id == layer_id:
                return space
        return None
    
    def _row_to_space(self, row: sqlite3.Row) -> Space:
        return Space(
            id=row["id"],
            host_cloud_id=row["host_cloud_id"],
            layer_id=row["layer_id"],
            mode=row["mode"],
            coordinate_dimensions=row["coordinate_dimensions"],
            scale=row["scale"],
            config_json=row["config_json"] or "{}",
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )


class LayerRepository:
    """Repository for Layer entities."""
    
    def create(self, layer: Layer) -> Layer:
        with get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO layers (name, order_index, scale, layer_type, config_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (layer.name, layer.order_index, layer.scale, layer.layer_type,
                 layer.config_json, layer.created_at)
            )
            layer.id = cursor.lastrowid
            conn.commit()
        return layer
    
    def get_by_id(self, layer_id: int) -> Optional[Layer]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM layers WHERE id = ?", (layer_id,)).fetchone()
        return self._row_to_layer(row) if row else None
    
    def get_by_name(self, name: str) -> Optional[Layer]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM layers WHERE name = ?", (name,)).fetchone()
        return self._row_to_layer(row) if row else None
    
    def get_all_ordered(self) -> List[Layer]:
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM layers ORDER BY order_index").fetchall()
        return [self._row_to_layer(row) for row in rows]
    
    def _row_to_layer(self, row: sqlite3.Row) -> Layer:
        return Layer(
            id=row["id"],
            name=row["name"],
            order_index=row["order_index"],
            scale=row["scale"],
            layer_type=row["layer_type"],
            config_json=row["config_json"] or "{}",
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )
