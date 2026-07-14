"""Transactional repository for the normalized V2 model."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from server.database import get_connection
from .schema import ensure_schema


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def radius_for(mass: float, density: float = 1.0) -> float:
    return min(250.0, max(6.0, 14.0 + 10.0 * math.sqrt(max(0.01, mass * density))))


class V2Repository:
    @contextmanager
    def transaction(self) -> Iterator[Any]:
        with get_connection() as conn:
            ensure_schema(conn)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def ensure_schema(self) -> None:
        with self.transaction():
            pass

    def get_or_create_cloud(
        self, conn: Any, cloud_type: str, canonical_name: str, *, mass: float = 1.0,
        density: float = 1.0, stability: float = 0.1, metadata: Optional[Dict[str, Any]] = None,
        touch: bool = True,
    ) -> Tuple[Dict[str, Any], bool]:
        row = conn.execute(
            "SELECT * FROM v2_clouds WHERE cloud_type = ? AND canonical_name = ? "
            "ORDER BY id LIMIT 1", (cloud_type, canonical_name)
        ).fetchone()
        now = utcnow()
        if row:
            if touch:
                conn.execute(
                    """UPDATE v2_clouds SET observation_count = observation_count + 1,
                    mass = mass + 0.1, stability = MIN(1.0, stability + 0.01), updated_at = ?
                    WHERE id = ?""", (now, row["id"])
                )
                row = conn.execute("SELECT * FROM v2_clouds WHERE id = ?", (row["id"],)).fetchone()
            return dict(row), False
        cursor = conn.execute(
            """INSERT INTO v2_clouds
            (cloud_type, canonical_name, mass, density, stability, base_activation,
             observation_count, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 0, 1, ?, ?, ?)""",
            (cloud_type, canonical_name, mass, density, stability, encode(metadata or {}), now, now),
        )
        row = conn.execute("SELECT * FROM v2_clouds WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row), True

    def set_cloud_accumulation(self, conn: Any, cloud_id: int, mass: float, observations: int) -> None:
        conn.execute(
            "UPDATE v2_clouds SET mass = ?, observation_count = ?, updated_at = ? WHERE id = ?",
            (max(0.0, mass), max(0, observations), utcnow(), cloud_id),
        )

    def get_cloud(self, cloud_id: int) -> Optional[Dict[str, Any]]:
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM v2_clouds WHERE id = ?", (cloud_id,)).fetchone()
            return dict(row) if row else None

    def get_space(self, space_id: int) -> Optional[Dict[str, Any]]:
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM v2_spaces WHERE id = ?", (space_id,)).fetchone()
            return dict(row) if row else None

    def ensure_space(
        self, conn: Any, space_type: str, owner_cloud_id: Optional[int] = None,
        parent_space_id: Optional[int] = None, seed: int = 0,
    ) -> Dict[str, Any]:
        if owner_cloud_id is None:
            row = conn.execute(
                "SELECT * FROM v2_spaces WHERE space_type = ? AND owner_cloud_id IS NULL ORDER BY id LIMIT 1",
                (space_type,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM v2_spaces WHERE space_type = ? AND owner_cloud_id = ? ORDER BY id LIMIT 1",
                (space_type, owner_cloud_id),
            ).fetchone()
        if row:
            return dict(row)
        cursor = conn.execute(
            """INSERT INTO v2_spaces(space_type, owner_cloud_id, parent_space_id, dimensionality,
            random_seed, metadata_json, created_at) VALUES (?, ?, ?, 2, ?, '{}', ?)""",
            (space_type, owner_cloud_id, parent_space_id, seed, utcnow()),
        )
        return dict(conn.execute("SELECT * FROM v2_spaces WHERE id = ?", (cursor.lastrowid,)).fetchone())

    def create_space(
        self, conn: Any, space_type: str, *, owner_cloud_id: Optional[int] = None,
        parent_space_id: Optional[int] = None, seed: int = 0,
    ) -> Dict[str, Any]:
        cursor = conn.execute(
            """INSERT INTO v2_spaces(space_type, owner_cloud_id, parent_space_id, dimensionality,
            random_seed, metadata_json, created_at) VALUES (?, ?, ?, 2, ?, '{}', ?)""",
            (space_type, owner_cloud_id, parent_space_id, seed, utcnow()),
        )
        return dict(conn.execute("SELECT * FROM v2_spaces WHERE id = ?", (cursor.lastrowid,)).fetchone())

    @staticmethod
    def stable_position(namespace: str, index: int = 0) -> Tuple[float, float]:
        value = int(hashlib.sha256(f"{namespace}:{index}".encode()).hexdigest()[:12], 16)
        return 100.0 + float(value % 1400), 100.0 + float((value // 1400) % 800)

    def create_placement(
        self, conn: Any, cloud_id: int, space_id: int, x: float, y: float, *,
        local_activation: float = 0.0, local_density: float = 1.0, local_gravity: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cloud = conn.execute("SELECT mass, density FROM v2_clouds WHERE id = ?", (cloud_id,)).fetchone()
        cursor = conn.execute(
            """INSERT INTO v2_cloud_placements
            (cloud_id, space_id, x, y, radius, local_activation, local_density, local_gravity,
             local_stability_modifier, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
            (cloud_id, space_id, x, y, radius_for(float(cloud["mass"]), float(cloud["density"])),
             local_activation, local_density, local_gravity, encode(metadata or {}), utcnow(), utcnow()),
        )
        return dict(conn.execute("SELECT * FROM v2_cloud_placements WHERE id = ?", (cursor.lastrowid,)).fetchone())

    def ensure_global_placement(self, conn: Any, cloud: Dict[str, Any], global_space_id: int) -> Dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM v2_cloud_placements WHERE cloud_id = ? AND space_id = ? ORDER BY id LIMIT 1",
            (cloud["id"], global_space_id),
        ).fetchone()
        if row:
            return dict(row)
        x, y = self.stable_position(f"{cloud['cloud_type']}:{cloud['canonical_name']}")
        return self.create_placement(conn, int(cloud["id"]), global_space_id, x, y, local_activation=0.1)

    def components(self, conn: Any, parent_cloud_id: int) -> List[Dict[str, Any]]:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM v2_structural_components WHERE parent_cloud_id = ? ORDER BY component_index",
            (parent_cloud_id,),
        ).fetchall()]

    def add_component(
        self, conn: Any, parent_cloud_id: int, child_cloud_id: int, component_index: int,
        component_role: str, local_x: float, local_y: float, metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn.execute(
            """INSERT INTO v2_structural_components(parent_cloud_id, child_cloud_id, component_index,
            component_role, weight, local_x, local_y, metadata_json)
            VALUES (?, ?, ?, ?, 1, ?, ?, ?)""",
            (parent_cloud_id, child_cloud_id, component_index, component_role, local_x, local_y, encode(metadata or {})),
        )

    def normalized_space(self, space_id: int) -> Dict[str, Any]:
        with self.transaction() as conn:
            space = conn.execute("SELECT * FROM v2_spaces WHERE id = ?", (space_id,)).fetchone()
            if not space:
                raise KeyError(space_id)
            placements = [dict(row) for row in conn.execute(
                "SELECT * FROM v2_cloud_placements WHERE space_id = ? ORDER BY id", (space_id,)
            ).fetchall()]
            cloud_ids = sorted({item["cloud_id"] for item in placements})
            clouds = {}
            if cloud_ids:
                marks = ",".join("?" for _ in cloud_ids)
                clouds = {str(row["id"]): dict(row) for row in conn.execute(
                    f"SELECT * FROM v2_clouds WHERE id IN ({marks})", cloud_ids
                ).fetchall()}
            return {"space": dict(space), "clouds": clouds, "placements": placements}
