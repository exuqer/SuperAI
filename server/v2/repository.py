"""Transactional repository for the canonical model."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

import server.database as database
from .schema import ensure_schema


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode(value: Optional[str], default: Any = None) -> Any:
    if not value:
        return {} if default is None else default
    return json.loads(value)


def radius_for(mass: float, density: float = 1.0) -> float:
    return min(250.0, max(6.0, 14.0 + 10.0 * math.sqrt(max(0.01, mass * density))))


class V2Repository:
    @contextmanager
    def transaction(self) -> Iterator[Any]:
        with database.get_connection() as conn:
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
        self,
        conn: Any,
        cloud_type: str,
        canonical_name: str,
        *,
        mass: float = 1.0,
        density: float = 1.0,
        stability: float = 0.1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        row = conn.execute(
            "SELECT * FROM clouds WHERE cloud_type = ? AND canonical_name = ? ORDER BY id LIMIT 1",
            (cloud_type, canonical_name),
        ).fetchone()
        if row:
            return dict(row), False
        now = utcnow()
        cursor = conn.execute(
            """INSERT INTO clouds
            (cloud_type, canonical_name, mass, density, stability, base_activation,
             observation_count, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 0, 1, ?, ?, ?)""",
            (cloud_type, canonical_name, mass, density, stability, encode(metadata or {}), now, now),
        )
        return dict(conn.execute("SELECT * FROM clouds WHERE id = ?", (cursor.lastrowid,)).fetchone()), True

    def strengthen_cloud(self, conn: Any, cloud_id: int, amount: float = 0.1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        before = dict(conn.execute("SELECT * FROM clouds WHERE id = ?", (cloud_id,)).fetchone())
        conn.execute(
            """UPDATE clouds SET mass = mass + ?, observation_count = observation_count + 1,
            stability = MIN(1.0, stability + 0.01), updated_at = ? WHERE id = ?""",
            (amount, utcnow(), cloud_id),
        )
        after = dict(conn.execute("SELECT * FROM clouds WHERE id = ?", (cloud_id,)).fetchone())
        return before, after

    def get_or_create_space(
        self,
        conn: Any,
        space_type: str,
        owner_cloud_id: Optional[int] = None,
        parent_space_id: Optional[int] = None,
        seed: int = 0,
    ) -> Tuple[Dict[str, Any], bool]:
        if space_type == "global_field":
            row = conn.execute("SELECT * FROM spaces WHERE space_type = 'global_field' LIMIT 1").fetchone()
        elif owner_cloud_id is not None:
            row = conn.execute(
                "SELECT * FROM spaces WHERE space_type = ? AND owner_cloud_id = ? LIMIT 1",
                (space_type, owner_cloud_id),
            ).fetchone()
        else:
            row = None
        if row:
            return dict(row), False
        cursor = conn.execute(
            """INSERT INTO spaces(space_type, owner_cloud_id, parent_space_id, dimensionality,
            random_seed, metadata_json, created_at) VALUES (?, ?, ?, 2, ?, '{}', ?)""",
            (space_type, owner_cloud_id, parent_space_id, seed, utcnow()),
        )
        return dict(conn.execute("SELECT * FROM spaces WHERE id = ?", (cursor.lastrowid,)).fetchone()), True

    def create_space(
        self,
        conn: Any,
        space_type: str,
        *,
        owner_cloud_id: Optional[int] = None,
        parent_space_id: Optional[int] = None,
        seed: int = 0,
    ) -> Dict[str, Any]:
        cursor = conn.execute(
            """INSERT INTO spaces(space_type, owner_cloud_id, parent_space_id, dimensionality,
            random_seed, metadata_json, created_at) VALUES (?, ?, ?, 2, ?, '{}', ?)""",
            (space_type, owner_cloud_id, parent_space_id, seed, utcnow()),
        )
        return dict(conn.execute("SELECT * FROM spaces WHERE id = ?", (cursor.lastrowid,)).fetchone())

    @staticmethod
    def stable_position(namespace: str, index: int = 0) -> Tuple[float, float]:
        value = int(hashlib.sha256(f"{namespace}:{index}".encode()).hexdigest()[:12], 16)
        return 100.0 + float(value % 1400), 100.0 + float((value // 1400) % 800)

    def create_placement(
        self,
        conn: Any,
        cloud_id: int,
        space_id: int,
        x: float,
        y: float,
        *,
        local_activation: float = 0.0,
        local_density: float = 1.0,
        local_gravity: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cloud = conn.execute("SELECT mass, density FROM clouds WHERE id = ?", (cloud_id,)).fetchone()
        now = utcnow()
        cursor = conn.execute(
            """INSERT INTO cloud_placements
            (cloud_id, space_id, x, y, radius, local_activation, local_density, local_gravity,
             local_stability_modifier, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
            (
                cloud_id,
                space_id,
                x,
                y,
                radius_for(float(cloud["mass"]), float(cloud["density"])),
                local_activation,
                local_density,
                local_gravity,
                encode(metadata or {}),
                now,
                now,
            ),
        )
        return dict(conn.execute("SELECT * FROM cloud_placements WHERE id = ?", (cursor.lastrowid,)).fetchone())

    def ensure_global_placement(self, conn: Any, cloud: Dict[str, Any], global_space_id: int) -> Tuple[Dict[str, Any], bool]:
        row = conn.execute(
            "SELECT * FROM cloud_placements WHERE cloud_id = ? AND space_id = ? ORDER BY id LIMIT 1",
            (cloud["id"], global_space_id),
        ).fetchone()
        if row:
            return dict(row), False
        x, y = self.stable_position(f"{cloud['cloud_type']}:{cloud['canonical_name']}")
        return self.create_placement(
            conn,
            int(cloud["id"]),
            global_space_id,
            x,
            y,
            local_activation=0.1,
            metadata={"placement_kind": "global"},
        ), True

    def components(self, conn: Any, parent_cloud_id: int) -> List[Dict[str, Any]]:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM structural_components WHERE parent_cloud_id = ? ORDER BY component_index",
            (parent_cloud_id,),
        ).fetchall()]

    def stats(self, conn: Any) -> Dict[str, Any]:
        clouds_by_type = {
            row["cloud_type"]: int(row["count"])
            for row in conn.execute("SELECT cloud_type, COUNT(*) AS count FROM clouds GROUP BY cloud_type")
        }
        spaces_by_type = {
            row["space_type"]: int(row["count"])
            for row in conn.execute("SELECT space_type, COUNT(*) AS count FROM spaces GROUP BY space_type")
        }
        def scalar(sql: str) -> int:
            return int(conn.execute(sql).fetchone()[0])
        return {
            "clouds_total": sum(clouds_by_type.values()),
            "clouds_by_type": clouds_by_type,
            "spaces_total": sum(spaces_by_type.values()),
            "spaces_by_type": spaces_by_type,
            "placements_total": scalar("SELECT COUNT(*) FROM cloud_placements"),
            "unique_word_forms": clouds_by_type.get("word_form", 0),
            "scene_components_total": scalar("SELECT COUNT(*) FROM scene_components"),
            "structural_components_total": scalar("SELECT COUNT(*) FROM structural_components"),
            "concepts_total": clouds_by_type.get("concept", 0),
        }

    def normalized_space(self, space_id: int) -> Dict[str, Any]:
        with self.transaction() as conn:
            space = conn.execute("SELECT * FROM spaces WHERE id = ?", (space_id,)).fetchone()
            if not space:
                raise KeyError(space_id)
            placements = [dict(row) for row in conn.execute(
                "SELECT * FROM cloud_placements WHERE space_id = ? ORDER BY id", (space_id,)
            )]
            cloud_ids = sorted({int(item["cloud_id"]) for item in placements})
            clouds: Dict[str, Dict[str, Any]] = {}
            if cloud_ids:
                marks = ",".join("?" for _ in cloud_ids)
                clouds = {
                    str(row["id"]): dict(row)
                    for row in conn.execute(f"SELECT * FROM clouds WHERE id IN ({marks})", cloud_ids)
                }
            return {
                "space": dict(space),
                "clouds": clouds,
                "placements": placements,
                "stats": self.stats(conn),
            }

    def get_cloud(self, cloud_id: int) -> Optional[Dict[str, Any]]:
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM clouds WHERE id = ?", (cloud_id,)).fetchone()
            return dict(row) if row else None

    def get_placement(self, placement_id: int) -> Optional[Dict[str, Any]]:
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM cloud_placements WHERE id = ?", (placement_id,)).fetchone()
            return dict(row) if row else None

    def trained_model_snapshot(self) -> Dict[str, Any]:
        tables = (
            "clouds", "spaces", "cloud_placements", "structural_components",
            "lexemes", "word_forms", "word_form_features", "cloud_compositions",
            "morph_pattern_data", "semantic_memberships", "scenes",
            "scene_components", "training_runs", "training_observations",
            "training_change_events",
        )
        ordering = {
            "lexemes": "cloud_id",
            "word_forms": "cloud_id",
            "morph_pattern_data": "cloud_id",
            "scenes": "cloud_id",
        }

        def snapshot_row(row: Any) -> Dict[str, Any]:
            item = dict(row)
            for key, value in tuple(item.items()):
                if key.endswith("_json"):
                    item[key.removesuffix("_json")] = decode(value)
                    del item[key]
            return item

        with self.transaction() as conn:
            model = {
                table: [
                    snapshot_row(row)
                    for row in conn.execute(f"SELECT * FROM {table} ORDER BY {ordering.get(table, 'id')}")
                ]
                for table in tables
            }
            schema_version = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            return {
                "schema_version": int(schema_version),
                "stats": self.stats(conn),
                "model": model,
            }

    def clear(self) -> None:
        with self.transaction() as conn:
            for table in (
                "hive_cell_matches", "hive_resonance_events", "hive_query_decisions",
                "hive_messages", "hive_cell_components", "hive_cells", "hives",
                "training_change_events", "training_observations", "training_runs",
                "scene_components", "scenes", "semantic_memberships", "word_forms",
                "lexemes", "structural_components", "cloud_placements", "spaces", "clouds",
            ):
                conn.execute(f"DELETE FROM {table}")
