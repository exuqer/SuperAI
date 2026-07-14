"""Placement-only physics with an explicit single-space boundary."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .repository import V2Repository, utcnow


class PlacementPhysicsV2:
    def __init__(self, space_id: int, repository: Optional[V2Repository] = None) -> None:
        self.space_id = space_id
        self.repository = repository or V2Repository()

    def tick(self) -> List[Dict[str, float]]:
        with self.repository.transaction() as conn:
            rows = [dict(row) for row in conn.execute(
                """SELECT p.*, c.mass, c.stability FROM v2_cloud_placements p
                JOIN v2_clouds c ON c.id = p.cloud_id WHERE p.space_id = ?""", (self.space_id,)
            ).fetchall()]
            updates = []
            for item in rows:
                dx = dy = 0.0
                for other in rows:
                    if other["id"] == item["id"]:
                        continue
                    distance_x, distance_y = other["x"] - item["x"], other["y"] - item["y"]
                    distance = max(1.0, math.hypot(distance_x, distance_y))
                    force = min(2.0, (float(item["mass"]) * float(other["mass"])) ** 0.5 / (distance * distance))
                    dx += force * distance_x / distance
                    dy += force * distance_y / distance
                damping = 1.0 - min(0.9, float(item["stability"]) + float(item["local_stability_modifier"]))
                x, y = float(item["x"]) + dx * damping, float(item["y"]) + dy * damping
                conn.execute("UPDATE v2_cloud_placements SET x = ?, y = ?, updated_at = ? WHERE id = ? AND space_id = ?", (x, y, utcnow(), item["id"], self.space_id))
                updates.append({"placement_id": item["id"], "x": x, "y": y})
            return updates

