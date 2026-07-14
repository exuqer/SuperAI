"""Placement-only physics with a strict one-space boundary."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .repository import V2Repository, utcnow


class PlacementPhysicsV2:
    def __init__(self, space_id: int, repository: Optional[V2Repository] = None) -> None:
        self.space_id = space_id
        self.repository = repository or V2Repository()

    def tick(self) -> List[Dict[str, float]]:
        with self.repository.transaction() as conn:
            space = conn.execute("SELECT * FROM spaces WHERE id = ?", (self.space_id,)).fetchone()
            if not space:
                raise KeyError(self.space_id)
            if space["space_type"] == "word_structure_space":
                raise ValueError("word structure spaces are not physical simulation spaces")
            rows = [dict(row) for row in conn.execute(
                """SELECT p.*, c.mass, c.stability FROM cloud_placements p
                JOIN clouds c ON c.id = p.cloud_id WHERE p.space_id = ?""",
                (self.space_id,),
            )]
            updates: List[Dict[str, float]] = []
            for item in rows:
                dx = dy = gravity = 0.0
                for other in rows:
                    if other["id"] == item["id"]:
                        continue
                    distance_x = float(other["x"]) - float(item["x"])
                    distance_y = float(other["y"]) - float(item["y"])
                    distance = max(1.0, math.hypot(distance_x, distance_y))
                    force = min(2.0, math.sqrt(float(item["mass"]) * float(other["mass"])) / (distance * distance))
                    gravity += force
                    dx += force * distance_x / distance
                    dy += force * distance_y / distance
                damping = 1.0 - min(
                    0.9,
                    float(item["stability"]) + float(item["local_stability_modifier"]),
                )
                x = float(item["x"]) + dx * damping
                y = float(item["y"]) + dy * damping
                conn.execute(
                    """UPDATE cloud_placements SET x = ?, y = ?, local_gravity = ?, updated_at = ?
                    WHERE id = ? AND space_id = ?""",
                    (x, y, gravity, utcnow(), item["id"], self.space_id),
                )
                updates.append({"placement_id": int(item["id"]), "x": x, "y": y, "local_gravity": gravity})
            return updates
