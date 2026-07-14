"""Public V2 hive facade."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .local_memory import V2LocalMemoryService, HiveLocalMemoryConfig
from .repository import V2Repository


class V2HiveService:
    def __init__(self, repository: Optional[V2Repository] = None, config: Optional[HiveLocalMemoryConfig] = None) -> None:
        self.service = V2LocalMemoryService(repository, config)

    def create(self, max_cells: int = 24) -> Dict[str, Any]:
        return self.service.create_hive(max_cells)

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

