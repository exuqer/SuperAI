"""Explicit, optional domain-pack loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .repository import V2Repository
from .training import TrainingPipelineV2


PACK_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
PACK_DIRECTORY = Path(__file__).resolve().parents[1] / "domain_packs"


class DomainPackService:
    def __init__(self, repository: V2Repository | None = None) -> None:
        self.repository = repository or V2Repository()

    @staticmethod
    def _path(name: str) -> Path:
        if not PACK_NAME.fullmatch(name):
            raise ValueError("invalid domain pack name")
        path = (PACK_DIRECTORY / f"{name}.json").resolve()
        if path.parent != PACK_DIRECTORY.resolve() or not path.is_file():
            raise KeyError(name)
        return path

    def load(self, name: str) -> dict[str, Any]:
        payload = json.loads(self._path(name).read_text(encoding="utf-8"))
        scenes = [
            str(scene).strip()
            for scene in payload.get("scenes", [])
            if str(scene).strip()
        ]
        result = TrainingPipelineV2(self.repository).train(
            " ".join(scenes),
            source_type=f"domain_pack:{name}",
        )
        return {
            "success": bool(result.get("success")),
            "name": name,
            "source_type": "domain_pack",
            "scene_count": len(scenes),
            "training_run_id": result.get("training_run_id"),
            "scenes": result.get("scenes", []),
            "stats": result.get("stats", {}),
        }

    @staticmethod
    def available() -> list[str]:
        if not PACK_DIRECTORY.is_dir():
            return []
        return sorted(path.stem for path in PACK_DIRECTORY.glob("*.json"))
