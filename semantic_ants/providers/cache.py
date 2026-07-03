from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


class JsonCache:
    """Файловый JSON-кэш HTTP-ответов и служебных данных."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        _replace_with_retry(tmp, path)

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"


def _replace_with_retry(tmp: Path, target: Path, attempts: int = 20) -> None:
    for index in range(attempts):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if index == attempts - 1:
                raise
            time.sleep(0.05 * (index + 1))
