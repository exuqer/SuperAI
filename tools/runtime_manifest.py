from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
ENTRYPOINTS = {
    "server.server",
    "server.modules.training.api.router",
    "server.modules.hive.api.router",
    "server.modules.hive.api.query_router",
    "server.modules.universe.api.router",
}
LEGACY_PREFIXES = (
    "server.hive", "server.bees", "server.spaces", "server.memory",
    "server.factories", "server.generation", "server.analytics",
    "server.visualization", "server.modules.model", "server.v2.repository",
    "server.v2.physics", "server.v2.maintenance", "server.v2.validation",
)


def _module(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    current = _module(path).split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names if alias.name.startswith("server."))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = current[:-node.level]
                if node.module:
                    base.extend(node.module.split("."))
                target = ".".join(base)
            else:
                target = str(node.module or "")
            if target.startswith("server."):
                result.add(target)
    return result


def build_manifest() -> list[dict[str, Any]]:
    modules = {_module(path): _imports(path) for path in SERVER.rglob("*.py")}
    reachable = set(ENTRYPOINTS)
    pending = list(ENTRYPOINTS)
    while pending:
        module = pending.pop()
        for dependency in modules.get(module, set()):
            if dependency in modules and dependency not in reachable:
                reachable.add(dependency)
                pending.append(dependency)
    manifest = []
    for module in sorted(modules):
        legacy = module.startswith(LEGACY_PREFIXES)
        manifest.append({
            "module": module,
            "reachable_from_app": module in reachable,
            "runtime_used": module in reachable and not legacy,
            "replacement": "server.v2.hybrid.pipeline" if legacy else None,
            "action": "ARCHIVE" if legacy else ("ACTIVE" if module in reachable else "DELETE_OR_ARCHIVE"),
            "category": "ACTIVE" if module in reachable and not legacy else ("INCOMPATIBLE" if legacy else "DEAD"),
        })
    return manifest


if __name__ == "__main__":
    target = ROOT / ".superai" / "runtime-manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
