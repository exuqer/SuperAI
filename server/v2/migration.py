"""Safe V1-to-V2 backup, rebuild, validation and comparison commands."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from server.database import get_db_path
from server.tokenizer import tokenize_hierarchical
from .repository import V2Repository
from .training import TrainingPipelineV2
from .validation import ModelInvariantValidator


def backup_database(path: Optional[Path] = None) -> Path:
    source = path or get_db_path()
    if not source.exists():
        raise FileNotFoundError(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = source.with_name(f"{source.stem}.v1-backup-{stamp}{source.suffix}")
    shutil.copy2(source, target)
    return target


def migrate_schema_v2() -> Dict[str, Any]:
    repository = V2Repository()
    backup_path = backup_database() if get_db_path().exists() else None
    repository.ensure_schema()
    return {"success": True, "schema": "v2", "backup": str(backup_path) if backup_path else None}


def rebuild_model_v2(*, backup: bool = True) -> Dict[str, Any]:
    repository = V2Repository()
    repository.ensure_schema()
    backup_path = backup_database() if backup and get_db_path().exists() else None
    with repository.transaction() as conn:
        tables = [row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'v2_%' AND name <> 'v2_schema_meta'"
        ).fetchall()]
        for table in reversed(tables):
            conn.execute(f'DELETE FROM "{table}"')
    pipeline = TrainingPipelineV2(repository)
    imported = 0
    try:
        with repository.transaction() as conn:
            rows = conn.execute("""SELECT s.sentence_text, s.scene_cloud_id, c.mass, c.observation_count
                FROM scenes s JOIN clouds c ON c.id = s.scene_cloud_id ORDER BY s.id""").fetchall()
    except Exception:
        rows = []
    for row in rows:
        result = pipeline.train(row["sentence_text"], source_type="v1_rebuild")
        if result["created_scene_cloud_ids"]:
            imported += 1
            with repository.transaction() as conn:
                repository.set_cloud_accumulation(conn, result["created_scene_cloud_ids"][0], float(row["mass"]), int(row["observation_count"]))
    transferred = 0
    try:
        with repository.transaction() as conn:
            source_clouds = conn.execute("""SELECT c.canonical_name, c.mass, c.observation_count, l.name AS layer
                FROM clouds c JOIN layers l ON l.id = c.layer_id""").fetchall()
            type_map = {"character": "character", "word_form": "word_form", "lexeme": "lexeme", "concept": "concept_candidate", "scene": "scene"}
            for source in source_clouds:
                cloud_type = type_map.get(source["layer"])
                if not cloud_type:
                    continue
                target = conn.execute(
                    "SELECT id FROM v2_clouds WHERE cloud_type = ? AND canonical_name = ? ORDER BY id LIMIT 1",
                    (cloud_type, source["canonical_name"]),
                ).fetchone()
                if target:
                    repository.set_cloud_accumulation(conn, int(target["id"]), float(source["mass"]), int(source["observation_count"]))
                    transferred += 1
    except Exception:
        transferred = 0
    return {"success": True, "backup": str(backup_path) if backup_path else None, "imported_scenes": imported, "transferred_clouds": transferred}


def synchronize_legacy_field(repository: Optional[V2Repository] = None) -> Dict[str, int]:
    repository = repository or V2Repository()
    pipeline = TrainingPipelineV2(repository)
    imported_scenes = 0
    imported_words = 0
    with repository.transaction() as conn:
        legacy_scenes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'scenes'"
        ).fetchone()
        legacy_clouds = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'clouds'"
        ).fetchone()
        if not legacy_scenes or not legacy_clouds:
            return {"imported_scenes": 0, "imported_words": 0}
        global_space = repository.ensure_space(conn, "global_field", seed=1337)
        scene_rows = conn.execute(
            """SELECT s.id, s.sentence_text, s.updated_at, c.mass, c.observation_count
            FROM scenes s JOIN clouds c ON c.id = s.scene_cloud_id
            ORDER BY s.id"""
        ).fetchall()
        for row in scene_rows:
            imported = conn.execute(
                "SELECT source_updated_at FROM v2_legacy_scene_imports WHERE legacy_scene_id = ?",
                (row["id"],),
            ).fetchone()
            if imported and imported["source_updated_at"] == row["updated_at"]:
                continue
            tokenization = tokenize_hierarchical(row["sentence_text"])
            scene_id: Optional[int] = None
            for sentence in tokenization.sentences:
                outcome = pipeline._train_sentence(
                    conn, sentence, int(global_space["id"]), "v1_synchronization"
                )
                scene_id = int(outcome["scene_cloud_id"])
            if scene_id is None:
                continue
            repository.set_cloud_accumulation(
                conn, scene_id, float(row["mass"]), int(row["observation_count"])
            )
            conn.execute(
                """INSERT INTO v2_legacy_scene_imports
                (legacy_scene_id, source_updated_at, v2_scene_cloud_id, imported_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(legacy_scene_id) DO UPDATE SET
                source_updated_at = excluded.source_updated_at,
                v2_scene_cloud_id = excluded.v2_scene_cloud_id,
                imported_at = excluded.imported_at""",
                (row["id"], row["updated_at"], scene_id, datetime.now(timezone.utc).isoformat()),
            )
            imported_scenes += 1
        word_rows = conn.execute(
            """SELECT c.canonical_name, c.mass, c.observation_count
            FROM clouds c JOIN layers l ON l.id = c.layer_id
            WHERE l.name = 'word_form' ORDER BY c.id"""
        ).fetchall()
        for row in word_rows:
            target = conn.execute(
                "SELECT id FROM v2_clouds WHERE cloud_type = 'word_form' AND canonical_name = ?",
                (row["canonical_name"],),
            ).fetchone()
            if not target:
                tokens = tokenize_hierarchical(row["canonical_name"]).all_tokens
                if len(tokens) != 1:
                    continue
                word, _ = pipeline._ensure_word(
                    conn,
                    tokens[0],
                    pipeline.morphology.parse(tokens[0].normalized),
                    int(global_space["id"]),
                )
                target = {"id": word["id"]}
                imported_words += 1
            repository.set_cloud_accumulation(
                conn, int(target["id"]), float(row["mass"]), int(row["observation_count"])
            )
    return {"imported_scenes": imported_scenes, "imported_words": imported_words}


def clear_model_v2(repository: Optional[V2Repository] = None) -> None:
    repository = repository or V2Repository()
    with repository.transaction() as conn:
        for table in (
            "v2_hive_cell_matches",
            "v2_hive_resonance_events",
            "v2_hive_query_decisions",
            "v2_hive_messages",
            "v2_hive_cell_components",
            "v2_hive_cells",
            "v2_hives",
            "v2_legacy_scene_imports",
            "v2_training_observations",
            "v2_scene_components",
            "v2_scenes",
            "v2_semantic_memberships",
            "v2_word_forms",
            "v2_lexemes",
            "v2_structural_components",
            "v2_cloud_placements",
            "v2_spaces",
            "v2_clouds",
        ):
            conn.execute(f"DELETE FROM {table}")


def compare_v1_v2() -> Dict[str, Any]:
    repository = V2Repository()
    with repository.transaction() as conn:
        try:
            v1 = int(conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0])
        except Exception:
            v1 = 0
        v2 = int(conn.execute("SELECT COUNT(*) FROM v2_scenes").fetchone()[0])
    return {"v1_scenes": v1, "v2_scenes": v2, "scene_count_match": v1 == v2}


def rollback_to_v1(backup_path: Optional[str] = None) -> Dict[str, Any]:
    target = get_db_path()
    candidates = sorted(target.parent.glob(f"{target.stem}.v1-backup-*{target.suffix}"))
    source = Path(backup_path) if backup_path else (candidates[-1] if candidates else None)
    if source is None or not source.exists():
        raise FileNotFoundError("No V1 backup available")
    shutil.copy2(source, target)
    return {"success": True, "restored": str(source)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["migrate_schema_v2", "rebuild_model_v2", "validate_model_v2", "compare_v1_v2", "rollback_to_v1"])
    parser.add_argument("--backup-path")
    args = parser.parse_args()
    actions = {
        "migrate_schema_v2": migrate_schema_v2,
        "rebuild_model_v2": rebuild_model_v2,
        "validate_model_v2": lambda: ModelInvariantValidator().validate(),
        "compare_v1_v2": compare_v1_v2,
        "rollback_to_v1": lambda: rollback_to_v1(args.backup_path),
    }
    print(actions[args.command]())


if __name__ == "__main__":
    main()
