"""Idempotent rebuild commands for derived universal knowledge."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable

from .event_core import UniversalEventPipeline
from .repository import V2Repository
from .training import RussianMorphology


REBUILD_STEPS = (
    "phrase_graphs",
    "entity_mentions",
    "appositions",
    "event_frames",
    "event_participants",
    "construction_templates",
    "role_distributions",
    "spatial_relations",
    "answer_surfaces",
    "concept_relations",
    "scene_concept_projections",
    "indexes",
)

STEP_TABLES = {
    "phrase_graphs": ("entity_mentions",),
    "entity_mentions": ("entities", "entity_aliases", "entity_mentions"),
    "appositions": (
        "entities", "entity_aliases", "entity_mentions", "concept_relations",
    ),
    "event_frames": (
        "events",
        "event_participants",
        "event_modifiers",
        "event_role_hypotheses",
    ),
    "event_participants": (
        "event_participants", "event_modifiers", "event_role_hypotheses",
    ),
    "construction_templates": (
        "construction_templates",
        "construction_arguments",
        "construction_evidence",
    ),
    "role_distributions": (
        "construction_arguments", "event_role_hypotheses",
    ),
    "spatial_relations": (
        "concept_relations", "concept_relation_evidence",
    ),
    "answer_surfaces": ("entity_mentions", "event_participants"),
    "concept_relations": (
        "concepts",
        "concept_members",
        "concept_evidence",
        "concept_relations",
        "concept_relation_evidence",
    ),
    "scene_concept_projections": ("scene_concept_projections",),
    "indexes": (),
}


class UniversalKnowledgeRebuilder:
    def __init__(self, repository: V2Repository | None = None) -> None:
        self.repository = repository or V2Repository()
        self.pipeline = UniversalEventPipeline(self.repository, RussianMorphology())

    @staticmethod
    def _remove_manual_seeds(conn: Any) -> Dict[str, int]:
        concept_ids = [
            str(row["action_concept_id"])
            for row in conn.execute(
                """SELECT action_concept_id
                   FROM action_variants
                   GROUP BY action_concept_id
                   HAVING SUM(CASE WHEN source_type<>'manual_seed' THEN 1 ELSE 0 END)=0"""
            ).fetchall()
        ]
        removed_projections = 0
        removed_concepts = 0
        for concept_id in concept_ids:
            removed_projections += int(conn.execute(
                "SELECT COUNT(*) FROM scene_concept_projections WHERE action_concept_id=?",
                (concept_id,),
            ).fetchone()[0])
            conn.execute("DELETE FROM action_concepts WHERE id=?", (concept_id,))
            removed_concepts += 1
        conn.execute(
            """DELETE FROM concepts
               WHERE source_type='manual_seed'
                 AND NOT EXISTS (
                   SELECT 1 FROM concept_evidence
                   WHERE concept_evidence.concept_id=concepts.id
                 )"""
        )
        return {
            "removed_manual_seed_concepts": removed_concepts,
            "removed_manual_seed_projections": removed_projections,
        }

    def _materialize_all(self, conn: Any) -> Dict[str, int]:
        processed = 0
        for row in conn.execute("SELECT cloud_id FROM scenes ORDER BY cloud_id").fetchall():
            self.pipeline.materialize_scene(conn, int(row["cloud_id"]))
            processed += 1
        return {
            "processed_scenes": processed,
            "entity_mentions": int(conn.execute(
                "SELECT COUNT(*) FROM entity_mentions"
            ).fetchone()[0]),
            "events": int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
            "construction_templates": int(conn.execute(
                "SELECT COUNT(*) FROM construction_templates"
            ).fetchone()[0]),
            "concept_relations": int(conn.execute(
                "SELECT COUNT(*) FROM concept_relations"
            ).fetchone()[0]),
            "scene_concept_projections": int(conn.execute(
                "SELECT COUNT(*) FROM scene_concept_projections"
            ).fetchone()[0]),
        }

    def rebuild(self, steps: Iterable[str] | None = None) -> Dict[str, Any]:
        raw_steps = list(steps or REBUILD_STEPS)
        invalid = sorted(set(raw_steps) - set(REBUILD_STEPS))
        if invalid:
            raise ValueError(f"unknown rebuild steps: {', '.join(invalid)}")
        requested_set = set(raw_steps)
        requested = [
            step for step in REBUILD_STEPS
            if step in requested_set
        ]
        with self.repository.transaction() as conn:
            before_counts = {
                table: int(conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0])
                for step in requested
                for table in STEP_TABLES[step]
            }
            before_scenes = [
                (int(row["cloud_id"]), str(row["sentence_text"]))
                for row in conn.execute(
                    "SELECT cloud_id,sentence_text FROM scenes ORDER BY cloud_id"
                ).fetchall()
            ]
            removed = self._remove_manual_seeds(conn)
            result = (
                self._materialize_all(conn)
                if any(step != "indexes" for step in requested)
                else {
                    "processed_scenes": 0,
                    "entity_mentions": int(conn.execute(
                        "SELECT COUNT(*) FROM entity_mentions"
                    ).fetchone()[0]),
                    "events": int(conn.execute(
                        "SELECT COUNT(*) FROM events"
                    ).fetchone()[0]),
                    "construction_templates": int(conn.execute(
                        "SELECT COUNT(*) FROM construction_templates"
                    ).fetchone()[0]),
                    "concept_relations": int(conn.execute(
                        "SELECT COUNT(*) FROM concept_relations"
                    ).fetchone()[0]),
                    "scene_concept_projections": int(conn.execute(
                        "SELECT COUNT(*) FROM scene_concept_projections"
                    ).fetchone()[0]),
                }
            )
            after_scenes = [
                (int(row["cloud_id"]), str(row["sentence_text"]))
                for row in conn.execute(
                    "SELECT cloud_id,sentence_text FROM scenes ORDER BY cloud_id"
                ).fetchall()
            ]
            if before_scenes != after_scenes:
                raise RuntimeError("source scenes changed during rebuild")
            after_counts = {
                table: int(conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0])
                for step in requested
                for table in STEP_TABLES[step]
            }
            reports = []
            for step in requested:
                tables = STEP_TABLES[step]
                reports.append({
                    "step": step,
                    "created": sum(
                        max(0, after_counts[table] - before_counts[table])
                        for table in tables
                    ),
                    "updated": (
                        sum(after_counts[table] for table in tables)
                        if tables
                        else 0
                    ),
                    "deleted": sum(
                        max(0, before_counts[table] - after_counts[table])
                        for table in tables
                    ),
                    "counts": {
                        table: after_counts[table]
                        for table in tables
                    },
                })
            return {
                "success": True,
                "schema_version": 8,
                "steps": requested,
                "reports": reports,
                "source_scenes_preserved": len(after_scenes),
                **removed,
                **result,
            }

    def rebuild_entity_mentions(self) -> Dict[str, Any]:
        return self.rebuild(["entity_mentions"])

    def rebuild_phrase_graphs(self) -> Dict[str, Any]:
        return self.rebuild(["phrase_graphs"])

    def rebuild_appositions(self) -> Dict[str, Any]:
        return self.rebuild(["appositions"])

    def rebuild_event_frames(self) -> Dict[str, Any]:
        return self.rebuild(["event_frames"])

    def rebuild_event_participants(self) -> Dict[str, Any]:
        return self.rebuild(["event_participants"])

    def rebuild_construction_templates(self) -> Dict[str, Any]:
        return self.rebuild(["construction_templates"])

    def rebuild_role_distributions(self) -> Dict[str, Any]:
        return self.rebuild(["role_distributions"])

    def rebuild_spatial_relations(self) -> Dict[str, Any]:
        return self.rebuild(["spatial_relations"])

    def rebuild_answer_surfaces(self) -> Dict[str, Any]:
        return self.rebuild(["answer_surfaces"])

    def rebuild_concept_relations(self) -> Dict[str, Any]:
        return self.rebuild(["concept_relations"])

    def rebuild_scene_concept_projections(self) -> Dict[str, Any]:
        return self.rebuild(["scene_concept_projections"])

    def rebuild_indexes(self) -> Dict[str, Any]:
        return self.rebuild(["indexes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    rebuild = subparsers.add_parser("rebuild")
    rebuild.add_argument("step", choices=(*REBUILD_STEPS, "all"))
    args = parser.parse_args()
    steps = None if args.step == "all" else [args.step]
    print(json.dumps(
        UniversalKnowledgeRebuilder().rebuild(steps),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
