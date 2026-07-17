from server.v2.hive import V2HiveService
from server.v2.maintenance import UniversalKnowledgeRebuilder
from server.v2.repository import V2Repository, utcnow
from server.v2.training import TrainingPipelineV2


def ask(service: V2HiveService, text: str) -> dict:
    hive_id = service.create()["hive"]["id"]
    service.query(hive_id, text)
    return service.vibration_run(hive_id, 3)["answer"]


def test_multidomain_entities_events_roles_and_compound_mentions():
    repository = V2Repository()
    TrainingPipelineV2(repository).train(
        "Артём ремонтирует робота Искра. "
        "Анна передала контейнер Сергею. "
        "Контейнер доставили в порт. "
        "Карта висит на южной стене. "
        "Зелье восстанавливает здоровье героя. "
        "Мастер работает молотком."
    )
    service = V2HiveService(repository)

    assert ask(service, "Кого ремонтирует Артём?")["surface_answer"] == "Искру."
    assert ask(service, "Кому Анна передала контейнер?")["surface_answer"] == "Сергею."
    assert ask(service, "Куда доставили контейнер?")["surface_answer"] == "В порт."
    assert ask(service, "Где висит карта?")["surface_answer"] == "На южной стене."
    assert ask(service, "Что восстанавливает зелье?")["surface_answer"] == "Здоровье героя."

    with repository.transaction() as conn:
        repair_scene = conn.execute(
            "SELECT cloud_id FROM scenes WHERE sentence_text LIKE 'Артём%'"
        ).fetchone()[0]
        repair_mentions = conn.execute(
            "SELECT surface,mention_type FROM entity_mentions WHERE source_scene_id=?",
            (repair_scene,),
        ).fetchall()
        assert [(row["surface"], row["mention_type"]) for row in repair_mentions] == [
            ("Артём", "noun_phrase"),
            ("робота Искра", "apposition"),
        ]
        health_scene = conn.execute(
            "SELECT cloud_id FROM scenes WHERE sentence_text LIKE 'Зелье%'"
        ).fetchone()[0]
        assert conn.execute(
            """SELECT COUNT(*) FROM event_participants ep
               JOIN events e ON e.id=ep.event_id WHERE e.source_scene_id=?""",
            (health_scene,),
        ).fetchone()[0] == 2
        modifier_roles = {
            row["role"]
            for row in conn.execute("SELECT role FROM event_modifiers")
        }
        assert {"attribute", "owner"} <= modifier_roles
        relation_types = {
            row["relation_type"]
            for row in conn.execute(
                "SELECT relation_type FROM concept_relations"
            )
        }
        assert {
            "IS_A", "OWNS", "HAS_PROPERTY", "LOCATED_ON", "USES",
        } <= relation_types


def test_constructions_accumulate_evidence_without_predicate_lists():
    repository = V2Repository()
    TrainingPipelineV2(repository).train(
        "Кот ремонтирует машину. Медведь ремонтирует стену. "
        "Робот ремонтирует карту."
    )
    with repository.transaction() as conn:
        template = conn.execute(
            """SELECT * FROM construction_templates
               WHERE predicate_lemma='ремонтировать'"""
        ).fetchone()
        assert template["evidence_count"] == 3
        assert template["status"] == "PROBABLE"
        roles = {
            row["semantic_role"]
            for row in conn.execute(
                "SELECT semantic_role FROM construction_arguments"
            )
        }
        assert roles <= {
            "agent", "patient", "theme", "object", "cause",
        }
        assert all("_or_" not in role for role in roles)
    service = V2HiveService(repository)
    hive_id = service.create()["hive"]["id"]
    result = service.query(hive_id, "Какую машину ремонтирует кот?")
    construction_stage = next(
        stage for stage in result["query_frame"]["retrieval_stages"]
        if stage["stage"] == "construction"
    )
    assert construction_stage["construction_ids"]
    assert construction_stage["considered"] == 1


def test_explicit_causal_and_temporal_markers_create_relations():
    repository = V2Repository()
    TrainingPipelineV2(repository).train(
        "Шторм вызывает задержку. Доставка после погрузки. "
        "Разрушитель разрушает стену. Разрушение вызывает остановку."
    )
    with repository.transaction() as conn:
        rows = conn.execute(
            """SELECT s.canonical_lemma AS subject,cr.relation_type,
                      o.canonical_lemma AS object
               FROM concept_relations cr
               JOIN entities s ON s.cloud_id=cr.subject_lexeme_cloud_id
               JOIN entities o ON o.cloud_id=cr.object_lexeme_cloud_id"""
        ).fetchall()
        relations = {
            (row["subject"], row["relation_type"], row["object"])
            for row in rows
        }
        morphological_link = conn.execute(
            """SELECT source_type FROM concept_relations
               WHERE relation_type='ALIAS_OF'
                 AND source_type='MORPHOLOGICAL_LINK'"""
        ).fetchone()
    assert ("шторм", "CAUSES", "задержка") in relations
    assert ("шторм", "RESULTS_IN", "задержка") in relations
    assert ("доставка", "AFTER", "погрузка") in relations
    assert ("погрузка", "BEFORE", "доставка") in relations
    assert morphological_link is not None


def test_rebuild_is_idempotent_preserves_sources_and_removes_only_manual_seeds():
    repository = V2Repository()
    TrainingPipelineV2(repository).train("Контейнер доставили в порт.")
    with repository.transaction() as conn:
        now = utcnow()
        conn.execute(
            """INSERT INTO action_concepts
               (id,canonical_name,display_name,status,confidence,mass,
                evidence_count,created_at,updated_at)
               VALUES('manual-only','manual-only','manual-only','PROBABLE',
                      .5,.5,1,?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO action_variants
               (id,action_concept_id,lemma,weight,evidence_count,source_type,
                created_at,updated_at)
               VALUES('manual-variant','manual-only','manual-only',.5,1,
                      'manual_seed',?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO action_concepts
               (id,canonical_name,display_name,status,confidence,mass,
                evidence_count,created_at,updated_at)
               VALUES('imported','imported','imported','PROBABLE',
                      .5,.5,1,?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO action_variants
               (id,action_concept_id,lemma,weight,evidence_count,source_type,
                created_at,updated_at)
               VALUES('imported-variant','imported','imported',.5,1,
                      'imported',?,?)""",
            (now, now),
        )
        scene_id = int(conn.execute("SELECT cloud_id FROM scenes").fetchone()[0])
        conn.execute(
            """INSERT INTO concepts
               (id,concept_kind,canonical_name,display_name,status,confidence,
                evidence_count,source_type,created_at,updated_at)
               VALUES('manual-normalized-empty','action','manual-normalized-empty',
                      'manual-normalized-empty','CANDIDATE',.5,0,'manual_seed',?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO concepts
               (id,concept_kind,canonical_name,display_name,status,confidence,
                evidence_count,source_type,created_at,updated_at)
               VALUES('manual-normalized-supported','action',
                      'manual-normalized-supported','manual-normalized-supported',
                      'PROBABLE',.7,1,'manual_seed',?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO concept_evidence
               (id,concept_id,source_scene_id,evidence_type,weight,confidence,
                independence_key,status,payload_json,created_at)
               VALUES('supported-evidence','manual-normalized-supported',?,
                      'EXPLICIT_SIMILARITY',.8,.8,'imported-observation','ACTIVE',
                      '{}',?)""",
            (scene_id, now),
        )
        conn.execute(
            """INSERT INTO semantic_constructions
               (id,predicate_lemma,pattern_type,argument_mapping_json,
                implied_semantics_json,confidence,evidence_count)
               VALUES('imported-construction','imported-construction',
                      'imported','{}','{}',.8,1)"""
        )
        source_before = [
            tuple(row)
            for row in conn.execute(
                """SELECT s.cloud_id,s.sentence_text,p.x,p.y
                   FROM scenes s
                   JOIN cloud_placements p ON p.cloud_id=s.cloud_id
                   ORDER BY s.cloud_id,p.id"""
            )
        ]
    rebuilder = UniversalKnowledgeRebuilder(repository)
    first = rebuilder.rebuild()
    with repository.transaction() as conn:
        counts_after_first = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "entity_mentions",
                "events",
                "construction_templates",
                "concept_relations",
                "scene_concept_projections",
            )
        }
    second = rebuilder.rebuild()
    ordered = rebuilder.rebuild(["concept_relations", "entity_mentions"])
    indexes_only = rebuilder.rebuild(["indexes"])
    with repository.transaction() as conn:
        counts_after_second = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in counts_after_first
        }
        source_after = [
            tuple(row)
            for row in conn.execute(
                """SELECT s.cloud_id,s.sentence_text,p.x,p.y
                   FROM scenes s
                   JOIN cloud_placements p ON p.cloud_id=s.cloud_id
                   ORDER BY s.cloud_id,p.id"""
            )
        ]
        concept_ids = {
            row["id"] for row in conn.execute("SELECT id FROM action_concepts")
        }
        normalized_concept_ids = {
            row["id"] for row in conn.execute("SELECT id FROM concepts")
        }
        construction_ids = {
            row["id"] for row in conn.execute("SELECT id FROM semantic_constructions")
        }
    assert first["success"] and second["success"]
    assert first["reports"] and second["reports"]
    assert ordered["steps"] == ["entity_mentions", "concept_relations"]
    assert indexes_only["processed_scenes"] == 0
    assert counts_after_first == counts_after_second
    assert source_before == source_after
    assert "manual-only" not in concept_ids
    assert "imported" in concept_ids
    assert "manual-normalized-empty" not in normalized_concept_ids
    assert "manual-normalized-supported" in normalized_concept_ids
    assert "imported-construction" in construction_ids


def test_indexed_exact_retrieval_does_not_scan_all_scenes():
    repository = V2Repository()
    TrainingPipelineV2(repository).train(
        "Техник ремонтирует модуль. Контейнер прибыл в порт. "
        "Карта лежит на столе. Герой выпил зелье."
    )
    service = V2HiveService(repository)
    hive_id = service.create()["hive"]["id"]
    result = service.query(hive_id, "Кто ремонтирует модуль?")
    metrics = result["query_frame"]["retrieval_metrics"]
    assert metrics["indexed"] is True
    assert metrics["scenes_considered"] == 1
    assert metrics["scenes_considered"] < metrics["scenes_total"]
    assert result["query_frame"]["retrieval_stages"][0]["stage"] == "exact_form"
    assert result["role_hypotheses"]
    assert result["scene_matches"]
    assert "pre_candidates" in result
    assert "accepted_candidates" in result
    assert "rejected_candidates" in result
