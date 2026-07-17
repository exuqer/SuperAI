from fastapi.testclient import TestClient

from server.server import app


def test_modular_routes_preserve_v2_contract():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "model": "cloud-space-placement",
            "version": "v2",
        }

        training = client.post("/api/v2/training/learn", json={"text": "Кот ест рыбу."})
        assert training.status_code == 200
        assert training.json()["success"] is True

        hive = client.post(
            "/api/v2/hives",
            json={"max_cells": 24, "conversation_id": "api-contract"},
        )
        assert hive.status_code == 200
        hive_id = hive.json()["hive"]["id"]

        query = client.post(
            f"/api/v2/hives/{hive_id}/query",
            json={"text": "Кот ест рыбу"},
        )
        assert query.status_code == 200
        assert query.json()["decision"]["decision"] in {"MISS", "PARTIAL_HIT", "LOCAL_HIT"}

        reasoning = client.post(
            f"/api/v2/hives/{hive_id}/reasoning",
            json={"text": "Кот ест рыбу", "config": {"reasoning_steps": 1}},
        )
        assert reasoning.status_code == 200
        analytics = client.get(f"/api/v2/hives/{hive_id}/analytics")
        assert analytics.status_code == 200
        assert analytics.json()["primary"]["run"]["id"] == reasoning.json()["run"]["id"]
        assert analytics.json()["primary"]["snapshots"]

        stop = client.post(f"/api/v2/hives/{hive_id}/reasoning/stop")
        assert stop.status_code == 409
        assert stop.json() == {"detail": "synchronous reasoning runs cannot be stopped"}


def test_modular_routes_preserve_not_found_shape():
    with TestClient(app) as client:
        response = client.get("/api/v2/clouds/999999")
    assert response.status_code == 404
    assert response.json() == {"detail": "cloud not found"}


def test_rebuild_scene_event_and_explicit_domain_pack_routes():
    with TestClient(app) as client:
        learned = client.post(
            "/api/v2/training/learn",
            json={"text": "Контейнер доставили в порт."},
        ).json()
        scene_id = learned["scenes"][0]["scene_cloud_id"]

        scene = client.get(f"/api/v2/scenes/{scene_id}")
        assert scene.status_code == 200
        assert scene.json()["scene"]["event"]["source_scene_id"] == scene_id
        assert scene.json()["scene"]["entity_mentions"]

        rebuild = client.post(
            "/api/v2/model/rebuild",
            json={"steps": ["entity_mentions", "event_frames", "indexes"]},
        )
        assert rebuild.status_code == 200
        assert [report["step"] for report in rebuild.json()["reports"]] == [
            "entity_mentions",
            "event_frames",
            "indexes",
        ]

        packs = client.get("/api/v2/knowledge/domain-packs")
        assert packs.status_code == 200
        assert "demo_food" in packs.json()["domain_packs"]
        loaded = client.post("/api/v2/knowledge/domain-packs/demo_food/load")
        assert loaded.status_code == 200
        assert loaded.json()["source_type"] == "domain_pack"
        assert loaded.json()["scene_count"] == 5
