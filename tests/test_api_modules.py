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
