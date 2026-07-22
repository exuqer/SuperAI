from __future__ import annotations

from fastapi.testclient import TestClient

from server.core.settings import settings
from server.server import create_app


def test_local_testing_reset_is_available_without_an_environment_flag(monkeypatch) -> None:
    monkeypatch.setattr(settings, "allow_test_reset", False)
    monkeypatch.setattr(settings, "admin_token", "")
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v2/testing/reset",
            json={
                "scope": "FULL_TEST_STATE",
                "mode": "FRESH_SCHEMA",
                "confirmation": "RESET TEST SPACE",
            },
        )
    assert response.status_code == 200


def test_local_testing_reset_contract(monkeypatch) -> None:
    monkeypatch.setattr(settings, "allow_test_reset", True)
    monkeypatch.setattr(settings, "test_reset_localhost_only", True)
    monkeypatch.setattr(settings, "admin_token", "")
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v2/testing/reset",
            json={
                "scope": "FULL_TEST_STATE",
                "mode": "FRESH_SCHEMA",
                "confirmation": "RESET TEST SPACE",
            },
        )
        listing = client.get("/api/universes")
        field = client.get("/api/v2/semantic-field")

    assert response.status_code == 200
    payload = response.json()
    assert payload["reset"] is True
    assert payload["field_revision"] == 0
    assert payload["invariants"]["runtime_caches_empty"] is True
    assert listing.status_code == 200
    assert listing.json()["universes"]
    assert field.status_code == 200
    assert field.json()["field_revision"] == 0


def test_testing_reset_requires_exact_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(settings, "allow_test_reset", True)
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v2/testing/reset",
            json={
                "scope": "FULL_TEST_STATE",
                "mode": "FRESH_SCHEMA",
                "confirmation": "reset",
            },
        )
    assert response.status_code == 422
