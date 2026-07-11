from superai.contracts import DomainEvent
from superai.database import SqliteDatabase
from superai.observability import TraceRecorder


def test_events_are_redacted_in_both_audit_and_trace_representations(tmp_path) -> None:
    database = SqliteDatabase(tmp_path / "trace.sqlite3")
    traces = TraceRecorder(database)
    traces.record_event(
        DomainEvent(
            id="event-envelope",
            task_id="task-secret",
            trace_id="trace-secret",
            tenant_id="tenant-a",
            kind="SensitiveEvent",
            producer="test",
            payload={"token": "actual-secret", "nested": {"api_key": "another-secret"}, "safe": "value"},
        )
    )

    event = traces.trace("trace-secret")["events"][0]
    assert event["payload"]["token"] == "[REDACTED]"
    assert event["payload"]["nested"]["api_key"] == "[REDACTED]"
    raw_outbox = database.one("SELECT payload_json FROM outbox WHERE event_id = ?", (event["event_id"],))
    assert "actual-secret" not in raw_outbox["payload_json"]
    assert "another-secret" not in raw_outbox["payload_json"]
    database.close()
