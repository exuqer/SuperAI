"""Structured, privacy-aware trace and domain-event recording."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

from .contracts import Budget, DomainEvent, ErrorEnvelope, SpanStatus, TraceSpan, new_id, utcnow
from .database import SqliteDatabase, json_dumps, json_loads


_SECRET_KEYS = {"authorization", "cookie", "password", "secret", "token", "api_key", "apikey"}
_MAX_VALUE_LENGTH = 2_048


def redact(value: Any, *, key: str = "") -> Any:
    """Remove secrets and boundedly summarize arbitrary diagnostic payloads."""
    if key.lower().replace("-", "_") in _SECRET_KEYS:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value[:100]]
    if isinstance(value, str) and len(value) > _MAX_VALUE_LENGTH:
        return value[:_MAX_VALUE_LENGTH] + "…[truncated]"
    return value


class TraceRecorder:
    def __init__(self, database: SqliteDatabase) -> None:
        self.database = database

    def start_span(
        self,
        *,
        trace_id: str,
        component: str,
        operation: str,
        parent_span_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        input_summary: Optional[Dict[str, Any]] = None,
        budget_before: Optional[Budget] = None,
    ) -> TraceSpan:
        span = TraceSpan(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            causation_id=causation_id,
            component=component,
            operation=operation,
            input_summary=redact(input_summary or {}),
            budget_before=budget_before,
        )
        with self.database.transaction() as connection:
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM trace_spans WHERE trace_id = ?", (trace_id,)
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO trace_spans(span_id, trace_id, parent_span_id, sequence, component, operation, status, payload_json, started_at, ended_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    span.span_id,
                    trace_id,
                    parent_span_id,
                    sequence,
                    component,
                    operation,
                    span.status.value,
                    json_dumps(span.model_dump(mode="json")),
                    span.started_at.isoformat(),
                    None,
                ),
            )
        return span

    def finish_span(
        self,
        span: TraceSpan,
        *,
        status: SpanStatus = SpanStatus.SUCCEEDED,
        output_summary: Optional[Dict[str, Any]] = None,
        budget_after: Optional[Budget] = None,
        error: Optional[ErrorEnvelope] = None,
    ) -> TraceSpan:
        ended = utcnow()
        span.status = status
        span.ended_at = ended
        span.duration_ms = max(0, int((ended - span.started_at).total_seconds() * 1000))
        span.output_summary = redact(output_summary or {})
        span.budget_after = budget_after
        span.error = error
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE trace_spans SET status = ?, payload_json = ?, ended_at = ? WHERE span_id = ?",
                (status.value, json_dumps(span.model_dump(mode="json")), ended.isoformat(), span.span_id),
            )
        return span

    def fail_span(self, span: TraceSpan, error: ErrorEnvelope) -> TraceSpan:
        return self.finish_span(span, status=SpanStatus.FAILED, error=error)

    def record_event(self, event: DomainEvent | Dict[str, Any]) -> None:
        if isinstance(event, dict):
            payload = dict(event)
            payload.setdefault("id", new_id("evt-envelope"))
            payload.setdefault("event_id", new_id("evt"))
            payload.setdefault("tenant_id", "local")
            payload.setdefault("task_id", "system")
            payload.setdefault("trace_id", "system")
            event = DomainEvent.model_validate(payload)
        with self.database.transaction() as connection:
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM domain_events WHERE trace_id = ?", (event.trace_id,)
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO domain_events(event_id, trace_id, task_id, sequence, kind, producer, causation_id, payload_json, occurred_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.trace_id,
                    event.task_id,
                    sequence,
                    event.kind,
                    event.producer,
                    event.causation_id,
                    json_dumps(redact(event.payload)),
                    event.occurred_at.isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO outbox(event_id, topic, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    event.event_id,
                    event.kind,
                    json_dumps(redact(event.model_dump(mode="json"))),
                    event.occurred_at.isoformat(),
                ),
            )

    def trace(self, trace_id: str) -> Dict[str, Any]:
        spans = []
        for row in self.database.all("SELECT payload_json FROM trace_spans WHERE trace_id = ? ORDER BY sequence", (trace_id,)):
            spans.append(json_loads(row["payload_json"]))
        events = []
        for row in self.database.all(
            "SELECT d.event_id, d.sequence, d.kind, d.producer, d.causation_id, d.payload_json, d.occurred_at, "
            "o.payload_json AS envelope_json "
            "FROM domain_events d LEFT JOIN outbox o ON o.event_id = d.event_id "
            "WHERE d.trace_id = ? ORDER BY d.sequence",
            (trace_id,),
        ):
            # Outbox stores the complete immutable DomainEvent atomically with
            # its audit timeline. Older rows can still be rendered from the
            # compact event row during a migration.
            event = json_loads(row.pop("envelope_json"), None)
            if event is None:
                event = {
                    "event_id": row["event_id"],
                    "id": row["event_id"],
                    "kind": row["kind"],
                    "producer": row["producer"],
                    "causation_id": row["causation_id"],
                    "payload": json_loads(row["payload_json"], {}),
                    "occurred_at": row["occurred_at"],
                }
            event["sequence"] = row["sequence"]
            events.append(event)
        return {"trace_id": trace_id, "spans": spans, "events": events}

    def pending_outbox(self, limit: int = 100) -> list[Dict[str, Any]]:
        return self.database.all(
            "SELECT * FROM outbox WHERE published_at IS NULL ORDER BY created_at LIMIT ?", (limit,)
        )

    def acknowledge_outbox(self, event_id: str) -> None:
        self.database.execute("UPDATE outbox SET published_at = ? WHERE event_id = ?", (utcnow().isoformat(), event_id))
