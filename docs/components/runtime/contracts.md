# Контракты runtime

`WorkItem` содержит `command_id`, `task_id`, `trace_id`, handler, versioned
payload, status, attempt, `scheduled_at`, idempotency key, snapshot budget и
normalised `last_error`. `TraceSpan` и `DomainEvent` несут causation и
correlation identifiers; тяжёлое содержимое передаётся через `ArtifactRef`.

Новые поля major 1 игнорируются на входной границе. Незнакомый major
`schema_version` отклоняется до постановки в очередь.
