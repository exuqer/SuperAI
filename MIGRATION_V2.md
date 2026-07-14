# V2 database reset

The V2 local-memory schema is intentionally destructive during this development phase. Stop the server and remove `.superai/state.sqlite`; the next startup recreates both legacy and V2 tables from the current schema. No migration or rollback path is maintained.
