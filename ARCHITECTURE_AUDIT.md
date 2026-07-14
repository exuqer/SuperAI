# Architecture Audit

- V1 stores mixed cloud identity, coordinates and structural placements in `clouds`, `spaces`, `cloud_placements`, `structural_components`, and JSON scene arrays.
- `GET /api/field/hierarchy` duplicates scene word arrays and projects child coordinates into the global field.
- V1 bee sampling reads global placements; V2 is isolated in `v2_*` tables and does not alter V1 behaviour.
