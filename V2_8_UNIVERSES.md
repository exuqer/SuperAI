# V2.8: dynamic micro-universes

The persistent model now has a fresh, non-migrating V2.8 schema.  An
incompatible database is discarded and recreated; legacy tables and role
annotations are neither read nor projected into the new model.

Each universe stores stable `Entity` records separately from contextual
`Occurrence` records.  It also owns its base geometry, soft entity clouds,
latent dimensions, dimension clouds, projections, learned transitions and
training history.  The initial universe registry is data-driven and contains
symbols, fragments, words, usages, clauses, events, scenes and abstractions.

`SparseResidualDiscoverer` is the first interchangeable
`DimensionDiscoverer`.  It derives candidate fields from repeated observable
context, evaluates their support and stability, activates sufficiently
confirmed dimensions, writes projections, and records only learned
`correlated` relations.  It never assigns a semantic role or assumes that one
dimension is nested below another.

The read API is rooted at `/api`:

- `GET /api/universes`
- `GET /api/universes/{id}/base-space`
- `GET /api/universes/{id}/dimensions`
- `GET /api/dimensions/{id}` and `/projections`
- `GET /api/entities/{id}/dimension-profile`
- `POST /api/entities/compare`
- `POST /api/visualization/project`
- `GET /api/training/history`

The `/space` client route consumes this API and deliberately renders a base
space separately from the parallel list of discovered dimensions.  Aliases
are stored as UI metadata and never affect discovery or retrieval.
