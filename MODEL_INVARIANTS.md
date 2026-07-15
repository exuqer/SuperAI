# Model Invariants

- Structural component indices are unique per parent.
- Word structures contain exactly one component per character position.
- A cloud may have many placements; a placement belongs to one space.
- Word form, lexeme and concept clouds are distinct.
- Scene components are unique by token index and always have a role and confidence.
- Hive component shares sum to one and hive coordinates are not copied from global placements.
- A stable semantic fog has one `concept` owner and one isolated `concept_space`.
- Semantic backfill is idempotent and never changes global or hive coordinates.
- A scene is admitted as an answer candidate only when the requested role exists and every required anchor scores at least `0.45`.
