# Architecture V2

`clouds` owns identity and accumulated properties. `spaces` owns coordinate systems. `cloud_placements` owns local coordinates and activation. `structural_components` represents ordered containment. `scene_components` represents token occurrences. Hive memory creates local placements in `hive_space` and keeps global provenance.

Storage schema v4 adds typed `semantic_evidence`, `concept_fog_registry`,
`concept_candidate_registry` and idempotent `semantic_backfill_state`. Stable `concept`
clouds own isolated `concept_space` projections; relaxation changes only placements in that
space. Query processing uses the bounded chain `QUERY_FRAME → CONTEXT_INHERITANCE →
QUERY_SCENE_COMPLETION → MEMORY_SCENE_SEARCH → CANDIDATE_RANKING`.
