# Hive vibration audit

Hive memory is implemented by `hives`, `hive_cells`, `hive_cell_components` and local `cloud_placements` in a `hive_space`. Global clouds and placements are provenance-only during chat. `V2LocalMemoryService` creates/reuses the hive by `conversation_id`; `HiveVibrationEngine` owns dynamic node state, snapshots and local physics.

