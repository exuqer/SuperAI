# V2 local-memory audit

V2 previously created a new hive and synchronously ranked global placements on every `forage` call. It had no persistent query decision, component-level activation, or local-first routing. The new path is `V2LocalMemoryService.query`: parse → lookup → resonance → unresolved-only bounded search → merge → decay.

Global V2 clouds and global placements are read-only during chat queries. Hive cells own local coordinates and activation state.

