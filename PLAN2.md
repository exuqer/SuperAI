# Hierarchical Local Edges + Reinforcement Scoring

## Summary
- Root edges live inside `hyper:__root__.subgraph.edges`, not as a top-level checkpoint field.
- Token-transition edges are stored inside each `hypernode.subgraph.edges`.
- Response synthesis runs against the active local subgraph: current Context Stack focus or `hyper:__root__`.
- NumPy scoring is used instead of optimizer-based gradient descent.

## Key Changes
- `Checkpoint`:
  - root `tokens` remain the global vector index;
  - `edges` are not serialized as a top-level checkpoint field;
  - `load_checkpoint()` migrates legacy root edges into `hyper:__root__.subgraph.edges`;
  - `save_checkpoint()` writes the tree-shaped SQLite checkpoint.
- `subgraph`:
  - `tokens` is a dict of local token duplicates;
  - `edges` is a dict of slim edge records;
  - edge format remains `"token:a|next|token:b": {"weight": 1.0, "pheromone": 1.0}`;
  - edge payloads do not need `id/source/target/created_at/updated_at`.
- Training:
  - hierarchy records create a chain of hypernodes;
  - token `next` edges for text are written into the leaf subgraph;
  - parent subgraphs keep child hypernode refs and `hierarchical_edge`;
  - non-hierarchy train/corpus/feedback writes into `hyper:__root__`.
- `_synthesize_response`:
  - uses the active subgraph from the Context Stack;
  - scores outgoing candidates with local edge weights, cosine similarity, and reinforcement;
  - milestone candidates receive a bonus and repeat-blacklisted candidates are heavily damped.
- Vectorization:
  - avoid loops over score arrays when possible;
  - use NumPy masks, `np.where`, broadcasting, and matrix ops.

## Public Interfaces
- Frontend/API `graph_data.edges` remains an array; engine reconstructs `id/source/target/relation/type` from slim edge keys on output.
- Train reports, health, and progress count `edges` as the sum of local `subgraph.edges`.
- `drill_down` and `drill_up` keep controlling the active hypernode through Context Stack.

## Test Plan
- Verify saved checkpoint is `checkpoint.sqlite`.
- Verify slim edge payload contains only `weight` and `pheromone`.
- Verify migration of legacy root edges into `hyper:__root__`.
- Verify hierarchy training writes token `next` edges only into the leaf subgraph.
- Verify `_synthesize_response` reads only the active subgraph.
- Verify scoring behavior and run `python -m unittest discover -s tests -q`.

## Assumptions
- Persisted hypernodes are created during training, not on every chat.
- Reinforcement weights change local edge weights and pheromone during training; synthesis only recomputes probabilities in NumPy.
