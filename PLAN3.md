# Emergency AntEngine Inference Optimization

## Summary
- Optimize web chat inference for 250k+ edges by keeping candidate retrieval in SQLite TOP-100 queries and moving candidate math to NumPy.
- Current repo already has an in-memory `edges` table and the exact `SELECT target, weight FROM edges WHERE source = ? ORDER BY weight DESC LIMIT 100` query; implementation will harden the index name, add RAM vector caches, and remove remaining high-cost Python scans from inference.
- `idx_tokens_label ON tokens(label)` will not be added as written because the active inference DB has no `tokens` table and checkpoint `tokens` has no `label` column. Use an in-memory token vector cache instead.

## Engine Changes
- In `semantic_ants/engine.py`, create `idx_edges_source_weight ON edges (source, weight DESC)` during `_edge_index_connection`; keep existing index harmlessly or replace it with the requested name.
- Add runtime caches rebuilt after load, training, feedback, and edge mutations:
  - token label to vector array lookup for candidate vectors;
  - normalized token vector matrix and norms for all-vocab scoring;
  - continuation/out-degree cache keyed by token node id;
  - optional active subgraph outgoing-edge cache for hypernode focus mode.
- Rewrite `_generation_candidates` as a fully vectorized candidate scorer:
  - fetch only TOP-100 outgoing candidates via SQLite when no active local subgraph is focused;
  - build `candidate_ids`, `labels`, `base_weights`, vectors, cosine similarities, repetition masks, transition boosts, degree divisors, terminal masks, and continuation masks as NumPy arrays;
  - compute `scores = base_weights * (1.0 + cosine_similarities)` and apply all boosts/masks vectorized;
  - return only positive-score candidates.
- Rewrite `_synthesize_response` to precompute per-request arrays/state:
  - resolve active graph once;
  - precompute `query_array`, milestone vectors, query token set, and local out-degree map;
  - avoid rebuilding Python lists for generated terminal counts where a simple counter/set can be maintained;
  - call optimized `_generation_candidates` for each generation step.
- Preserve behavior for root graph and hypernode-focused generation.

## Fast Chat Follow-Up
- Keep the previous fast-chat split:
  - default `chat` response returns text/result/backpack metadata only;
  - graph/backpack visuals load from lazy endpoint;
  - chat persistence is debounced async, while training remains sync-save.
- Do not build recursive backpack layers in default chat response.

## Tests
- Add focused tests that verify:
  - `_outgoing_edges` uses TOP-100 behavior and returns strongest edges first;
  - `_generation_candidates` ranks by vectorized `weight * (1 + cosine)` behavior;
  - root generation does not call `_active_graph_edges` or scan all edges;
  - caches refresh after training and feedback.
- Run regressions:
  - `tests/test_engine_vectors.py`
  - `tests/test_engine_preprocess.py`
  - `tests/test_hierarchy_backpack.py`

## Assumptions
- Preserve existing staged/user changes and build on them.
- Optimize web/server inference first, not one-shot CLI startup.
- If execution resumes outside Plan Mode, implement these changes directly without asking further questions.
