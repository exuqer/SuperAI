# Dynamic Hypernodes + Fast Canvas Graph

## Summary
- Hypernodes are built from explicit `hierarchy`.
- Backend computes structure, depth, and graph coordinates.
- Frontend renders through Cytoscape.js Canvas; drag is enabled; browser heavy layout is not used.

## Key Changes
- `preprocess.py` supports dialogue and hierarchy modes.
- Dialogue mode filters JSONL and writes `[__user__] ... [__assistant__] ... .`
- Hierarchy mode walks directories, code/doc files, and `.txt/.text` with headings, then emits `{"hierarchy":[...],"text":"..."}`.
- `engine.py`:
  - `train_jsonl` ingests raw dialogue JSONL directly;
  - `train_jsonl` ingests hierarchy JSONL and builds persisted `hypernodes`;
  - each `hypernode` has `vector`, `label`, `parent`, `subgraph`;
  - subgraph contains local token nodes, `hierarchical_edge`, and `transition_edge`.
- Context stack:
  - session-scoped `backpack_stack`;
  - `drill_down(node_id)` push;
  - `drill_up()` pop;
  - cosine scoring uses the current stack focus vector.
- `POST /api/chat/message` returns `current_depth`, `total_depth_layers`, `active_focus_label`, `graph_data.nodes`, and `graph_data.edges`.

## Frontend Graph
- Use pinned Cytoscape.js `3.31.2` in the current static web.
- Use `layout: { name: "preset" }`; coordinates come from backend `x/y`.
- Do not put fCOSE in the main path; it is CPU-heavy for the browser.
- Enable Cytoscape drag/pan/zoom.
- For performance:
  - hide labels on large graphs;
  - keep rendering limited to the current focus/layer viewport;
  - do not recreate the instance without a graph id change;
  - update selected/active via classes.

## Test Plan
- `python -m unittest discover -s tests -q`.
- Add tests for hierarchy preprocessing, hypernode training, stack depth, backpack schema, and edge types.
- Check UI on 48, 160, and 1000 nodes: render, drag, pan, zoom, selection.

## Assumptions
- For current limits up to `1000` nodes, Cytoscape Canvas is better than the current SVG path.
- If the graph later needs tens of thousands of nodes, switch separately to Sigma.js/WebGL.
