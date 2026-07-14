# Partial V2 search

`ExternalSearchRequest` contains unresolved components, expected roles, local anchors, excluded known components and a calculated bee/iteration budget. `V2BoundedSwarm` samples only the global field and returns trace data; it never copies anchors into global state.

