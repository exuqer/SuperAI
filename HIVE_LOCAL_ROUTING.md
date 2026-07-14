# Hive local routing

`HiveQueryRouter` classifies each parsed component as `LOCAL_HIT`, `PARTIAL_HIT`, `MISS`, `CONFLICT`, `AMBIGUOUS`, or `STALE_HIT`. A local hit uses no bees. A partial hit sends only unresolved components and local anchors to the bounded V2 sampler.

Support includes match quality, composition share, role compatibility, activation, retention and recency. Background shares are capped and cannot independently satisfy the local-hit threshold.

