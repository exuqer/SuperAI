# Test Scenarios

Train the control corpus, repeat `Кот ест рыбу.`, validate distinct `рыбу` scene placements,
inspect word character structure, run a V2 hive query, and validate model invariants.

For contextual semantics, train `Кот это кошечка`, ask `Что ест кошечка?`, then `А ещё
что?`: prior answers must be excluded, a learned concept match may pass, and unrelated market
scenes must remain visible but rejected by anchor validation. Also verify explicit `кроме`,
inflected forms of `другой`, semantic backfill idempotence, frontend scene counters, and the
production build.
