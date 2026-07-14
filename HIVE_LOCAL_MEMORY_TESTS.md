# Local-memory test scenarios

- repeated trained scene: second query is `LOCAL_HIT` with zero bees;
- missing object: `PARTIAL_HIT` searches only the unresolved object;
- weak two-percent component: remains `MISS`;
- repeated token: mention factor is logarithmic and bounded;
- negated scene: polarity mismatch is `CONFLICT`;
- near-identical nectar: `MERGE_EXISTING` does not create a cell;
- validator: all local values remain in `0..1` and resonance references remain valid.

