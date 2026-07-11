# Контракты Улья

`HiveView` включает identity, lifecycle state, текущий revision
`TaskContract`, hot/warm `ContextEntry`, references на знания/планы, critic
reports и budget ledger. `ContextEntry` несёт `content_type`, size, source,
relevance, protected flag и reconstruction cost. `EvictionDecision` всегда
содержит reason-code и score до/после.
