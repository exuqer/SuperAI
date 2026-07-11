# Контракты Cosmos

`Concept` — каноническая label/type/aliases запись. `Claim` всегда ссылается на
`source_id`, source artifact и exact fragment, имеет sectors, `AccessScope`,
verification status и раздельные scores. `RetrievedClaim` добавляет score,
reasons и IDs противоречащих claims, не меняя исходное утверждение.
