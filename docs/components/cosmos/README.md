# Cosmos

Космос хранит concepts и claims, а не единую «истину» или векторный индекс.
Каждый claim содержит subject/predicate/object, точный source artifact и
fragment, sector views, access scope, verification status и независимые
scores. Противоречащие claims остаются отдельными строками.

Импорт `.txt`/Markdown проходит archive → quarantine → детерминированный
sentence/term extraction → интеграция. Повторный content hash не создаёт новый
source или claims. Удаление source маркирует только зависимые claims.

Retrieval делает bounded lexical candidate search, access/project filtering,
reranking и diversity. Он возвращает score/reasons, exact source и отмеченные
противоречия; источники другой области доступа не загружаются.
