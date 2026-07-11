# Операции Cosmos

Импорт: `POST /api/v1/sources`; список claims и concepts доступен через
`/api/v1/cosmos/*`. `DELETE /api/v1/sources/{source_id}` логически удаляет
связанные claims и блокирует чтение source artifact; байты забирает storage GC
после grace period. Не меняйте score или provenance прямо в SQLite.
