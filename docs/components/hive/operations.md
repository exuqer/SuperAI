# Операции Улья

`POST /api/v1/hives/{hive_id}/freeze` создаёт snapshot; restore выполняется
через `/restore`. Перед restore проверяются tenant и checksum. При переполнении
сначала изучите `evictions` в Hive inspector: защищённую цель нельзя просто
удалить, её нужно сократить или поднять явный budget.
