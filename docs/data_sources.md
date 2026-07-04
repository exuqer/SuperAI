# Источники данных

## ConceptNet

Прототип использует ConceptNet REST API. Запросы кэшируются в `.semantic_ants/cache`, чтобы повторные эксперименты не зависели от сети.

## WordNet

WordNet не подключается отдельной тяжелой зависимостью. В MVP используются WordNet-derived связи, которые приходят из ConceptNet через `dataset` и `sources`.

## Synthetic-Persona-Chat

Команда `download-dataset spc` скачивает CSV-сплиты Google Research Synthetic-Persona-Chat по прямым `raw.githubusercontent.com` URL без авторизации. Конвертер извлекает `user 1 personas`, `user 2 personas` и `Best Generated Conversation`, затем сохраняет соседние реплики как JSONL-пары `stimulus` -> `accepted_answer` с историей предыдущих turns.

## Лицензии

При использовании данных ConceptNet и WordNet в производных работах нужно сохранять атрибуцию источников и проверять условия лицензий. Этот прототип хранит `source`, `dataset`, `license` и `edge_id` в metadata ребра.
Synthetic-Persona-Chat распространяется под CC-BY 4.0; конвертер сохраняет эту лицензию в metadata каждого примера.
