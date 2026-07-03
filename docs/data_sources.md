# Источники данных

## ConceptNet

Прототип использует ConceptNet REST API. Запросы кэшируются в `.semantic_ants/cache`, чтобы повторные эксперименты не зависели от сети.

## WordNet

WordNet не подключается отдельной тяжелой зависимостью. В MVP используются WordNet-derived связи, которые приходят из ConceptNet через `dataset` и `sources`.

## Лицензии

При использовании данных ConceptNet и WordNet в производных работах нужно сохранять атрибуцию источников и проверять условия лицензий. Этот прототип хранит `source`, `dataset`, `license` и `edge_id` в metadata ребра.
