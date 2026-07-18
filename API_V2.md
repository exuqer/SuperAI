# API SuperAI V2.8

Базовый адрес при локальном запуске: `http://127.0.0.1:8000`. Полная
машиночитаемая схема доступна в `/openapi.json`, интерактивная — в `/docs`.

## Обучение

`POST /api/v2/training/learn`

```json
{
  "text": "Борис настроил датчик.",
  "source_type": "training",
  "independent_key": "document-17",
  "domain_key": "devices"
}
```

Ответ содержит `source_id`, статус фактуальности, события, локальные слоты,
наборы слотов, прототипы и полный языковой анализ. Вопрос, условие или гипотеза
получают `STAGED` и не создают `graph_events`.

`POST /api/v2/training/stage` принимает тот же контракт.

`POST /api/v2/training/commit`

```json
{"source_id": "graph-source-...", "manual_validation": false}
```

`POST /api/v2/training/retract`

```json
{"source_id": "graph-source-...", "reason": "outdated"}
```

Ручной commit не может превратить вопрос, команду или гипотезу в факт мира.
Повторный stage подтверждённого источника не понижает его статус.

`POST /api/v2/training/reprocess` повторно обрабатывает указанный источник.

Batch-операции:

- `POST /api/v2/training/batches/preview`;
- `POST /api/v2/training/batches/commit`;
- `POST /api/v2/training/batches/rollback`.

`batch_id` задаёт точную группу источников: commit и rollback не затрагивают
источники других batch.

## Диалог

`POST /api/v2/hives`

```json
{"max_cells": 24, "conversation_id": "dialogue-1"}
```

`POST /api/v2/hives/query/parse`

```json
{"text": "Кто настроил датчик?"}
```

Возвращает `query_graph` и `language_analysis`, не изменяя диалог.

`POST /api/v2/hives/{id}/query`

```json
{
  "text": "Кто настроил датчик?",
  "resolved_mode": "NEW_QUERY",
  "retrieval_scope": "LOCAL_ONLY"
}
```

`NEW_QUERY` запрещает наследование предыдущего графа. `FOLLOW_UP` и
`CORRECTION` оставляют предыдущий граф доступным для структурного продолжения.

Основные поля ответа:

```json
{
  "query_graph": {},
  "candidate_bindings": [],
  "rejected_events": [],
  "selected_binding": {},
  "answer": {
    "status": "RESOLVED",
    "surface": "Борис.",
    "provenance": {
      "source_event_ids": [],
      "independent_source_count": 1
    },
    "validation": {"valid": true},
    "versions": {}
  },
  "trace": {}
}
```

`GET /api/v2/hives/{id}/trace` возвращает морфологические гипотезы,
кандидатов упоминаний и событий, гипотезы слотов и конструкций, preliminary и
final QueryGraph, memory feedback, принятые/отклонённые события, binding,
ResponsePlan и validation.

`GET /api/v2/hives/{id}/bindings` возвращает текущие candidate bindings,
выбранное binding и причины отклонения событий.

Дополнительно доступны:

- `GET /api/v2/hives/{id}` — состояние улья;
- `POST /api/v2/hives/{id}/query/preview` — разбор без исполнения запроса;
- `GET /api/v2/hives/{id}/space-export` — полный переносимый снимок выбранного
  диалогового пространства;
- `POST /api/v2/hives/{id}/rank/step` и `/rank/run` — шаг или серия шагов
  вибрационного ранжирования;
- `POST /api/v2/graphs/parse` — разобрать текст в `QueryGraph`;
- `GET /api/v2/graphs/hives/{id}` — рабочее состояние QueryGraph для улья.

## Микровселенные

Эти методы читают производную память V2.8. После `learn` ответ содержит
`universe_update`; он сообщает, был ли подтверждённый источник спроецирован в
микровселенные.

| Метод | Назначение |
| --- | --- |
| `GET /api/universes` | Реестр микровселенных и их статистика. |
| `GET /api/universes/{id}/base-space` | Сущности, облака и при необходимости употребления контекста. Параметры: `limit`, `min_mass`, `min_stability`, `selected_context`. |
| `GET /api/universes/{id}/dimensions` | Обнаруженные измерения. Фильтры: `status`, `scope`, `min_stability`, `min_utility`, `owner_cloud_id`. |
| `GET /api/dimensions/{id}` | Измерение и его связи. |
| `GET /api/dimensions/{id}/projections` | Проекции; фильтры `source_type`, `limit`, `min_membership`, `context_id`, `sort`. |
| `GET /api/entities/{id}/dimension-profile` | Сущность, употребления и принадлежности измерениям. |
| `POST /api/entities/compare` | Сравнить ровно две сущности одной микровселенной. |
| `POST /api/visualization/project` | Получить двумерную экранную проекцию выбранного пространства. |
| `GET /api/universes/{id}/transitions` | Наблюдённые переходы из микровселенной. |
| `GET /api/training/history` | События самоорганизации; фильтры `universe_id`, `dimension_id`, `limit`. |
| `PUT /api/dimensions/{id}/alias` | Сохранить UI-псевдоним измерения. |

Пример сравнения:

```json
POST /api/entities/compare
{
  "universe_id": "words",
  "entity_ids": ["universe-entity-…", "universe-entity-…"]
}
```

`POST /api/visualization/project` принимает `universe_id`, необязательные
`space_type`, `dimension_ids` (не больше восьми), `projection_method`, `limit`
и `filters`. Экранные координаты — только представление: они не равны расстоянию
модели.

## Операции с памятью

- `GET /api/health` — статус процесса и версия событийного графа.
- `GET /api/export/memory` — переносимый JSON-снимок всех таблиц текущей памяти.
- `POST /api/reset` — **разрушительно** удаляет граф и производные
  микровселенные, затем создаёт пустой реестр микровселенных.
