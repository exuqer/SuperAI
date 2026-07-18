# API SuperAI V2.7

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
