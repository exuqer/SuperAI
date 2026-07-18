# Architecture SuperAI V2.7

## Границы вычислительного ядра

Универсальные узлы:

```text
EVENT
MENTION
ENTITY_REFERENCE
VALUE
RELATION_INSTANCE
GAP
CONSTRUCTION
```

Структурные связи:

```text
EVENT_HAS_PARTICIPANT
MENTION_HAS_COMPONENT
VALUE_ATTACHED_TO_NODE
COREFERS_TO
EXCLUDES
CONTINUES
SUPPORTED_BY
CONTRADICTS
```

## Модули

- `graph_models.py` — неизменяемые контракты узлов, сигнатур, слотов и binding.
- `graph_schema.py` — свежая SQLite-схема V2.7 и индексы.
- `graph_learning.py` — similarity, центроиды, local slots, slot sets,
  prototypes, anonymous semantic clusters и construction clusters.
- `event_graph.py` — поздняя морфология и материализация подтверждённых событий.
- `query_graph.py` — gap projection, строгий графовый допуск, binding,
  планирование и обратная проверка ответа.
- `graph_service.py` — staging/commit/retract, persistent dialogue и безопасные
  training episodes; batch membership хранится явно и изолирует commit/rollback.

## Допуск запроса

```text
predicate index
→ confirmed/actual filter
→ known node identity
→ required mention components
→ structural relations
→ candidate unbound nodes
→ exclusions
→ slot/construction compatibility
→ bounded ranking
→ selected binding
```

Мягкая динамика видит только события, прошедшие строгие проверки.
Сигнатура известного узла участвует в мягкой оценке после строгого совпадения
его идентичности, компонентов и структурного отношения.

## Самообучение

Успешный ответ создаёт зависимый `training_episode`. Он может усилить
конструкцию и совместимость gap со слотом, но вопрос и ответ остаются
лингвистическими свидетельствами со статусом `STAGED`. Событие мира при этом не
создаётся. `UNRESOLVED`, конфликтные и невалидные ответы не обучают модель.

## Версионирование

Каждый артефакт фиксирует:

```text
event_schema_version
slot_model_version
construction_model_version
semantic_cluster_version
query_graph_version
generation_version
migration_version
```

`migration_version = fresh-v2.7`: перенос старой схемы намеренно не выполняется.
