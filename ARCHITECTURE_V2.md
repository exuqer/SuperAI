# Architecture SuperAI V3.0

V3.0 — единый пространственно-графовый runtime. Непрерывное Semantic Field и
динамические микровселенные организуют смысл, контекстные проекции, скрытые
измерения и маршруты поиска. Событийный граф без фиксированных ролей хранит
доказательства, provenance, отрицания и временные ограничения. Пространственная
близость предлагает гипотезы, а Event Graph подтверждает или отклоняет
фактический ответ.

## Evidential layer: событийный граф V3.0

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
- `graph_schema.py` — свежая SQLite-схема V3.0 и индексы обоих слоёв.
- `graph_learning.py` — similarity, центроиды, local slots, slot sets,
  prototypes, anonymous semantic clusters и construction clusters.
- `event_graph.py` — поздняя морфология и материализация подтверждённых событий.
- `query_graph.py` — gap projection, строгий графовый допуск, binding,
  планирование и обратная проверка ответа.
- `query_operator_learning.py` — профили конкретных употреблений операторов,
  их проекции, slot history и shadow-прогнозы без семантического словаря.
- `graph_service.py` — staging/commit/retract, persistent dialogue, безопасные
  training episodes и передача подтверждённых источников в `UniverseService`.

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

## Самообучение графа

Успешный ответ создаёт зависимый `training_episode`. Он может усилить
конструкцию и совместимость gap со слотом, но вопрос и ответ остаются
лингвистическими свидетельствами со статусом `STAGED`. Событие мира при этом не
создаётся. `UNRESOLVED`, конфликтные и невалидные ответы не обучают модель.

Каждый вопросительный GAP с поверхностным оператором дополнительно сохраняется
как `QueryOperatorOccurrence`. Подтверждённый binding обновляет лишь
наблюдаемые проекции: словоформу, морфологию, контекст запроса, локальные
слоты, облако ответа, отношение к событию и контекст диалога. Значения
`кто/что/где/...` не задаются кодом. На текущем этапе эти профили работают в
режиме `SHADOW`: они видны в trace и собирают историю принятых и отклонённых
кандидатов, но не меняют admission и ранжирование `GraphMatcher`.

## Semantic layer: динамические микровселенные и Semantic Field V3.0

`server/v2/universe.py` строит отдельную производную память. Реестр содержит
микровселенные `symbols`, `morphemes`, `word_forms`, `words`, `usages`,
`clauses`, `events`, `scenes` и `abstractions`.

В каждой микровселенной разделены:

- **Entity** — стабильная сущность данного масштаба;
- **Occurrence** — её конкретное употребление с контекстом и источником;
- **base space** — стабильная трёхмерная стартовая позиция сущностей;
- **entity cloud** — мягкая группа сущностей или употреблений;
- **latent dimension** и **dimension cloud** — обнаруженная структура и её
  область применимости;
- **projection** — принадлежность сущности либо употребления измерению;
- **transition** — наблюдённая связь между масштабами.

После обучения `GraphTrainingService` передаёт в этот слой только источник со
статусом `CONFIRMED`. Semantic Field строит revisioned cloud projections и
участвует в Query Field Projection, локальной активации и dimensional retrieval.
Его результаты хранятся отдельно как `SpatialSupport`; только записи Event Graph
могут стать `GraphEvidence` для режима `OBSERVED`. При retract производные употребления и осиротевшие
сущности удаляются, слабые измерения могут быть помечены `pruned`.

Первый взаимозаменяемый алгоритм — `SparseResidualDiscoverer`. Он извлекает
наблюдаемые контекстные признаки, заводит измерение после двух свидетельств и
делает его `active` при как минимум четырёх свидетельствах от двух сущностей.
Он записывает лишь подтверждённые корреляции; измерения не считаются вложенными
и не получают фиксированных семантических ролей.

Пользовательский alias измерения и `display_label` кластера — только метаданные
интерфейса. Они не участвуют в discovery, retrieval, ранжировании или генерации.

## Версионирование и хранение

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

Схема SQLite: `schema_version = 38`,
`migration_version = fresh-v3.0-spatial-reset`. Несовместимое хранилище
пересоздаётся: перенос прежних таблиц и backfill намеренно не выполняются.



## Воспроизводимый тестовый reset

`TestingResetService` является единственной реализацией разрушительной очистки.
Он используется HTTP API, CLI и compact experiment runner. `FULL_TEST_STATE /
FRESH_SCHEMA` закрывает reusable SQLite handle, удаляет основной файл, WAL и
SHM, создаёт schema 38, восстанавливает пустой реестр микровселенных и очищает
process-local acceleration indexes. `DERIVED_SEMANTIC_SPACE` сохраняет
подтверждённый Event Graph и не запускает неявную пересборку. Каждый reset
создаёт новый `database_generation_id`, audit record и отчёт инвариантов.
