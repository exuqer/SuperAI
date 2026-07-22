# Реализованные изменения V3.0

## Пространственно-графовый runtime

- Event Graph оставлен evidential layer: события, provenance, отрицания, время и
  строгая проверка фактического ответа.
- Semantic Field и микровселенные оформлены как semantic layer: облака,
  контекстные проекции, скрытые измерения, пространственная активация и
  ассоциативный поиск.
- `GraphEvidence` и `SpatialSupport` разделены. Пространственный результат без
  графового evidence не может получить epistemic mode `OBSERVED`.
- Semantic Field использует event-local coactivation вместо притяжения всех
  понятий одного документа или всего пространства.
- Force trace получает relation, predicate, polarity, event и source context.
- Ревизии поля хранят previous/proposed/applied positions, validation и причины
  движения.
- Contextual projections и source contributions пересчитываются из активных
  подтверждённых источников и не усиливаются от технического rebuild.
- Координаты скрытых измерений связаны с разреженными Universe projections;
  XYZ используется только как display/bootstrap projection.

## Очистка пространства для тестирования

- Добавлен единый `TestingResetService` для API, CLI, экспериментов и старого
  admin-alias `/api/reset`.
- Поддержаны области `FULL_TEST_STATE`, `DERIVED_SEMANTIC_SPACE`,
  `DIALOGUE_STATE`, `REASONING_TRACES`, `EXPERIMENT_STATE`.
- Полный режим `FRESH_SCHEMA` удаляет SQLite, WAL/SHM, создаёт свежую схему,
  реестр микровселенных и новый generation id.
- `CLEAR_DATA` выполняет атомарную FK-aware очистку выбранной области.
- Process-local acceleration indexes и runtime caches очищаются через registry.
- Добавлены reset audit, before/after counters и проверяемые invariants.
- Добавлена защищённая кнопка «Очистить тестовое пространство» в клиенте,
  очистка Pinia/localStorage/sessionStorage и последующая проверка backend.
- Выборочная очистка Semantic Field сохраняет Event Graph и предлагает явный
  rebuild производного слоя.

## Инфраструктура

- Схема обновлена до `schema_version = 38`, migration
  `fresh-v3.0-spatial-reset`.
- Health отделён от readiness; readiness реально проверяет SQLite-схему и
  доступность `pymorphy3`.
- Добавлены `requirements.lock`, frontend lock и GitHub Actions CI.
- Удалён недостижимый role-based multilevel runtime, бинарные cache-файлы и
  временные debug-скрипты.

## Локальная проверка

В среде сборки выполнены:

```text
python -m compileall -q server tests
python -m pytest -q tests/test_testing_reset.py \
  tests/api/test_testing_reset_api.py \
  tests/semantic_field/test_reset_invariants.py \
  tests/test_settings.py
```

Результат целевого набора: `8 passed`.

Полная языковая регрессия требует установки locked dependency `pymorphy3`.

## Исправление Auto retrieval и перехода Field → Evidence Graph

- Режим интерфейса «Авто» теперь передаёт `LOCAL_THEN_GLOBAL`; отдельные режимы
  `LOCAL_ONLY` и `GLOBAL_ONLY` сохраняют строгую семантику.
- Обычный query contract использует `retrieval_scope`; `resonance_scope`
  остаётся только у отдельного локального resonance action.
- `LOCAL_THEN_GLOBAL` выполняет локальную фазу, а затем глобальный fallback
  только при отсутствии устойчивого evidential ответа.
- В debug payload добавлен `scope_trace`: запрошенная и разрешённая область,
  число локальных/глобальных событий, фаза и факт fallback.
- Структурированный query anchor использует `concept_id/entity_id/node_id`, а не
  строковое представление словаря.
- Неоднозначные predicate lemmas сохраняются из морфологических анализов и
  проверяются по GraphEvidence; словарный hardcode отдельных омонимов удалён.
- Необученный question operator остаётся широким `EVENT_ATTACHMENT` до
  подтверждённой специализации его shadow profile.
- Добавлен `EVENT_ATTACHMENT` candidate builder с сохранением предлога в
  surface answer, например `у забора`.
- Добавлен этап `FIELD_TO_GRAPH_BRIDGE`: активированное semantic cloud
  разрешается в concept, participant, event и только затем в `GraphEvidence`.
- Field Bee возвращает `SPATIAL_SUPPORT_FOUND`, а не ложный общий `FOUND`;
  нулевой результат получает `NO_RESULT`.
- Spatial support дедуплицируется по query, field revision, cloud и region и не
  размножается при повторном обнаружении пчелой.
- Добавлены чистые regression tests для anchor identity, predicate ambiguity,
  field-to-graph bridge, Bee statuses и zero-score rejection.
