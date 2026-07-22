# Проверка SuperAI

Полная регрессия:

```bash
python3 -m pytest -q
```

Полезные целевые прогоны:

```bash
python3 -m pytest -q tests/test_event_graph_v27.py
python3 -m pytest -q tests/test_dynamic_universes.py
```

`test_event_graph_v27.py` проверяет роль-независимый событийный граф: разбор
предложений, морфологические альтернативы, обязательные компоненты, активные и
пассивные конструкции, QueryGraph, продолжения диалога, retraction, batch и
проверку ответа.

`test_dynamic_universes.py` проверяет, что подтверждённое обучение создаёт
сущности и употребления, лексемы отделены от словоформ, а HTTP API
микровселенных не зависит от ролей графа.

Перед ручной проверкой HTTP-сервиса запускайте тесты с отдельной БД, как это
делает `tests/conftest.py`; рабочая база по умолчанию находится в
`.superai/state.sqlite` и несовместимая схема пересоздаётся.


## Чистое тестовое пространство

Для полного воспроизводимого сброса локальной тестовой памяти:

```bash
SUPERAI_ALLOW_TEST_RESET=true \
python3 -m server.v2.testing_reset \
  --scope full \
  --mode fresh-schema \
  --confirm "RESET TEST SPACE"
```

Для очистки только производного Semantic Field и микровселенных с сохранением
Event Graph:

```bash
SUPERAI_ALLOW_TEST_RESET=true \
python3 -m server.v2.testing_reset \
  --scope derived \
  --mode clear-data \
  --confirm "RESET TEST SPACE"
```

После выборочной очистки производный слой пересобирается явно через
`POST /api/v2/testing/rebuild-derived-space`. Технический rebuild не должен
увеличивать evidence, activation count или массу облаков.

Целевой reset-набор:

```bash
python3 -m pytest -q \
  tests/test_testing_reset.py \
  tests/api/test_testing_reset_api.py \
  tests/semantic_field/test_reset_invariants.py
```

## Regression: Auto retrieval и Field → Graph bridge

```bash
python -m pytest -q tests/test_scope_field_bridge_regressions.py
```

Набор проверяет:

- стабильный identity структурированного query anchor;
- широкий `EVENT_ATTACHMENT` для необученного question operator;
- выбор predicate hypothesis через graph event без lexical hardcode;
- переход semantic cloud → GraphEvidence;
- различие `SPATIAL_SUPPORT_FOUND` и `GAP_FILLED`;
- отказ Field Bee от zero-score результата.
