# SuperAI V3.0

Локальная русскоязычная пространственно-графовая система. Непрерывное
семантическое пространство организует понятия, контекстные проекции и скрытые
измерения, а событийный граф хранит проверяемые наблюдения, provenance и строгие
ограничения ответа. Это не LLM и не готовая энциклопедия: фактические ответы
разрешены только при поддержке обученных подтверждённых источников.

## Текущий контур

```text
текст
→ морфологические гипотезы и именные группы
→ подтверждённое событие с неназванными участниками
→ проекция события в микровселенные и Semantic Field
→ QueryGraph
→ активация поля + извлечение доказательств из Event Graph
→ bounded workspace / BindingConfiguration / resonance
→ проверенный AnswerStructure с provenance и epistemic mode
```

Event Graph является evidential source of truth: он фиксирует, что именно
наблюдалось. Semantic Field и микровселенные являются semantic state: они
организуют близость, ассоциации, контекст и маршруты поиска, но не имеют права
самостоятельно создавать статус `OBSERVED`.

## Запуск

Требуется Python 3.9+.

```bash
python3 -m pip install -e ".[dev]"
python3 -m uvicorn server.server:app --reload
```

Сервис доступен на `http://127.0.0.1:8000`; интерактивная схема FastAPI —
`/docs`.

## Документация

- [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) — границы слоёв, поток данных и
  модель микровселенных.
- [API_V2.md](API_V2.md) — действующие HTTP-эндпоинты и важные параметры.
- [MODEL_INVARIANTS.md](MODEL_INVARIANTS.md) — инварианты, на которые можно
  опираться при изменении кода.
- [TESTING.md](TESTING.md) — состав регрессии и команды проверки.

## Основные API-группы

- `/api/v2/training/*` — обучение, staging, подтверждение, отзыв и batch.
- `/api/v2/hives/*`, `/api/v2/graphs/*` — диалоговый поиск и инспекция
  `QueryGraph`.
- `/api/universes/*`, `/api/dimensions/*`, `/api/entities/*` — чтение
  микровселенных и их измерений.
- `/api/reset` — полное удаление текущей памяти; `/api/export/memory` — её
  переносимый JSON-снимок.

## Хранилище и совместимость

SQLite-файл по умолчанию: `.superai/state.sqlite`. Текущая схема имеет
`schema_version = 38` и `migration_version = fresh-v3.0-spatial-reset`.
Несовместимая база намеренно удаляется и создаётся заново: legacy-таблицы,
ролевые аннотации и backfill не поддерживаются. Перед обновлением сохраните
нужные данные через `GET /api/export/memory`.

## Компактный воспроизводимый эксперимент

Сценарий из `PLAN.md` запускается одной командой:

```bash
python3 -m server.v2.experiment \
  --seed 1729 \
  --output .superai/experiment-report.json
```

Он пересоздаёт чистую схему, выполняет train 48 → holdout 16 →
continual 16 → повторный holdout → blind regression → smoke 25/50/100
и сохраняет конфигурационный hash, порядок обучения, границы batch и
итоговый JSON-отчёт. Smoke ограничен 100 событиями и не является
доказательством масштабируемости на больших корпусах.

`POST /api/reset` остаётся admin-alias полного сброса. Для ручного тестирования
используйте `POST /api/v2/testing/reset` или CLI:

```bash
SUPERAI_ALLOW_TEST_RESET=true python -m server.v2.testing_reset \
  --scope full --mode fresh-schema --confirm "RESET TEST SPACE"
```

Полный сброс пересоздаёт SQLite-файл, WAL/SHM, схему, реестр микровселенных и
process-local индексы. Выборочная очистка производного пространства сохраняет
Event Graph и требует отдельного `rebuild-derived-space`.

## Ограничения

Языковой контур ориентирован на русский текст и покрывается регрессией на
конкретных конструкциях, а не корпусной метрикой. Размерные и семантические
свойства микровселенных являются эвристической производной наблюдений, а не
доказанной моделью реального мира.

## Optional acceleration

For development with the optional native acceleration backends, install:

```bash
pip install -e ".[dev,acceleration]"
```

`SUPERAI_ACCELERATION_MODE` accepts `auto` (default), `native`, or `python`.
`auto` records an unavailable optional backend and continues with the exact
NumPy/SQLite implementation; `native` fails during startup when a required
backend is unavailable; and
`python` is the deterministic reference path used by parity tests.
