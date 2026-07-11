# 07. Переиспользование `master` и `alg2`

## Проверенные точки

- `master`: commit `f8fa87c`.
- `alg2`: commit `7593bb7`.
- текущая ветка `alg3`: новая архитектурная работа без перенесённого приложения.

Переносить ветки целиком нельзя: `alg2` удаляет большую часть модульной структуры и Vue-клиент `master`, заменяя их крупным экспериментальным движком и статическим web-приложением. Нужна выборочная миграция через новые интерфейсы.

## Матрица решений

| Источник | Наработка | Решение | Условие переноса |
| --- | --- | --- | --- |
| `master/web` | Vue 3/Vite/TS/Router/Pinia/SCSS setup | переносить основой клиента | обновить маршруты и DTO под `/api/v1` |
| `master/web` | `AppShell`, общая навигация | адаптировать | не сохранять старую предметную структуру страниц |
| `master/web` | Cytoscape `GraphViewer` и `NodeInspector` | переносить как feature | отвязать от старых `GraphNode/Edge`, добавить large-graph limits |
| `master/web` | Vitest и Playwright tests | переносить setup и полезные сценарии | заменить ожидания старых страниц новыми fixtures |
| `master/server` | FastAPI app factory, CORS, static dist | адаптировать | route handlers сделать тонкими, новый composition root |
| `master/server` | job registry/service | использовать как spike/characterization source | durable queue реализовать отдельно; in-memory job state не считать надёжным runtime |
| `master/core` | нормализация, graph dataclasses | переносить после тестов | заменить `source: str` на полноценный provenance/access model |
| `master` | старые decoder/understanding/ACO | оставить экспериментальными adapters | включать только при сравнении с простой baseline |
| `alg2` | dataset preprocessing, hierarchy extraction, ignore rules | извлечь в import adapter | убрать узкие эвристики из общего интерфейса, сохранить tests |
| `alg2` | SQLite checkpoint serialization helpers | использовать как исследовательский материал | не делать checkpoint канонической моделью нового хранилища |
| `alg2` | vector batching/cache/index идеи | перенести отдельными оптимизациями | только после профилирования и characterization tests |
| `alg2` | hierarchy/backpack/drill-down UX | использовать как прототип навигации | сопоставить с Hive snapshot/Cosmos views, не переносить терминологию автоматически |
| `alg2` | 3500+ строк `SemanticEngine` | не переносить целиком | выделять только чистые функции и adapters |
| `alg2/web` | статический HTML/JS/CSS и vendored Cytoscape | не брать за основу | допустим только как визуальный reference |
| обе ветки | `__pycache__`, checkpoints, generated/build data | не переносить | добавить правила `.gitignore` |

## Что взять из `master` первым

Рекомендуемый минимальный набор:

```text
web/package.json
web/tsconfig.json
web/vite.config.ts
web/vitest.config.ts
web/playwright.config.ts
web/src/main.ts
web/src/app/App.vue
web/src/app/router.ts
web/src/widgets/app-shell/
web/src/shared/styles/
web/src/features/graph-viewer/
web/src/features/node-inspector/
```

Также использовать идеи из:

- `web/src/shared/api/client.ts` — общий HTTP wrapper;
- `web/src/shared/api/types.ts` — перечень нужных представлений, но не сами старые типы;
- `semantic_ants/server/app.py` — app factory и static mount;
- `semantic_ants/server/schemas.py` — Pydantic boundary;
- `semantic_ants/server/jobs.py` — только semantics статусов и ошибок.

Старый `runtime store` лучше разделить по use cases: tasks, traces, hive snapshots, cosmos views. Один глобальный store быстро свяжет все экраны.

## Что взять из `alg2`

### Preprocess adapter

Полезны:

- обход каталога с исключениями вроде `node_modules`;
- извлечение иерархии из путей и Markdown headings;
- потоковая обработка JSONL;
- фильтры мусора и лимиты;
- тесты на accepted/skipped статистику.

Переносить как `SourcePreprocessor` с типизированным отчётом. Языковые и dataset-specific эвристики должны настраиваться профилем импорта.

### Иерархические представления

Идеи `hypernodes`, drill-down и lazy graph payload полезны для:

- представления структуры источника;
- локальной навигации по Космосу;
- ограничения размера ответа API;
- загрузки тяжёлого графа только по запросу.

Не считать иерархию файлов универсальной семантической иерархией. Она остаётся одним видом provenance/structure edges.

### Векторные оптимизации

Batch embedding, матричный cosine search и кэш норм могут стать сменным `VectorCandidateIndex`. Канонические concepts/claims не должны жить только внутри массивов `SemanticEngine`.

### Checkpoint

Тесты нормализации старого state и SQLite serialization пригодны как миграционный spike. Новый runtime должен хранить отдельные агрегаты, артефакты и версии схем, а не один общий `Checkpoint` со слабо типизированным `meta`.

## Порядок безопасной миграции

1. Создать список файлов и лицензий зависимостей.
2. Зафиксировать characterization tests исходного фрагмента на его ветке/во временном worktree.
3. Сформулировать новый целевой интерфейс.
4. Перенести минимальный код без старой глобальной модели состояния.
5. Прогнать старые characterization и новые contract tests.
6. Подключить feature flag или adapter.
7. Сравнить с простой новой реализацией по correctness и стоимости.
8. Удалить adapter, если выигрыша нет.

Не делать merge `master` или `alg2` в `alg3` и не cherry-pick крупных архитектурных commits: это вернёт удалённые данные, checkpoints, `__pycache__` и конфликтующие реализации web/backend.

## Spikes перед основным переносом

### Spike A — клиент `master`

- собрать зависимости;
- запустить unit и e2e smoke;
- заменить один старый API type новым `TraceView`;
- оценить объём изменений shell/router/styles.

Решение: переносить, если setup собирается на текущей версии Node и не требует массового переписывания.

### Spike B — preprocess `alg2`

- прогнать на fixture-каталоге;
- измерить потоковую память;
- проверить Unicode, binary files, symlinks и большие файлы;
- сравнить структуру результата с новым `SourceArtifact`.

Решение: переносить чистый scanner/parser, если он проходит новый контракт без зависимости от `SemanticEngine`.

### Spike C — graph viewer

- отобразить mock Cosmos graph и execution graph через один базовый canvas adapter;
- проверить 100/1000/5000 элементов;
- определить server-side limits и lazy expansion.

## Gate legacy migration

- Каждый перенесённый фрагмент имеет владельца и целевой интерфейс.
- Ни один новый пакет не импортирует `SemanticEngine` как глобальный контейнер архитектуры.
- Vue-клиент собирается и тестируется независимо от старого backend.
- Старые checkpoints и `__pycache__` не попадают в новую историю.
- Для каждого экспериментального алгоритма есть простая baseline и измеримое основание оставить его.
