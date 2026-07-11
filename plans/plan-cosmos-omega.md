# Космос-ΩE: исполнимый план разработки

## 1. Результат проекта

Нужно расширить текущий SuperAI от детерминированного trace-first прототипа до системы, которая в ограниченной учебной среде умеет:

1. держать несколько структурированных гипотез;
2. выделять им ограниченный вычислительный бюджет;
3. строить различающиеся прогнозы;
4. выбирать проверку, которая лучше всего разделяет гипотезы;
5. создавать кандидат нового концепта из повторяющейся структуры;
6. проверять полезность концепта на скрытых задачах;
7. компилировать повторяющийся успешный маршрут в кандидат навыка;
8. повторно использовать концепт или навык и измеримо улучшать результат;
9. сохранять происхождение каждого решения и откатывать изменения.

Первая цель проекта — уровень **E3: проверенный новый концепт**. Морфогенез, свободное самоизменение и метаобучение не входят в первый релиз.

Главный критерий успеха:

> Система без заранее запрограммированного конкретного правила создаёт внутреннюю структуру, которая причинно и воспроизводимо улучшает решение новых задач.

---

## 2. Что уже есть в репозитории

Новый контур строится поверх существующего модульного монолита, а не как отдельная система.

| Возможность | Текущая реализация | Что нужно расширить |
|---|---|---|
| Событийный runtime | `superai/runtime.py`, durable `work_items`, retry, бюджеты | очередь событий активного графа и лимиты на ветви рассуждения |
| Трассировка | `superai/observability.py`, `trace_spans`, `domain_events` | причинные события гипотез, экспериментов, концептов и навыков |
| Улей | `superai/hive.py`, hot/warm entries, freeze/restore | активный подграф, доска гипотез и бюджет задачи |
| Космос | `superai/cosmos.py`, concepts, claims, provenance, retrieval | типизированные рёбра, причинные утверждения и версии графа |
| Атлас и план | `superai/execution.py`, capabilities, `ExecutionPlan` | несколько альтернативных планов и маршрутизация по гипотезам |
| Обучение | `superai/learning.py`, compost, skill lifecycle, holdout | кандидаты концептов, benchmark/ablation и безопасное продвижение |
| API и UI | `superai/api.py`, Vue inspector-страницы | наблюдение за конкуренцией гипотез и жизненным циклом концептов |

Существующие гарантии нельзя ослаблять:

- tenant/project access boundaries;
- provenance до исходного артефакта;
- идемпотентные команды;
- сохранение и восстановление после перезапуска;
- версионированные Pydantic-контракты;
- явный lifecycle кандидатов;
- holdout до активации навыка;
- trace для каждого значимого перехода.

---

## 3. Границы первой версии

### Входит в ΩE-MVP

- небольшая искусственная среда с повторяющимися скрытыми правилами;
- активный подграф в памяти на время одной задачи;
- 3–5 гипотез на задачу;
- фиксированный бюджет шагов, времени и размера графа;
- структурированные evidence/pro/con и прогнозы;
- внутренние детерминированные эксперименты;
- причинные рёбра со статусом и provenance;
- кандидат концепта и его lifecycle;
- компиляция одного типа навыка из повторяющихся планов;
- быстрый компост после задачи;
- baseline, holdout, ablation и replay;
- диагностический API и минимальный UI.

### Не входит

- распределённые сервисы и внешняя очередь сообщений;
- миллионы узлов и полный обход графа;
- спайковые нейроны или биологическая симуляция;
- произвольное выполнение сгенерированного кода;
- автоматическая модификация ядра, контрактов и правил безопасности;
- внешние реальные эксперименты без отдельного разрешения;
- автоматическое продвижение концепта или навыка в active;
- LLM как единственный судья качества;
- морфогенез, эволюция топологии и метаобучение до завершения E3.

---

## 4. Целевой вертикальный сценарий

Использовать одну малую среду, например задачи преобразования последовательностей или объектов по скрытым правилам. Dataset должен содержать train, validation и закрытый holdout.

Один запуск проходит следующий путь:

```text
TaskContract
    -> выбор/создание Улья
    -> bounded retrieval из Космоса
    -> построение активного подграфа
    -> создание 3–5 HypothesisRecord
    -> распределение бюджета
    -> прогнозы гипотез
    -> выбор ExperimentRecord
    -> выполнение проверки в Sandbox
    -> обновление evidence и confidence
    -> выбор ответа
    -> быстрый компост
    -> поиск повторяющегося подграфа
    -> ConceptCandidate
    -> validation + holdout + ablation
    -> promotion или rejection
    -> повторный запуск с новым концептом
```

Демонстрация считается успешной, если на новом наборе задач:

- версия с проверенным концептом лучше baseline по заранее выбранной метрике;
- при отключении концепта улучшение исчезает или существенно уменьшается;
- результат воспроизводится с тем же seed и версией графа;
- trace объясняет происхождение концепта и его влияние на решение;
- после перезапуска системы концепт сохраняет статус и поведение.

---

## 5. Архитектурные решения до начала реализации

Оформить решения отдельными ADR до изменения схемы данных.

### ADR-0004: активный граф поверх канонического Космоса

- SQLite остаётся каноническим хранилищем.
- Активный граф задачи — ограниченная in-memory проекция.
- Временные состояния не записываются как каноническое знание автоматически.
- В Космос попадают только структуры, прошедшие lifecycle проверки.
- Полный граф никогда не обходится для одной задачи.

### ADR-0005: lifecycle структур обучения

Единая модель состояний:

```text
candidate -> validating -> validated -> shadow -> active
    |             |           |           |
    +----------> rejected <----+-----------+
                                  |
                               rolled_back
```

Каждый переход является командой, проверяет текущую версию записи и создаёт domain event.

### ADR-0006: граница Песочницы

- В MVP Песочница исполняет только зарегистрированные детерминированные операции.
- Эксперимент не может повышать статус собственного результата до `observed` или `verified`.
- Симуляция, вычисление, пользовательское наблюдение и внешний источник имеют разные статусы.
- Любой внешний side effect по умолчанию запрещён.

### ADR-0007: benchmark и доказательство эмержентности

- Dataset manifest фиксирует seed, split, версию среды и метрики.
- Holdout недоступен генератору концептов и компилятору навыков.
- Сравниваются baseline, ΩE, random-search и budget-matched baseline.
- Для принятого концепта обязателен ablation-run.

---

## 6. Новые доменные контракты

Контракты добавить в `superai/contracts.py`. Все записи должны содержать `schema_version`, `tenant_id`, `project_id` или `AccessScope`, timestamps и version.

### 6.1 Граф

```text
GraphNode
    node_id
    node_type: concept | claim | observation | hypothesis | prediction | procedure
    content_ref
    status
    activation
    confidence
    novelty
    utility
    provenance_refs[]
    version

GraphEdge
    edge_id
    source_id
    target_id
    edge_type: semantic | evidence_for | evidence_against | causal | temporal | procedure
    weight
    confidence
    scope
    provenance_refs[]
    valid_from / valid_to
    version
```

### 6.2 Активный граф и бюджет

```text
ActiveGraphSnapshot
    task_id
    hive_id
    cosmos_version
    node_ids[]
    edge_ids[]
    frontier[]
    event_count
    budget_ledger
    random_seed

ResourceBudget
    max_steps
    max_wall_time_ms
    max_active_nodes
    max_active_edges
    max_hypotheses
    max_experiments
    exploration_share
```

### 6.3 Гипотезы и эксперименты

```text
HypothesisRecord
    hypothesis_id
    task_id
    family_id
    statement
    assumptions[]
    evidence_for[]
    evidence_against[]
    predictions[]
    confidence
    novelty
    allocated_budget
    spent_budget
    status: proposed | active | merged | falsified | selected | archived
    parent_ids[]

PredictionRecord
    prediction_id
    hypothesis_id
    experiment_input
    expected_output
    tolerance

ExperimentRecord
    experiment_id
    task_id
    competing_hypothesis_ids[]
    operation
    input
    expected_information_gain
    estimated_cost
    risk
    result_ref
    evidence_status
    status: proposed | approved | running | completed | failed
```

### 6.4 Концепты и навыки

```text
ConceptCandidate
    concept_id
    name
    definition_ref
    source_subgraph_refs[]
    positive_examples[]
    negative_examples[]
    train_task_ids[]
    holdout_manifest_ref
    metrics
    state
    rollback_target

ConceptEvaluation
    concept_id
    baseline_run_id
    treatment_run_id
    ablation_run_id
    quality_delta
    cost_delta
    transfer_delta
    accepted
```

Существующий `SkillManifest` расширять только при нехватке полей; не создавать параллельную сущность «фабрика». В коде и API термин **skill** является реализацией архитектурного понятия «фабрика».

---

## 7. Хранение и миграции

Добавлять таблицы итеративно в `superai/database.py`, с индексами по tenant/project/task/status.

Минимальный набор:

- `graph_edges` — типизированные и версионированные связи;
- `active_graph_snapshots` — replay/debug snapshot, не источник истины;
- `hypotheses`;
- `hypothesis_evidence`;
- `predictions`;
- `experiments`;
- `concept_candidates`;
- `concept_evaluations`;
- `benchmark_runs`.

Правила хранения:

- большие payload сохранять в object store, в SQLite держать `ArtifactRef`;
- evidence всегда ссылается на trace, claim, observation или result artifact;
- удаление source инвалидирует производные evidence и запускает переоценку зависимых концептов;
- optimistic version проверяется при каждом lifecycle transition;
- replay snapshot содержит `cosmos_version`, версии операций и seed;
- временные события активного графа можно удалять после retention-периода, итоговая причинная трасса остаётся.

---

## 8. Этапы разработки

Каждый этап заканчивается вертикальным результатом, тестами и наблюдаемым trace. Следующий этап начинается только после прохождения gate предыдущего.

### Этап Ω0. Baseline и воспроизводимость

**Цель:** получить контрольную точку, относительно которой измеряется любое «улучшение».

Задачи:

- [ ] Добавить `benchmarks/omega/` с manifest, train/validation/holdout split и фиксированными seeds.
- [ ] Сделать CLI-команду запуска benchmark без ΩE-механизмов.
- [ ] Сохранять `BenchmarkRun`: git revision, config hash, dataset version, seed, latency, cost, quality.
- [ ] Добавить replay завершённой задачи по trace и сохранённым версиям входов.
- [ ] Зафиксировать метрики текущего Atlas/Planner/Skill pipeline.
- [ ] Добавить UI/API чтения benchmark-run; создание запуска может пока оставаться CLI.

Основные файлы:

- `superai/contracts.py`
- `superai/database.py`
- `superai/service.py`
- новый `superai/benchmark.py`
- `superai/cli.py`
- `tests/integration/test_omega_baseline.py`

Gate Ω0:

- один benchmark дважды запускается с одинаковым seed;
- маршрут и результаты совпадают в пределах явно заданных допусков;
- baseline сохранён и доступен для последующих сравнений;
- holdout payload не читается кодом обучения.

### Этап Ω1. Событийный активный граф

**Цель:** обрабатывать только ограниченный подграф, а не весь Космос.

Задачи:

- [ ] Добавить `GraphEdge`, `ActiveGraphSnapshot`, `ResourceBudget`.
- [ ] Реализовать `ActiveGraph` как in-memory структуру в новом `superai/emergence/graph.py`.
- [ ] Загружать начальные узлы через существующий bounded retrieval.
- [ ] Реализовать priority queue событий: activate, propagate, inhibit, expire.
- [ ] Учитывать TTL, fatigue и повтор состояния для остановки циклов.
- [ ] Останавливать обработку по max_steps, wall time и размеру графа.
- [ ] Писать `GraphNodeActivated`, `GraphEdgeTraversed`, `GraphBudgetExhausted` в trace.
- [ ] Сохранять финальный snapshot для replay.

Тесты:

- unit: приоритет событий, TTL, inhibition, fatigue, budget exhaustion;
- integration: активируется ограниченная область Космоса;
- durability: snapshot читается после перезапуска;
- access: узел другого project не попадает в проекцию.

Gate Ω1:

- задача завершается при циклическом графе;
- число активных узлов и событий не превышает budget;
- одинаковый seed даёт одинаковую последовательность событий;
- trace объясняет, почему каждый узел был включён.

### Этап Ω2. Экосистема гипотез

**Цель:** заменить один заранее выбранный маршрут несколькими конкурирующими структурированными моделями.

Задачи:

- [ ] Добавить `HypothesisRecord`, `PredictionRecord` и evidence links.
- [ ] Реализовать `HypothesisBoard` в `superai/emergence/hypotheses.py`.
- [ ] Создавать 3–5 различающихся гипотез из зарегистрированных generators.
- [ ] Требовать от каждой гипотезы минимум один проверяемый prediction.
- [ ] Выделять exploitation и exploration budgets раздельно.
- [ ] Пересчитывать score по evidence, novelty, predictive value и cost.
- [ ] Поддержать merge без удаления родителей.
- [ ] Архивировать проигравшие варианты с причиной, а не удалять их.
- [ ] Сохранять доску гипотез в Hive entries.
- [ ] Показать доску, budget и evidence на `CosmosPage` или отдельной вкладке trace inspector.

Тесты:

- unit: score, budget allocation, merge, falsification;
- integration: как минимум две гипотезы дают разные predictions;
- regression: лидер не получает exploration reserve;
- UI: пользователь видит выбранную и отклонённые гипотезы с причинами.

Gate Ω2:

- система хранит модели, а не несколько текстовых ответов;
- каждая активная гипотеза имеет прогноз и provenance;
- слабая гипотеза теряет ресурс, но остаётся доступной в trace;
- общий расход не превышает task budget.

### Этап Ω3. Причинный слой и Песочница

**Цель:** различать наблюдение, предположение, симуляцию и причинную проверку.

Задачи:

- [ ] Добавить causal edge types, conditions, mechanism и applicability scope.
- [ ] Ввести evidence statuses: `simulated`, `computed`, `observed`, `source_backed`, `verified`.
- [ ] Реализовать реестр разрешённых sandbox operations.
- [ ] Оценивать ожидаемый information gain эксперимента.
- [ ] Выбирать дешёвый безопасный эксперимент, разделяющий лидирующие гипотезы.
- [ ] Применять результат как evidence, не как автоматическую истину.
- [ ] Запрещать незарегистрированные операции и внешние side effects.
- [ ] Добавить контрфактический запрос `intervene(X)` для учебной среды.

Основные файлы:

- новый `superai/emergence/causal.py`
- новый `superai/emergence/sandbox.py`
- `superai/execution.py`
- `superai/contracts.py`
- `superai/database.py`

Gate Ω3:

- выбранный эксперимент в среднем лучше random-check сокращает неопределённость;
- trace содержит ожидаемый и фактический information gain;
- симуляция не получает статус внешнего наблюдения;
- контрфактический результат не записывается как факт Космоса.

### Этап Ω4. Концептогенез — целевой релиз E3

**Цель:** создать и доказать полезность новой внутренней абстракции.

Задачи:

- [ ] Искать повторяющиеся подграфы только в успешных trace-backed задачах.
- [ ] Создавать `ConceptCandidate` с positive и negative examples.
- [ ] Считать compression gain, prediction gain, transfer gain и maintenance cost.
- [ ] Проверять кандидата сначала на validation, затем отдельным runner на holdout.
- [ ] Выполнять ablation: тот же запуск с отключённым кандидатом.
- [ ] Добавить lifecycle candidate → validated → shadow → active/rejected.
- [ ] Запрещать promotion без provenance, holdout и ablation.
- [ ] При отзыве source помечать зависимый концепт stale и исключать из активного retrieval.
- [ ] Добавить API списка, карточки и lifecycle концептов.
- [ ] Добавить UI сравнения baseline/treatment/ablation.

Тесты:

- contract: сериализация и неизвестный major schema version;
- integration: создание кандидата из повторяющегося подграфа;
- leakage: holdout не присутствует в input генератора;
- ablation: отключение полезного концепта ухудшает метрику;
- revocation: удаление source инвалидирует зависимый концепт;
- rollback: active-кандидат возвращается к прошлой версии.

Gate Ω4 / E3:

- хотя бы один концепт создан не прямым правилом разработчика;
- quality или sample-efficiency на holdout лучше budget-matched baseline;
- improvement сохраняется на новых примерах и после перезапуска;
- ablation подтверждает причинную роль концепта;
- все результаты воспроизводимы и видны в UI/trace.

После этого gate можно принимать решение о продолжении проекта.

### Этап Ω5. Компиляция фабрик/навыков — E5

**Цель:** повторяющийся успешный маршрут становится повторно используемым навыком.

Задачи:

- [ ] Расширить `ExperienceCompiler`, не создавать второй compiler.
- [ ] Выделять повторяющийся типизированный procedure graph из разных conversations.
- [ ] Формировать preconditions, input/output schemas и applicability scope.
- [ ] Использовать существующий `SkillManifest` lifecycle.
- [ ] Сравнивать candidate skill с исходным planner на holdout.
- [ ] Запускать сначала shadow, затем явную activation.
- [ ] Добавить rollback при regressions или отзыве provenance.
- [ ] Разрешить ограниченную parameter mutation только внутри Sandbox.

Gate Ω5 / E5:

- навык не был задан как готовая последовательность вручную;
- он снижает стоимость или повышает качество на holdout;
- переносится хотя бы на один новый вариант задачи;
- отключение навыка возвращает поведение к baseline;
- activation и rollback не нарушают текущие задачи.

### Этап Ω6. Долговременный компост и гомеостаз

**Цель:** удерживать полезные структуры и ограничивать рост графа.

Задачи:

- [ ] Добавить background consolidation как идемпотентные work items.
- [ ] Ввести decay по неиспользованию, стоимости и ошибкам.
- [ ] Защитить редкие знания с подтверждённым provenance от простого frequency decay.
- [ ] Объединять дубли с сохранением исходных идентификаторов и истории.
- [ ] Архивировать контрфактические и falsified структуры отдельно.
- [ ] Ввести лимиты роста на tenant/project и аварийный regulator.
- [ ] Добавить метрики graph growth, churn, reuse и stale dependencies.

Gate Ω6:

- серия задач не создаёт неограниченный рост hot/warm графа;
- полезный концепт переживает завершение Улья и консолидацию;
- удалённый source корректно инвалидирует всю цепочку производных структур;
- compaction воспроизводима и имеет dry-run.

### Этап Ω7. Зеркало, морфогенез и метаобучение — исследовательский трек

Начинать только после стабильных Ω4–Ω6 и отдельного ADR.

Допустимый первый объём:

- read-only анализ стоимости и ошибок;
- предложение изменения без автоматического применения;
- проверка предложения в изолированном benchmark-run;
- ручное одобрение promotion;
- автоматический rollback по заранее заданным порогам.

Запрещено на этом этапе без нового решения по безопасности:

- менять access control, schema validation, audit или rollback;
- изменять собственные критерии успеха;
- обучаться на holdout после его раскрытия;
- автоматически публиковать изменение в active;
- исполнять произвольный сгенерированный код.

---

## 9. API и UI, необходимые для разработки

API добавляется по мере появления этапов, а не заранее.

Минимальные read endpoints:

```text
GET /api/v1/tasks/{task_id}/active-graph
GET /api/v1/tasks/{task_id}/hypotheses
GET /api/v1/tasks/{task_id}/experiments
GET /api/v1/concepts
GET /api/v1/concepts/{concept_id}
GET /api/v1/benchmarks/{run_id}
```

Lifecycle endpoints должны быть командами с idempotency key и expected version:

```text
POST /api/v1/concepts/{concept_id}/validate
POST /api/v1/concepts/{concept_id}/shadow
POST /api/v1/concepts/{concept_id}/activate
POST /api/v1/concepts/{concept_id}/rollback
```

UI не является редактором графа. Для MVP он должен отвечать на пять вопросов:

1. Какие узлы активировались и почему?
2. Какие гипотезы конкурировали?
3. Куда ушёл бюджет?
4. Какой эксперимент изменил ranking?
5. Почему концепт или навык был принят либо отклонён?

Использовать существующие `TraceInspector.vue`, `CosmosPage.vue` и API store. Новую страницу создавать только если эти экраны становятся перегружены.

---

## 10. Метрики и правила приёмки

### Основные метрики

| Группа | Метрики |
|---|---|
| Качество | success rate, accuracy, повторные ошибки |
| Стоимость | p50/p95 latency, steps, active nodes, external calls, peak RAM |
| Гипотезы | diversity, prediction rate, falsification rate, calibration |
| Эксперименты | information gain, cost per eliminated hypothesis |
| Концепты | compression gain, transfer gain, ablation delta, lifetime |
| Навыки | reuse rate, quality delta, cost delta, rollback count |
| Граф | growth rate, active/cold ratio, duplicate rate, stale dependencies |
| Надёжность | replay success, recovery success, idempotent redelivery |

Численные пороги не придумывать заранее. На Ω0 измерить baseline, затем зафиксировать пороги в benchmark manifest до запуска treatment.

### Доказательство новой способности

Заявление E3 или E5 принимается только при наличии:

- frozen baseline;
- hidden holdout;
- одинакового бюджета сравниваемых вариантов;
- нескольких seeds;
- random-search baseline;
- ablation предполагаемой новой структуры;
- trace и provenance;
- повторного запуска после перезапуска процесса;
- заранее записанного критерия принятия.

---

## 11. Общий Definition of Done для каждой задачи

- контракт и миграция согласованы;
- команда идемпотентна, query не меняет состояние;
- tenant/project boundary проверена;
- тяжёлый payload вынесен в artifact store;
- domain events и trace не содержат секретов и полного пользовательского payload без необходимости;
- happy path, budget exhaustion, retry и invalid transition покрыты тестами;
- replay/durability проверены там, где появляется состояние;
- API DTO и frontend transport обновлены вместе;
- документация компонента и operations обновлены;
- benchmark не ухудшился сверх зафиксированного порога;
- для изменения есть rollback или безопасный способ его отключить.

---

## 12. Риски и обязательные защиты

| Риск | Защита | Проверка |
|---|---|---|
| Взрыв графа | budgets, TTL, decay, cold storage | stress test с циклическим графом |
| Зацикливание | state hash, fatigue, max steps | выполнение всегда завершается |
| Доминирование гипотезы | exploration reserve | альтернативная ветвь получает ресурс |
| Ложный концепт | negative examples, holdout, ablation | candidate отклоняется на shift dataset |
| Утечка holdout | отдельный loader и audit | leakage test |
| Имитация проверки | evidence statuses | simulated не становится verified |
| Катастрофическое изменение | shadow, versioning, rollback | regression вызывает rollback |
| Потеря provenance | dependency graph | source revocation инвалидирует производное |
| Межпроектная утечка | AccessScope на каждой записи | adversarial access tests |
| Невоспроизводимость | seed + version hashes | replay совпадает |

---

## 13. Порядок первых задач

Это стартовая очередь, которую можно переносить в issue tracker.

1. **OMEGA-001** — ADR активного графа и lifecycle обучения.
2. **OMEGA-002** — benchmark manifest и искусственная среда.
3. **OMEGA-003** — baseline runner и `BenchmarkRun`.
4. **OMEGA-004** — replay по seed, trace и версиям.
5. **OMEGA-005** — контракты `GraphEdge`, `ActiveGraphSnapshot`, `ResourceBudget`.
6. **OMEGA-006** — миграция `graph_edges` и `active_graph_snapshots`.
7. **OMEGA-007** — in-memory `ActiveGraph` и priority queue.
8. **OMEGA-008** — budget/TTL/fatigue/cycle detector.
9. **OMEGA-009** — интеграция активного графа в текущий task pipeline.
10. **OMEGA-010** — graph events и чтение snapshot через API.
11. **OMEGA-011** — контракты и таблицы гипотез/predictions/evidence.
12. **OMEGA-012** — `HypothesisBoard` и 3–5 generator strategies.
13. **OMEGA-013** — allocation exploitation/exploration budget.
14. **OMEGA-014** — UI доски гипотез и ledger бюджета.
15. **OMEGA-015** — sandbox operation registry.
16. **OMEGA-016** — experiment selection по information gain.
17. **OMEGA-017** — causal/evidence statuses и контрфактический тест.
18. **OMEGA-018** — поиск повторяющихся подграфов.
19. **OMEGA-019** — `ConceptCandidate` lifecycle.
20. **OMEGA-020** — validation/holdout/ablation runner.
21. **OMEGA-021** — UI карточки концепта и сравнения запусков.
22. **OMEGA-022** — end-to-end доказательство Gate Ω4/E3.

Первые четыре задачи не добавляют «интеллект», но без них последующее улучшение нельзя доказать.

---

## 14. Контрольная точка проекта

После **OMEGA-022** подготовить короткий отчёт:

- какой концепт возник;
- из каких trace и подграфов он получен;
- чем он отличается от заранее заданной capability;
- результаты baseline/treatment/random/ablation;
- перенос на holdout;
- стоимость улучшения;
- известные ограничения;
- решение: остановить, повторить эксперимент или переходить к Ω5.

До положительного Gate Ω4 не начинать автоматическое изменение топологии, правил обучения или ядра безопасности.
