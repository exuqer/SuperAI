# План следующей итерации SuperAI

## Улучшение микровселенных, скрытых измерений, роёв и множественных GAP без большого датасета

## 1. Цель итерации

Не пытаться пока доказать масштабируемость на тысячах событий.

Цель этапа — экспериментально проверить четыре ключевые гипотезы:

```text
1. Скрытые измерения возникают не из одиночных служебных признаков,
   а из устойчивых семантико-структурных закономерностей.

2. Найденные измерения переносятся на ранее не встречавшиеся лексемы.

3. Измерения сохраняют функцию и геометрию после последующего дообучения.

4. Рои реально используют пространства и измерения для поиска,
   а QueryGraph строго проверяет найденные результаты.
```

Дополнительная обязательная цель:

```text
полностью завершить переход
selected_binding → selected_bindings
и реализовать согласованное заполнение нескольких GAP.
```

Не заявлять:

* доказанную масштабируемость;
* устойчивость на больших корпусах;
* универсальное понимание языка;
* превосходство над трансформерами.

---

# 2. Ограничения итерации

Использовать компактный эксперимент:

```text
Train:             48 примеров
Transfer holdout:  16 примеров
Continual learning:16 примеров
Blind regression:  небольшой закрытый набор
Smoke performance: 25 / 50 / 100 событий
```

Не включать:

```text
1000+ событий
массовую генерацию синтетического корпуса
сложные миграции БД
распределённое выполнение
GPU-оптимизацию
недетерминированные рои
```

Все лимиты должны задаваться конфигурацией.

---

# 3. Воспроизводимость эксперимента

Добавить единый сценарий:

```text
reset database
→ create clean schema
→ train 48
→ consolidate
→ evaluate holdout 16
→ train continual 16
→ consolidate
→ repeat holdout evaluation
→ run blind regression
→ run smoke 25/50/100
→ export report
```

Для каждого запуска сохранять:

```text
experiment_id
dataset_version
dataset_split
schema_version
pipeline_versions
configuration_hash
random_seed
training_order
batch_boundaries
started_at
completed_at
```

При одинаковых:

```text
dataset_version
configuration_hash
random_seed
```

результат должен быть детерминированным либо отличаться только в пределах явно заданного допуска.

---

# 4. Подготовка компактного датасета

## 4.1. Train — 48 примеров

Train должен содержать несколько тематических областей:

```text
еда и предметы
роботы и инструменты
животные и движение
дроны и доставка
```

В каждой области нужны повторяющиеся структурные закономерности:

```text
X поднял Y
X дал Y Z
X положил Y в Z
X разрезал Y инструментом
Y находится в/на/под Z
после события A произошло событие B
```

При этом:

* не размечать семантические роли;
* не добавлять `agent`, `object`, `location`;
* разрешается техническая разметка train/holdout и ожидаемых ответов;
* ожидаемые ответы не должны попадать в память как дополнительные факты.

## 4.2. Transfer holdout — 16 примеров

Holdout должен содержать:

* ранее не встречавшиеся леммы;
* знакомые структурные конфигурации;
* изменённый порядок слов;
* другие словоформы;
* хотя бы одну новую тему;
* активные и пассивные конструкции;
* вопросительные формулировки, отличающиеся от train.

Пример переноса:

```text
Train:
Механик дал роботу болт.
Девочка дала мальчику книгу.

Holdout:
Оператор передал дрону посылку.
```

Система не обязана считать `дать` и `передать` одной лексемой, но должна иметь возможность обнаружить сходную структурную конфигурацию через облака и измерения.

## 4.3. Continual learning — 16 примеров

Набор должен включать:

```text
подтверждение существующих закономерностей
новые лексемы
контрпримеры
изменённый порядок слов
пассив
одно изменение состояния объекта
одну потенциальную полисемию
один пример для объединения похожих облаков
один пример, расширяющий границу облака
```

## 4.4. Blind regression

Создать небольшой набор, который:

* не используется при настройке порогов;
* не показывается алгоритму discovery;
* запускается только после завершения итерации.

Он нужен для проверки, что система не была подогнана под известные 80 примеров.

---

# 5. Разделение признаков

Создать явную типизацию признаков.

## 5.1. SemanticStructuralFeature

Семантико-структурные признаки:

```text
contextual lexeme distributions
predicate concept
construction cluster
anonymous local slot prototype
clause neighbourhood
event neighbourhood
cross-universe transition
co-occurrence across different sources
substitution behaviour
retrieval co-activation
scene-level temporal relation
```

Эти признаки разрешено использовать как первичную основу latent discovery.

## 5.2. ControlFeature

Контрольные и служебные признаки:

```text
case
gender
number
tense
person
surface word form
absolute token position
before/after predicate
sentence index
word length
capitalization
character shape
```

Они:

* сохраняются;
* участвуют в GraphMatcher;
* используются для морфологической совместимости;
* используются как контрольные переменные;
* не должны самостоятельно создавать активное скрытое измерение.

## 5.3. ControlledStructuralCandidate

Не запрещать служебным признакам навсегда участвовать в найденной закономерности.

ControlFeature может войти в состав кандидата только при выполнении условий:

```text
закономерность переносится между разными лексемами;
закономерность существует в нескольких конструкциях;
есть положительный holdout retrieval gain;
результат не исчезает после перестановки слов;
кандидат не сводится к одному окончанию или одному падежу;
есть семантико-структурные признаки, поддерживающие тот же паттерн.
```

Пример допустимого открытия:

```text
не «винительный падеж» как измерение,

а устойчивая конфигурация:
морфологический профиль
+ поведение при замене
+ позиционная вариативность
+ связь с определённым кластером событий.
```

---

# 6. Модель скрытых измерений

## 6.1. Сущность LatentDimension

```text
LatentDimension:
  id
  canonical_dimension_id
  revision
  universe_id
  scope
  representation_type
  status

  semantic_basis
  control_features_used
  core_entities
  peripheral_entities
  projection_parameters

  entity_support
  source_support
  domain_support
  train_support
  holdout_support

  stability_point_estimate
  stability_lower_bound
  holdout_retrieval_gain
  shadow_retrieval_gain

  retrieval_contribution_count
  graph_admitted_contribution_count
  validated_answer_contribution_count
  usage_count

  created_at
  activated_at
  last_updated_at
```

## 6.2. Жизненный цикл

```text
candidate
→ probation
→ active
→ shared
```

Дополнительные состояния:

```text
weak
merged
split
pruned
frozen
```

## 6.3. Условия candidate → probation

Кандидат переводится в `probation`, если:

```text
entity_support >= configured minimum
source_support >= configured minimum
domain_support >= configured minimum
не является копией существующего измерения
basis содержит семантико-структурную информацию
есть устойчивое ядро
```

Пример стартовой конфигурации:

```text
minimum_entity_support = 6
minimum_source_support = 4
minimum_domain_support = 2
```

Все значения вынести в конфигурацию.

## 6.4. Условия probation → active

Активировать только при одновременном выполнении:

```text
stability_point_estimate >= 0.75
stability_lower_bound >= configured threshold
holdout_retrieval_gain > 0
entity diversity sufficient
source diversity sufficient
domain diversity sufficient
dimension is not dominated by one ControlFeature
no near-duplicate active dimension exists
```

Низкое число подтверждений не должно компенсироваться высоким средним stability.

---

# 7. Lineage измерений

Не требовать сохранения неизменного технического ID после дообучения.

Добавить:

```text
DimensionLineage:
  canonical_dimension_id
  current_revision_id
  parent_dimension_ids
  merged_from
  split_from
  replaced_by
  lineage_reason
```

После Continual learning проверять:

```text
core_overlap
projection_rank_correlation
centroid_drift
retrieval_set_overlap
utility_delta
applicability_overlap
```

Считать измерение сохранившимся, если его функция и геометрия узнаваемы, даже если:

* изменилось ядро;
* появилась новая revision;
* произошло контролируемое объединение;
* произошло обоснованное разделение.

---

# 8. Shadow evaluation измерений

Candidate и probation не должны напрямую управлять итоговым ответом.

Для них выполнять параллельный теневой поиск:

```text
baseline retrieval
shadow dimensional retrieval
```

Теневой результат:

* не передаётся генератору;
* не влияет на ответ;
* сравнивается с базовым поиском;
* проходит тот же GraphMatcher.

Сохранять:

```text
shadow_candidate_events
shadow_graph_admitted_events
shadow_correct_event_rank
shadow_retrieval_gain
shadow_false_positive_count
```

Это предотвращает замкнутый круг:

```text
неактивное измерение
→ не используется
→ не получает utility
→ никогда не активируется
```

---

# 9. Utility измерений

Разделить utility на уровни доказательности.

```text
projection_usage_count
retrieval_contribution_count
graph_admitted_contribution_count
validated_answer_contribution_count
```

Основной utility увеличивается только когда:

```text
измерение участвовало в поиске;
найденное событие прошло QueryGraph;
binding или BindingConfiguration прошли validation;
ответ оказался корректным по тесту.
```

Shadow utility сохраняется отдельно и не должна подменять validated utility.

Не повышать utility только за:

* посещение пчелой;
* попадание события в предварительный top-K;
* близость сущностей;
* совпадение служебного признака.

---

# 10. Облака

## 10.1. Дедупликация

Перед созданием нового облака сравнивать его с существующими по:

```text
core overlap
projection similarity
membership rank correlation
context distribution similarity
retrieval contribution overlap
```

При почти полном совпадении:

```text
merge
```

При допустимом частичном пересечении:

```text
DimensionRelation:
  type: overlapping
  source_dimension_id
  target_dimension_id
  overlap_score
  shared_core_entities
```

Не сливать облака только из-за близких центроидов.

## 10.2. Лимиты

Добавить конфигурацию:

```text
max_candidate_dimensions_per_universe
max_probation_dimensions_per_universe
max_active_dimensions_per_universe
max_clouds_per_dimension
max_core_entities_per_cloud
max_peripheral_entities_per_cloud
```

При достижении лимита:

```text
сначала merge
затем prune weak
затем refuse new candidate
```

Причина должна попадать в trace.

---

# 11. Полный переход на selected_bindings

## 11.1. Каноническая модель

```text
selected_bindings: Binding[]
```

Использовать её во всех компонентах:

```text
QueryResult
DialogueTurn
HiveState
TrainingEpisode
ConstructionLearning
SlotCompatibilityLearning
AnswerGenerator
API
Frontend
Export
Validation
```

## 11.2. Устаревшее поле

```text
selected_binding
```

оставить временно только как deprecated alias:

```text
selected_binding =
  selected_bindings[0]
  when selected_bindings.length > 0
```

Запрещено использовать alias:

* в поиске;
* в обучении;
* в продолжениях;
* в validation;
* в генерации;
* в сохранении состояния.

Добавить hardcode audit, который ищет внутренние обращения к `selected_binding`.

---

# 12. Множественные GAP

## 12.1. QueryGraph

Заменить одиночные поля основным представлением:

```text
question_operators: QuestionOperator[]
target_gaps: GapNode[]
implicit_gaps: GapNode[]
```

Для:

```text
Кто, кому и что дал?
```

создавать:

```text
target_gaps.length == 3
```

Где каждый GAP имеет:

```text
gap_id
surface
token_indices
morphology_hypotheses
question_signature
required
coordination_group_id
```

## 12.2. BindingConfiguration

Главной единицей ранжирования для multi-GAP становится не Binding, а:

```text
BindingConfiguration:
  configuration_id
  query_graph_id
  event_id
  bindings_by_gap
  all_required_gaps_bound
  distinct_node_count
  configuration_score
  graph_validation
  status
```

## 12.3. Ограничения

Для трёх GAP требовать:

```text
три binding;
три разных participant node;
одно event_id;
полное соответствие target_gaps;
прохождение GraphMatcher;
прохождение обратной surface validation.
```

Возвращать `UNRESOLVED`, если:

* заполнена только часть GAP;
* один узел используется для двух GAP;
* bindings относятся к разным событиям;
* один GAP заполнен fallback-значением без подтверждения;
* отсутствует полная обратная проверка.

## 12.4. Обучение

Обучать construction и slot compatibility отдельно по каждому binding, но связывать их через общий:

```text
configuration_id
event_id
query_graph_id
```

Не считать bindings одного события независимыми источниками.

---

# 13. Продолжения диалога

Продолжение может наследовать event anchor только от:

```text
предыдущего QueryResult,
имеющего валидную BindingConfiguration
или согласованный selected_bindings.
```

Для ротации GAP хранить происхождение узлов:

```text
EXPLICIT_CURRENT
EXPLICIT_INHERITED
RESOLVED_PREVIOUS_TARGET
INFERRED_CONTEXT
EVENT_ANCHOR
```

Алгоритм:

```text
1. Взять предыдущий валидный event anchor.
2. Определить новый question operator.
3. Найти участника, совместимого с новым GAP.
4. Освободить его из known_nodes.
5. Сохранить остальные bindings как ограничения.
6. Перезапустить binding внутри anchored event.
7. Только после неудачи разрешить глобальный поиск.
```

Контекст должен ослабляться после:

* новой темы;
* приветствия;
* команды начать новый разговор;
* нескольких нерелевантных ходов;
* явно созданной новой сессии.

---

# 14. Архитектура роёв

## 14.1. Отдельный рой на каждый GAP

Для каждого `requested GAP` запускать детерминированный `GapSwarm`.

```text
GapSwarm:
  swarm_run_id
  gap_id
  query_graph_id
  deterministic_seed
  status
  termination_reason
  budget
```

## 14.2. Типы пчёл

```text
Scout
Worker
Assembly
Observer
```

### Scout

Ищет начальные опоры:

```text
predicate projections
known-node projections
GAP signature
exclusions
event indexes
```

### Worker

Расширяет соседей:

```text
entity clouds
dimension projections
usage neighbourhood
event transitions
```

### Assembly

Перемещает evidence между микровселенными:

```text
Words
→ Word Forms when morphology is required
→ Usages
→ Clauses
→ Events
→ Scenes or Abstractions
```

### Observer

Объединяет результаты:

```text
deduplicates events
aggregates evidence
tracks conflicts
reports candidate events
```

## 14.3. Seeds

Формировать без фиксированных ролей:

```text
predicate concept
known node concepts
GAP signature
QueryGraph exclusions
active dimension projections
event anchor
```

Не использовать маршруты:

```text
кто → agent
что → object
где → location
```

---

# 15. Координатор между GAP-роями

Добавить:

```text
JointBindingCoordinator
```

Его задача:

```text
1. Получить candidate events от каждого GapSwarm.
2. Пересечь или объединить event_id.
3. Найти события, способные заполнить все GAP.
4. Построить BindingConfiguration.
5. Проверить уникальность участников.
6. Передать конфигурацию GraphMatcher.
7. Вернуть ranked configurations.
```

Пример недопустимой сборки:

```text
кто  → механик из event A
кому → дрон из event B
что  → ключ из event C
```

Такая конфигурация должна быть отброшена до генерации ответа.

---

# 16. Бюджеты роёв

Конфигурация на один GAP:

```text
max_bees = 8
max_rounds = 4
max_vertical_transitions = 4
max_nectar_packets = 128
```

Дополнительные лимиты:

```text
max_candidate_events_per_bee
max_candidate_events_per_swarm
max_dimension_projections_per_round
max_index_hits_per_seed
max_graph_match_attempts
```

Завершение:

```text
STABLE_CANDIDATES
BUDGET_EXHAUSTED
NO_SEEDS
NO_CANDIDATES
GRAPH_ADMITTED_RESULT
INDEX_FALLBACK_COMPLETED
CANCELLED
```

`STABLE_CANDIDATES` означает два последовательных раунда без изменения top candidate set.

---

# 17. Swarm trace

Сохранять:

```text
SwarmRun
BeeMission
BeeStep
NectarPacket
UniverseTransition
CandidateEventObservation
```

## SwarmRun

```text
id
query_graph_id
gap_id
status
termination_reason
bee_count
active_bee_count
round_count
packet_count
transition_count
events_considered
events_returned
retrieval_mode
started_at
completed_at
```

## BeeMission

```text
bee_id
bee_type
mission_type
seed
budget
visited_universes
successful
termination_reason
```

## NectarPacket

```text
packet_id
source_universe
target_universe
source_entity_ids
dimension_ids
event_ids
evidence_weight
provenance
```

---

# 18. Значения bee_count

Не требовать пчёл во всех микровселенных.

Для каждой Universe возвращать:

```text
visited_in_last_query
last_query_bee_count
active_bee_count
successful_bee_count
terminated_bee_count
fallback_bee_count
```

Критерий:

```text
для каждой микровселенной,
которая присутствует в QueryPlan
и реально посещена роем,
last_query_bee_count > 0.
```

Для непосещённой микровселенной корректно:

```text
visited_in_last_query = false
last_query_bee_count = 0
```

---

# 19. Режимы поиска

Добавить:

```text
retrieval_mode:
  SWARM_DIMENSIONAL
  SWARM_MIXED
  INDEX_FALLBACK
  DIRECT_EVENT_LOOKUP
```

Сохранять:

```text
dimension_evidence_ratio
index_evidence_ratio
fallback_reason
active_dimensions_used
candidate_dimensions_shadowed
```

Критерий готовности:

```text
хотя бы один обязательный тест должен быть решён
в режиме SWARM_DIMENSIONAL
без INDEX_FALLBACK как основного источника.
```

Fallback разрешён, если:

* активных проекций нет;
* проекции пусты;
* рой исчерпал бюджет;
* discovery ещё не завершён.

Но причина должна быть видна в trace.

---

# 20. QueryPlan

Перед запуском роя строить:

```text
QueryPlan:
  query_graph_id
  requested_gap_ids
  event_anchor_id
  seed_entities
  seed_predicates
  required_universes
  optional_universes
  enabled_dimensions
  budgets
  fallback_policy
```

QueryPlan не должен заранее указывать семантические роли.

Пример маршрута:

```text
Words
→ Usages
→ Events
```

При морфологической неоднозначности:

```text
Word Forms
→ Words
→ Usages
→ Events
```

При временном вопросе:

```text
Events
→ Scenes
```

---

# 21. GraphMatcher остаётся строгим фильтром

Рои только предлагают события.

GraphMatcher проверяет:

```text
predicate compatibility
known node retention
all required GAP bound
distinct participants where required
same event for multi-GAP
exclusions
temporal constraints
negation
event anchor
morphological compatibility
```

Высокий swarm score не может отменить структурное противоречие.

Результат роя может быть:

```text
FOUND_AND_ADMITTED
FOUND_BUT_REJECTED
NOT_FOUND
FALLBACK_FOUND_AND_ADMITTED
```

---

# 22. Производительность

## 22.1. Инкрементальные обновления

После одного нового источника:

* обновлять существующие Usage;
* обновлять существующие projections;
* обновлять массы и локальную статистику;
* не выполнять полный discovery всех измерений.

Полный discovery запускать:

```text
после завершения batch;
после контрольной точки;
по ручной команде;
при превышении накопленного изменения.
```

## 22.2. Индексы

Добавить индексы по:

```text
predicate concept
event participant concept
event source
projection dimension/entity
universe transition source/target
word-form → lexeme
usage → event
scene → event
swarm run → query
binding → gap
binding configuration → event
```

## 22.3. Метрики запроса

В trace сохранять:

```text
events_total
events_indexed
events_scanned
index_hits
candidate_events
candidate_bindings
binding_configurations
dimension_projections_read
nectar_packets_created
bee_steps
universe_transitions
graph_match_attempts
database_queries
elapsed_ms
```

---

# 23. Smoke-тест 25 / 50 / 100

Создать синтетический набор похожих, но различимых событий.

Запустить одинаковый набор запросов на:

```text
25 событиях
50 событиях
100 событиях
```

Сравнивать:

```text
elapsed_ms
events_scanned
database_queries
bee_steps
nectar_packets
graph_match_attempts
memory usage
```

Не требовать доказательства строгой асимптотики.

Считать smoke-тест проваленным при явном признаке квадратичной регрессии, например:

```text
объём вырос в 2 раза,
а events_scanned или graph_match_attempts
стабильно выросли примерно в 4 раза
без архитектурной причины.
```

Масштабирование на тысячи событий оставить отдельной задачей.

---

# 24. База данных

При изменении schema version:

```text
удалить старую БД
создать чистую схему
повторно запустить эксперимент
```

Не реализовывать:

```text
migration
backfill
совместимость со старым schema
```

Добавить таблицы:

```text
dimension_history
dimension_lineage
dimension_evaluations
shadow_retrieval_runs
swarm_runs
bee_missions
bee_steps
nectar_packets
universe_transitions
binding_configurations
experiment_runs
experiment_metrics
```

---

# 25. API

## Query API

Возвращать обязательно:

```text
query_graph
selected_bindings
binding_configuration
swarm
validation
answer
```

## `/bindings`

Возвращает только канонический массив:

```text
selected_bindings
candidate_bindings
binding_configurations
```

## Universe summary

Добавить:

```text
last_swarm_run_id
visited_in_last_query
last_query_bee_count
active_bee_count
successful_bee_count
retrieval_mode
```

## Dimension detail

Добавить:

```text
train_support
holdout_support
continual_support
entity_support
source_support
domain_support
stability components
stability lower bound
retrieval gain
shadow retrieval gain
validated utility
lineage
merge and overlap relations
```

---

# 26. Frontend

## 26.1. Swarm Inspector

Показывать:

```text
GapSwarm по каждому GAP
тип пчелы
миссия
маршрут
посещённые микровселенные
использованные измерения
созданные nectar packets
найденные события
причина завершения
```

## 26.2. Multi-GAP Inspector

Для запроса:

```text
Кто, кому и что дал?
```

показывать:

```text
GAP 1 → Механик
GAP 2 → роботу
GAP 3 → болт
Общее событие → Механик дал роботу болт
Configuration validation → passed
```

## 26.3. Dimension Inspector

Показывать:

```text
status
revision
lineage
semantic basis
control features
train core
holdout projections
continual projections
stability
retrieval gain
shadow gain
validated utility
merge/overlap relations
```

## 26.4. Retrieval mode

В интерфейсе явно показывать:

```text
Dimensional swarm
Mixed swarm
Index fallback
Direct lookup
```

Не маскировать fallback под полноценный dimensional reasoning.

---

# 27. Обязательные функциональные тесты

## 27.1. Одиночные GAP

```text
Кто поднял ключ?
→ Робот.

Что поднял робот?
→ Ключ.

Кому механик дал болт?
→ Роботу.

Что механик дал роботу?
→ Болт.

Чем робот затянул болт?
→ Ключом.
```

## 27.2. Три GAP

```text
Кто, кому и что дал?
→ Механик дал роботу болт.
```

Проверки:

```text
question_operators.length == 3
target_gaps.length == 3
selected_bindings.length == 3
binding_configuration.event_id один
resolved_node_id уникальны
all_required_gaps_bound == true
validation.valid == true
```

## 27.3. Неполный binding

При отсутствии одного участника:

```text
Кто, кому и что передал?
```

Нельзя выдавать частичный ответ как `RESOLVED`.

Ожидание:

```text
UNRESOLVED
reason: INCOMPLETE_BINDING_CONFIGURATION
```

## 27.4. Свободный порядок слов

```text
Кому дал болт механик?
Болт кому дал механик?
Механик кому дал болт?
```

Ответ должен оставаться одинаковым.

## 27.5. Пассив

```text
Роботом был поднят ключ.
Кем был поднят ключ?
→ Роботом.
```

Пассив не должен создавать отдельные фиксированные роли, но должен сохранять структуру события.

## 27.6. Ротация GAP

```text
Кто дал роботу болт?
→ Механик.

А что?
→ Болт.

А кому?
→ Роботу.

А кто?
→ Механик.
```

На каждом ходе:

```text
event_anchor_id сохраняется
ровно один participant освобождается
selected_bindings.length == 1
released node соответствует новому GAP
```

## 27.7. Неоднозначность

После добавления:

```text
Оператор дал дрону посылку.
```

Запрос:

```text
Кто дал?
```

должен вернуть неоднозначность, а не случайный первый результат.

## 27.8. Изменение состояния

```text
Ключ лежал рядом с роботом.
Позже робот положил ключ в ящик.
```

Запрос:

```text
Где ключ?
```

Должен:

* перечислить состояния;
* либо запросить временное уточнение.

## 27.9. Полисемия

```text
Лук растёт на грядке.
Охотник натянул лук.
```

Одна лемма может иметь несколько смысловых облаков.

---

# 28. Проверка измерений

## 28.1. Перенос на новые слова

Измерение, обнаруженное на Train, должно:

* проецировать новые holdout-лексемы;
* улучшать retrieval хотя бы на части holdout;
* не требовать совпадения исходных слов.

## 28.2. Устойчивость после Continual learning

После 16 новых наблюдений проверять:

```text
lineage preserved
core overlap above threshold
projection rank correlation above threshold
retrieval set overlap above threshold
utility not collapsed
```

## 28.3. Отрицательный контроль

```text
Робот поднял ключ.
Ключ поднял робот.
Роботом был поднят ключ.
```

Изменение:

```text
падежа
позиции
залога
```

не должно автоматически создавать три независимых активных семантических измерения.

## 28.4. Измерение в реальном ответе

Хотя бы один контрольный ответ должен:

```text
использовать active dimension;
иметь маршрут роя;
получить candidate event через dimensional retrieval;
пройти GraphMatcher;
дать валидный ответ;
увеличить validated_answer_contribution_count.
```

---

# 29. Критерии готовности итерации

Итерация считается успешной, если выполнены все обязательные условия.

## Множественные GAP

```text
selected_bindings является основным источником результата;
три GAP заполняются одной согласованной конфигурацией;
частичные конфигурации не выдаются как RESOLVED;
legacy selected_binding не используется внутренней логикой.
```

## Измерения

```text
служебный признак сам по себе не становится active dimension;
хотя бы одно измерение переносится на новые лексемы;
хотя бы одно измерение сохраняет lineage после дообучения;
holdout retrieval gain положительный;
stability threshold и support thresholds выполнены.
```

## Рои

```text
для каждого requested GAP создаётся GapSwarm;
JointBindingCoordinator собирает общую конфигурацию;
рои соблюдают бюджеты;
маршруты и пакеты сохраняются;
хотя бы один ответ получен через SWARM_DIMENSIONAL;
GraphMatcher может отклонить результат роя.
```

## Облака

```text
нет почти идентичных дублирующихся облаков;
overlapping relations сохраняются явно;
лимиты dimensions/clouds не превышены;
merge и prune имеют trace.
```

## Производительность

```text
25/50/100 smoke-тест завершён;
events_scanned и graph_match_attempts измерены;
нет очевидного квадратичного ухудшения;
масштабирование на тысячи событий не заявляется доказанным.
```

---

# 30. Рекомендуемый порядок реализации

## Этап 1. Канонический `selected_bindings`

* убрать внутреннюю зависимость от `selected_binding`;
* обновить backend, API, export и frontend;
* добавить audit.

## Этап 2. Multi-GAP и BindingConfiguration

* массив question operators;
* массив target gaps;
* совместный GraphMatcher;
* строгая конфигурационная validation.

## Этап 3. Ротация GAP и состояние диалога

* происхождение known nodes;
* event anchor;
* release/rebind;
* затухание контекста.

## Этап 4. Детерминированные рои

* GapSwarm;
* типы пчёл;
* бюджеты;
* маршруты;
* trace;
* индексный fallback.

## Этап 5. JointBindingCoordinator

* объединение GAP-роёв;
* пересечение event candidates;
* создание BindingConfiguration.

## Этап 6. Разделение признаков

* SemanticStructuralFeature;
* ControlFeature;
* ControlledStructuralCandidate.

## Этап 7. Discovery и shadow evaluation

* candidate/probation;
* shadow retrieval;
* holdout gain;
* support diversity;
* stability lower bound.

## Этап 8. Lineage, merge, overlap, prune

* revision;
* lineage;
* облачные отношения;
* лимиты.

## Этап 9. Подключение active dimensions к роям

* dimensional seeds;
* dimensional transitions;
* validated utility.

## Этап 10. Полный компактный эксперимент

```text
48 train
→ 16 holdout
→ 16 continual
→ blind regression
→ smoke 25/50/100
→ итоговый отчёт
```

---

# 31. Итоговый артефакт итерации

Агент должен сформировать единый отчёт:

```text
Experiment Summary

Dataset:
  train
  holdout
  continual
  blind
  smoke

Dimensions:
  discovered
  probation
  active
  merged
  pruned
  holdout gain
  post-training stability
  lineage

Swarms:
  runs
  bees
  routes
  packets
  visited universes
  fallback rate
  dimensional retrieval rate

Bindings:
  single-GAP accuracy
  multi-GAP accuracy
  unresolved correctness
  ambiguity handling
  dialogue continuation accuracy

Performance:
  25 events
  50 events
  100 events

Open limitations:
  no proof above 100 events
  limited language coverage
  limited domain diversity
  early-stage dimension discovery
```

Главный результат этапа должен звучать не как:

```text
«система выдерживает большой корпус»
```

а как:

```text
«на компактном воспроизводимом эксперименте
система обнаружила переносимое измерение,
сохранила его функцию после дообучения,
использовала его в маршруте детерминированного роя
и корректно заполнила один или несколько GAP
после строгой проверки QueryGraph».
```
