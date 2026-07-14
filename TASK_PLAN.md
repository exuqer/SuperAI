# Непрерывная туманность со смысловыми понятиями

## Кратко

- Итоговые уровни: `signal → character → word_form → lexeme → concept → scene → context`.
- `scene` хранит конкретное предложение, `word_form` — написанное слово, `lexeme` объединяет его формы, `concept` образует нечёткую смысловую область вроде «сосуд».
- Понятие не является жёстким контейнером: «бокал» может одновременно входить в области «сосуд», «стекло», «напиток».
- Основной зум остаётся непрерывным; понятия отображаются крупными пересекающимися полями поверх слов, а не отдельным режимом.

## Обучение и данные

- Добавить слой `lexeme`. Для русского получать нормальную форму через [`pymorphy3>=2.0.6`](https://github.com/no-plagiarism/pymorphy3), для неизвестных и латинских слов использовать `casefold`.
- Связывать `lexeme → word_form → character` структурными компонентами. Предложение продолжает содержать исходные словоформы, поэтому на экране остаётся «рыбу», а не только «рыба».
- Для каждой лексемы накапливать контекстный вектор: соседние лексемы в пределах предложения с весом `1 / (1 + расстояние)`. Частотные слова подавлять через PPMI.
- Сравнивать контекстные векторы косинусной близостью. После минимум трёх разных контекстов создавать устойчивый центр из двух и более лексем при сходстве не ниже `0.72`.
- Хранить в метаданных понятия его контекстный центроид. Нечёткий вклад лексемы вычислять как `smoothstep(0.55, 0.85, cosine)`; одна лексема может иметь ненулевой вклад в несколько понятий.
- Совпавший новый кластер присоединять к существующему понятию при сходстве центроидов от `0.85`; понятия со сходством от `0.92` объединять.
- До появления общего смыслового якоря подписывать центр тремя сильнейшими представителями: «кружка · стакан · бокал». Если наблюдаемая лексема имеет высокую центральность и контекстный охват минимум в `1.25×` выше медианы участников, использовать её как имя — например «сосуд».
- Удалить текущую логику, превращающую мешок слов одного предложения в `concept`. Сцены создавать отдельно и сразу, с сохранением порядка слов.
- Выполнить миграцию схемы: добавить `lexeme` и материализованные контекстные признаки, очистить старые несовместимые concept-кластеры, но сохранить символы и словоформы. Переходы между слоями определять через `order_index`, а не арифметикой идентификаторов.

## Пространство, API и интерфейс

- Сцены сближать по взвешенному Jaccard-сходству лексем: `(r1 + r2) × clamp(1.15 − 1.1 × similarity, 0.35, 1.15)`.
- Понятие проецировать в текущую область как взвешенный центр видимых словоформ; его радиус покрывает участников по среднеквадратичному расстоянию. Раздельные группы одного понятия остаются раздельными проекциями.
- Расширить `GET /api/field/hierarchy?max_depth=3`: вернуть сцены, структурные пространства, лексемы и `semantic_overlays` с локальными центрами, радиусами и вычисленными вкладами.
- При среднем масштабе одновременно показывать слова и более крупные полупрозрачные поля «сосуд», «стекло», «напиток». На глубоком масштабе проявлять словоформы и буквы; родительские поля сохранять с минимальной прозрачностью `0.12`.
- Дочерний уровень проявлять через `smoothstep(140px, 260px)` экранного радиуса. Камера работает в диапазоне `0.22–64×` относительно курсора.
- Одинаковые слова и буквы объединять только при физическом пересечении их проекций. Далёкие появления одной лексемы не стягивать в общий экранный центр.
- Убрать Structural/Semantic-переключатель; инспектор показывает лексему, морфологические формы и все смысловые вклады с весами.

## Тесты

- «кружка», «кружку», «кружкой» создают одну лексему и разные словоформы.
- Сходные контексты для «кружка», «стакан», «бокал» создают общее понятие; простое совместное появление слов без сходных контекстов не создаёт ложный кластер.
- После обучения слову «сосуд» составная подпись заменяется смысловым якорем.
- «Бокал» одновременно получает вклады в несколько пересекающихся понятий.
- Два предложения «Кот ест рыбу» и «Кот ест мясо» остаются пересекающимися сценами; при зуме общие слова отображаются один раз, затем раскрываются буквы.
- Проверить миграцию существующей базы, отсутствие дубликатов компонентов, иерархический API, плавный LOD, Python/Vitest/Playwright и production-сборку.

## Допущения

- Понятия возникают только из накопленного опыта; система не использует готовый словарь синонимов.
- Контекстное сходство описывает общую смысловую область, поэтому в одном понятии допустимы синонимы, виды общего класса и противоположности вроде «горячее/холодное».
- Семантические связи не отображаются рёбрами: принадлежность выражается плотностью, расстоянием и пересечением полей.

---

## Implementation Checklist

### Phase 1: Database Schema & Migration
- [x] Add `lexeme` layer to database schema (layer order_index=3)
- [x] Create `lexemes` table with morphological info (canonical_form, pos_tag, features)
- [x] Create `word_form_to_lexeme` table linking word_form clouds to lexemes
- [x] Create `context_vectors` table for PPMI-weighted context vectors
- [x] Create `concept_centroids` table for concept centroids
- [x] Create `lexeme_concept_membership` table for fuzzy membership
- [x] Create `scenes` table with ordered word_form and lexeme IDs
- [x] Create `scene_similarity` table for weighted Jaccard similarity
- [x] Create `semantic_overlays` table for concept projections
- [x] Update `layers` table with all 7 layers in correct order
- [x] Bump SCHEMA_VERSION to 5 and write migration logic
- [x] Clean old incompatible concept clusters on migration

### Phase 2: Lexeme Service (Russian Morphology + Context Vectors)
- [x] Implement `normalize_russian()` using pymorphy3
- [x] Implement `normalize_unknown()` using casefold for non-Russian
- [x] Implement `get_or_create_lexeme()` with language detection
- [x] Implement `link_word_form_to_lexeme()` 
- [x] Implement `accumulate_context()` with weighted co-occurrence (1/(1+distance))
- [x] Implement PPMI calculation for context vectors
- [x] Implement `cosine_similarity()` for context vectors
- [x] Implement `find_or_create_concept()` with thresholds (0.72, 0.85, 0.92)
- [x] Implement centroid computation and storage
- [x] Implement `_smoothstep(0.55, 0.85, x)` for fuzzy membership
- [x] Implement concept naming (top 3 representatives → semantic anchor)
- [x] Implement `merge_concepts()` for 0.92+ similarity
- [x] Add lexeme layer to layer initialization

### Phase 3: Training Pipeline Integration
- [x] Update `TrainingManager.learn()` to use new hierarchy
- [x] Add `_learn_lexemes()` method (tokenize → lexeme lookup → context accumulation)
- [x] Update `_learn_concepts()` to use lexeme context vectors
- [x] Update `_create_scene()` to preserve word order, store lexeme IDs
- [x] Remove old concept-from-bag-of-words logic
- [x] Ensure character → word_form → lexeme structural links created
- [x] Update `TrainingConfig` with new layer flags
- [x] Test with semantic space + scene similarity
- [x] Implement weighted Jaccard similarity for scenes: `(r1 + r2) * clamp(1.15 - 1.1 * similarity, 0.35, 1.15)`
- [x] Store scene similarities in `scene_similarity` table
- [x] Add API endpoint for scene similarity queries

### Phase 4: Scene Similarity & Semantic Overlays
- [x] Implement scene similarity computation and storage
- [x] Implement semantic overlay projection for concepts
- [x] Update `_ensure_semantic_space()` to create overlays
- [x] Compute concept centers and radii from member word forms

### Phase 5: API Extensions
- [x] Extend `GET /api/field/hierarchy` to return:
  - Scenes with sentence text and ordered word forms
  - Structural spaces with children
  - Lexemes with morphological forms
  - Semantic overlays with centers, radii, member weights
- [ ] Add `/api/lexemes` endpoints for lexeme inspection
- [ ] Add `/api/scenes` endpoints
- [ ] Add `/api/concepts/{id}/members` for fuzzy membership

### Phase 6: Frontend - Visualization & LOD
- [ ] Update `NebulaRenderer.vue` for continuous zoom (0.22-64x)
- [ ] Implement smoothstep(140px, 260px) for child level reveal
- [ ] Show word forms + concept fields at medium zoom
- [ ] Show characters at deep zoom, keep parents at 0.12 opacity
- [ ] Merge overlapping projections only on physical overlap
- [ ] Distant same-lexeme appearances not merged
- [ ] Remove Structural/Semantic mode toggle from UI
- [ ] Update inspector to show lexeme, morphological forms, semantic contributions

### Phase 7: Tests & Validation
- [x] Test: "кружка", "кружку", "кружкой" → one lexeme, three word forms
- [x] Test: "кружка", "стакан", "бокал" similar contexts → shared concept
- [x] Test: co-occurrence without context similarity → no false cluster
- [x] Test: "сосуд" learned → composite label replaced by semantic anchor
- [x] Test: "бокал" → memberships in multiple concepts
- [ ] Test: "Кот ест рыбу" + "Кот ест мясо" → intersecting scenes, shared words once
- [x] Test: Database migration preserves characters/word_forms, cleans concepts
- [x] Test: No duplicate structural components
- [x] Test: Hierarchical API returns all layers correctly
- [ ] Test: Smooth LOD transitions
- [x] Run Python tests (pytest)
- [ ] Run Vitest frontend tests
- [ ] Run Playwright e2e tests
- [ ] Verify production build works