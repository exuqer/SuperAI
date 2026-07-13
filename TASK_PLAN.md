# Task Plan: Рекурсивная многоуровневая система «туманностей»

## Этап 1: Аудит ✅
- [x] Изучить database.py, physics.py, tokenizer.py, training.py, server.py
- [x] Написать AUDIT_REPORT.md
- [x] Создать структуру новых папок (models, services, repositories)

## Этап 2: Базовая модель (Database Schema & Repositories)
- [ ] database.py - полная переработка схемы БД
  - [ ] Таблица `layers` (слои: signal, character, word_form, concept, scene, context)
  - [ ] Таблица `clouds` (туманности вместо concepts)
  - [ ] Таблица `spaces` (локальные пространства туманностей)
  - [ ] Таблица `cloud_placements` (локальные появления туманностей)
  - [ ] Таблица `structural_components` (внутренний состав)
  - [ ] Таблица `activation_events` (события активации)
  - [ ] Таблица `coactivation_stats` (статистика совместной активации)
  - [ ] Таблица `condensation_candidates` (кандидаты конденсации)
  - [ ] Индексы для производительности
- [ ] repositories/cloud_repository.py - методы работы с туманностями
- [ ] repositories/space_repository.py - методы работы с пространствами
- [ ] Миграция существующих данных (concepts → clouds)

## Этап 3: Структурный зум (Character → Word Form)
- [ ] models/cloud.py - модель Cloud с полями по спецификации
- [ ] models/space.py - модель Space
- [ ] tokenizer.py - иерархическая токенизация (текст → предложения → слова → символы)
- [ ] training.py - character training, word-form condensation
- [ ] Реализовать создание «мяч» из «м», «я», «ч»
- [ ] API: /api/layers, /api/spaces/{id}, /api/clouds/{id}/children
- [ ] Зум: слово → буквы

## Этап 4: Физика туманностей
- [ ] services/spatial_index.py - spatial hash grid / quadtree
- [ ] services/activation.py - управление активацией
- [ ] physics.py - локальная физика в пространстве
  - [ ] Притяжение от совместной активации
  - [ ] Отталкивание
  - [ ] Stability, damping
  - [ ] Пересечение облаков (overlap)
  - [ ] Симуляция только активного улья
- [ ] Детерминированность по seed

## Этап 5: Обучение
- [ ] services/condensation.py - логика конденсации
- [ ] training.py - многоуровневое обучение
  - [ ] Character training
  - [ ] Word-form condensation (candidates → clouds)
  - [ ] Coactivation stats обновление
  - [ ] Concept candidates
  - [ ] Stability thresholds
  - [ ] Decay активации
- [ ] Исключить создание семантических ребер

## Этап 6: Семантический зум
- [ ] Semantic spaces (mode=semantic)
- [ ] Projections (cloud_placements для semantic mode)
- [ ] Region selection API (/api/select-region)
- [ ] Отображение пересечений без линий

## Этап 7: Сцены
- [ ] Scene layer
- [ ] Sequence condensation

## Этап 8: Оптимизация
- [ ] Профилирование
- [ ] Lazy loading
- [ ] WebSocket delta updates
- [ ] Тест производительности (10k+ clouds)

## Тесты (по спецификации)
- [ ] Тест 1: Буквы
- [ ] Тест 2: Повторное слово
- [ ] Тест 3: Порядок
- [ ] Тест 4: Семантическое сближение
- [ ] Тест 5: Омонимия
- [ ] Тест 6: Зум
- [ ] Тест 7: Пересечение
- [ ] Тест 8: Производительность

## Финальные entregables
- [ ] Список измененных файлов
- [ ] Описание схемы БД
- [ ] Описание API
- [ ] Инструкции запуска
- [ ] Инструкции миграции
- [ ] Результаты тестов
- [ ] Известные ограничения