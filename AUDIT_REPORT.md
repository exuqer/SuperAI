# Технический аудит текущей реализации

## Текущая архитектура (до изменений)

### database.py
- **Таблица:** `concepts` (id, token, position, mass)
- **Схема:** SCHEMA_VERSION = 2
- **Индексы:** только `idx_concepts_mass`
- **Проблемы:** 
  - Одноуровневая структура (нет слоев)
  - Нет разделения формы слова и понятия
  - Нет пространств, размещений (placements), структурных компонентов
  - Нет статистики коактивации, кандидатов конденсации
  - Нет событий активации

### physics.py
- **Модель:** Глобальная 2D симуляция всех концептов сразу
- **Сила:** Гравитация (притяжение массы к массе) + отталкивание
- **Импульсы:** Позиционные импульсы от предложений
- **Порядок:** `apply_ordered_trajectory` для обучения порядку
- **Проблемы:**
  - O(N²) перебор всех пар
  - Нет локальных пространств (spaces)
  - Нет стабильности (stability)
  - Нет плотностной модели (только точка + радиус)
  - Нет пересечения облаков (overlap)
  - Нет детерминированности по seed
  - Симулируется вся база, а не только активный улей

### tokenizer.py
- **Функции:** `tokenize`, `tokenize_with_surfaces`, `split_sentences`, `normalize_text`, `canonical_token`
- **Возвращает:** Список токенов (слов)
- **Проблемы:**
  - Не возвращает символы и их порядок
  - Не сохраняет позиции в предложении
  - Не возвращает иерархию (предложения → слова → символы)
  - Не подготавливает данные для последовательной конденсации

### training.py
- **TrainingManager:** координирует обучение
- **Процесс:** токенизация → ensure_concepts → физика → обновление БД
- **Проблемы:**
  - Одноуровневое обучение (нет разделения по слоям)
  - Нет character training
  - Нет word-form condensation
  - Нет concept candidates
  - Нет порогов устойчивости
  - Создаются семантические связи через физику (импульсы предложений)
  - Нет decay активации

### server.py
- **Эндпоинты:** `/api/train`, `/api/space`, `/api/reset`, `/api/health`
- **Проблемы:**
  - Нет API для слоев, пространств, зума
  - Нет выборки по области
  - Нет WebSocket для симуляции
  - Возвращает всю базу при запросе пространства

## Что можно переиспользовать

1. **Базовая инфраструктура SQLite** - расширить схему
2. **Токенизатор** - расширить возвращаемую структуру
3. **PhysicsConfig** - адаптировать под локальные пространства
4. **FastAPI сервер** - добавить новые эндпоинты
4. **ConceptState** - расширить поля (layer_id, stability, activation history, etc.)

## Необходимые миграции (по этапам)

### Этап 2: Базовая модель
- `layers` таблица
- `clouds` таблица (замена concepts)
- `spaces` таблица
- `cloud_placements` таблица
- `structural_components` таблица
- Миграция существующих данных

### Этап 3: Структурный зум
- Добавить character и word_form слои
- Реализовать structural_components для порядка букв

### Этап 4: Физика туманностей
- Spatial index (grid/quadtree)
- Локальная симуляция в space
- Притяжение от коактивации
- Stability, overlap вычисления

### Этап 5: Обучение
- Character training
- Word-form condensation с condensation_candidates
- Coactivation_stats
- Concept candidates

### Этап 6: Семантический зум
- Semantic spaces
- Projections (cloud_placements для semantic mode)
- Region selection API

### Этап 7: Сцены
- Scene layer
- Sequence condensation

## Файлы для создания/изменения

### Новые модули:
- `server/models/cloud.py` - модель Cloud
- `server/models/space.py` - модель Space
- `server/services/condensation.py` - логика конденсации
- `server/services/activation.py` - управление активацией
- `server/services/spatial_index.py` - пространственный индекс
- `server/services/zoom.py` - навигация зум
- `server/repositories/cloud_repository.py`
- `server/repositories/space_repository.py`

### Изменяемые файлы:
- `server/database.py` - полная переработка схемы
- `server/physics.py` - локальная физика с spatial index
- `server/tokenizer.py` - иерархическая токенизация
- `server/training.py` - многоуровневое обучение
- `server/server.py` - новые API эндпоинты

## План реализации по этапам (соответствует задаче)

1. **Аудит** ✓ (текущий отчет)
2. **Базовая модель** - схемы БД, репозитории
3. **Структурный зум** - character → word_form
4. **Физика туманностей** - локальная симуляция
5. **Обучение** - многослойное с конденсацией
6. **Семантический зум** - projections, region selection
7. **Сцены** - scene layer
8. **Оптимизация** - профилирование, lazy loading