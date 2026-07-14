# Current Model Audit

| Подсистема | Обнаружено | Исправление |
|---|---|---|
| Legacy database | V1 и `v2_*` существовали одновременно | Одна каноническая схема без префикса |
| Training | Структура создавалась только при `word_created` | `WordFormStructureService.ensure_structure` вызывается при каждом наблюдении |
| Scene training | Сцены добавлялись в `structural_components` | Сцена использует только `scene_components` |
| Telemetry | Возвращались только ID сцен | Все изменения пишутся в `training_change_events` |
| Hive | Ячейки хранили собственные `x/y`, но не hive placement | Координаты принадлежат `cloud_placements` внутри `hive_space` |
| Chat | Выполнялась legacy-синхронизация | Чат читает только каноническую модель |
| Frontend | Экран обучения загружал V1 concepts | UI использует normalized V2 space DTO |
