# Training Telemetry

Каждый запуск получает `training_run_id`. Изменения фиксируются атомарно в `training_change_events`:

- `CLOUD_CREATED`
- `CLOUD_STRENGTHENED`
- `SPACE_CREATED`
- `PLACEMENT_CREATED`
- `PLACEMENT_MOVED`
- `STRUCTURE_CREATED`
- `SCENE_REUSED`
- `LEXEME_LINKED`
- `CANDIDATE_CREATED`
- `ACTIVATION_CHANGED`

Ответ обучения группирует те же события без повторного вычисления состояния.
