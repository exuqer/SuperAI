# Model V2 Reset

Миграции V1 отсутствуют. При переходе старый `.superai/state.sqlite` удаляется.

`DELETE /api/v2/model` удаляет обучение, placements, spaces, telemetry и все ульи. Следующий запрос поля создаёт пустой `global_field`.
