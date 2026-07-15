# SuperAI V2

Cloud / Space / Placement модель с идемпотентным обучением, локальной физикой сцен и изолированной памятью улья.

## Запуск

```powershell
python -m pip install -e ".[dev]"
python -m uvicorn server.server:app --reload
cd web
npm install
npm run dev:frontend
```

Backend: `http://127.0.0.1:8000`. Frontend: `http://127.0.0.1:5173`.

## API

- `POST /api/v2/training/learn`
- `GET /api/v2/field`
- `GET /api/v2/stats`
- `DELETE /api/v2/model`
- `GET /api/v2/spaces/{id}`
- `POST /api/v2/spaces/{id}/physics/tick`
- `GET /api/v2/placements/{id}`
- `GET /api/v2/clouds/{id}`
- `GET /api/v2/clouds/{id}/structure`
- `GET /api/v2/scenes/{id}`
- `POST /api/v2/hives`
- `POST /api/v2/hives/{id}/query`

Иерархия морфологии и генерации поверхности:

- `GET /api/v2/hives/{id}/hierarchy`
- `POST /api/v2/hives/{id}/cells/{cell_id}/expand`
- `POST /api/v2/hives/{id}/subspaces/{subspace_id}/collapse`
- `POST /api/v2/hives/{id}/generate`
- `GET /api/v2/hives/{id}/generation-candidates`
- `GET /api/v2/hives/{id}/generation-candidates/{candidate_id}`
- `POST /api/v2/hives/{id}/generation-candidates/{candidate_id}/select`
- `POST /api/v2/hives/{id}/validate-surface`

Морфология хранится нормализованно в `word_form_features`, `cloud_compositions` и
`morph_pattern_data`. Временные варианты поверхности находятся только в
`hive_generation_candidates`; ручной выбор переводит вариант в статус `SELECTED`.
Старый reasoning JSON остаётся совместимым (`schema_version: 2`), а текущий экспорт
дополнен разделами `subspaces`, `generation_candidates`, `sentence_plan`,
`selected_surface`, `reverse_validation` и `morphology_trace`.

Текущая схема хранилища — v4. Аддитивная миграция сохраняет V2-данные, добавляет
морфологические типы, типизированные семантические свидетельства и реестр туманностей.
При запуске приложения существующие сцены идемпотентно проходят семантический backfill;
глобальные координаты при этом не изменяются. Перенос данных из V1 не поддерживается.

Продолжения с `ещё`, формами `другой` и конструкцией `кроме` проходят отдельную стадию
`CONTEXT_INHERITANCE`. Для сопоставления ролей используется единая шкала: точная форма
`1.00`, лемма `0.95`, устойчивое понятие `0.85`, связанное понятие `0.65`, общая
категория `0.45`. Сцена становится кандидатом ответа только после проверки всех
обязательных опорных ролей; отклонённые результаты остаются видимыми в трассировке.
