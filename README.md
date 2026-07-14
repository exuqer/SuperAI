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

Схема создаётся с нуля. V1 и перенос старых данных не поддерживаются.
