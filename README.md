# SuperAI

Минимальное приложение для 2D-гравитационного обучения слов. Реализует семантическое пространство с физической симуляцией гравитации.

## Быстрый запуск

```bash
python3 -m pip install -e '.[dev]'
./dev.sh
```

API доступен по `http://127.0.0.1:8000/docs`, клиент — по `http://localhost:3000`.
Скрипт запускает backend и Vite одновременно; backend можно запустить отдельно
через `python3 -m uvicorn server.server:app`.

По умолчанию состояние сохраняется в `.superai/`. Его можно переназначить
переменной `SUPERAI_DATA_DIR`.

## API

- `POST /api/v1/training/learn` — обучить на тексте `{ "text": "..." }`
- `GET /api/v1/training/space` — получить текущее пространство слов
- `DELETE /api/v1/training/space` — сбросить пространство слов
- `GET /api/health` — проверка здоровья

## Проверка

```bash
python3 -m pytest
cd web && npm install && npm run build
```

## Структура проекта

- `server/` — Python пакет
  - `server.py` — FastAPI сервер
  - `training.py` — логика обучения и физики
  - `tokenizer.py` — токенизация русского/латинского текста
  - `database.py` — SQLite хранение
  - `physics.py` — гравитационная физика
- `web/` — Vue/Vite фронтенд
- `tests/` — pytest тесты
