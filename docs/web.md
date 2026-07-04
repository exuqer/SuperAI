# Web UI

Веб-клиент состоит из двух частей:

- Python HTTP API поверх `SemanticEngine`, `Trainer`, `ACOTrainer` и checkpoint;
- Vue 3/Vite клиент в `web/`.

## Установка

```bash
python3.11 -m pip install -e ".[web]"
cd web
npm install
```

Если локальный `pip` слишком старый для editable install, обновите pip или установите web-зависимости напрямую:

```bash
python3.11 -m pip install "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9"
```

## Запуск для разработки

Одна команда из корня проекта:

macOS/Linux:

```bash
python3 scripts/run_web.py --no-cache-refresh
```

Windows PowerShell:

```powershell
py scripts\run_web.py --no-cache-refresh
```

Скрипт запускает `semantic_ants.server.cli` и `npm run dev`, сам выбирает `npm` или `npm.cmd`, а если `web/node_modules` отсутствует, сначала выполнит `npm install`. По `Ctrl+C` он останавливает и клиент, и сервер.

Полезные параметры:

```bash
python3 scripts/run_web.py --api-port 8766 --ui-port 5174 --state-dir .semantic_ants_dev
python3 scripts/run_web.py --no-install
```

Ручной запуск в двух терминалах:

В первом терминале:

```bash
semantic-ants-web --state-dir .semantic_ants --no-cache-refresh
```

Во втором терминале:

```bash
cd web
npm run dev
```

Клиент будет доступен на `http://127.0.0.1:5173`, API проксируется на `http://127.0.0.1:8765`.

## Production build

```bash
cd web
npm run build
cd ..
semantic-ants-web --state-dir .semantic_ants --static-dir web/dist
```

Если `web/dist` существует, сервер может раздавать SPA сам.

## Основные API

- `POST /api/analyze` возвращает `result`, `graph`, `trace_interpretation`;
- `POST /api/chat/message` сохраняет контекст сессии;
- `GET /api/graph` возвращает checkpoint/seed graph с фильтрами;
- `GET /api/concepts/detail?uri=...` возвращает узел и его связи;
- `POST /api/training/train`, `/learn`, `/learn-dialogues` создают фоновые jobs;
- `GET /api/jobs/{job_id}` возвращает состояние фоновой задачи.
