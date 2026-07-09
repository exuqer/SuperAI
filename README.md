# semantic_ants

`semantic_ants` - локальный прототип dense token graph AI.

Система хранит состояние в SQLite checkpoint и работает в двух основных режимах:

- текстовое обучение и чат на токен-графе;
- иерархическое обучение на `{"hierarchy": [...], "text": "..."}` с гипернодами и вложенными subgraph'ами.

## Что хранится

- токены с 384-мерными векторами;
- направленные связи `next` между токенами;
- гиперноды для иерархий `hyper:<path>`;
- `transition_memory` для контекстных переходов;
- `sessions` и `results`;
- `training_runs` и служебные поля в `meta`.

Канонический файл состояния: `.semantic_ants/checkpoint.sqlite`.
Рядом создается `.semantic_ants/checkpoint.json` как manifest.

## Preprocess

Диалоговый JSONL:

```powershell
python -m semantic_ants.preprocess --input_path .\data\raw.jsonl --output_path .\data\train.txt
```

Иерархический источник:

```powershell
python -m semantic_ants.preprocess --input_path .\docs --output_path .\data\hierarchy.jsonl --mode hierarchy
```

Авто-режим:

```powershell
python -m semantic_ants.preprocess --input_path .\data\source --output_path .\data\out.jsonl --mode auto
```

Поведение:

- `dialogue` читает JSONL с парами `question/answer`, `prompt/response`, `input/output`, `instruction/response`, `source/target`, `query/answer` или `messages`;
- учитывает `relevance`;
- отбрасывает короткие/спамные пары;
- ограничивает однотипные стартеры вроде `привет` и `давай`;
- пишет корпус строками вида `[__user__] ... [__assistant__] ... .`
- `hierarchy` обходит директории, code/doc файлы и `.txt/.text` с заголовками, пропускает мусорные каталоги вроде `.git`, `node_modules`, `dist`, `build`, большие файлы и пишет JSONL записей `{"hierarchy":[...],"text":"..."}`.

## Training

Обычный текст:

```powershell
python -m semantic_ants train --text "Привет как дела? => Все хорошо спасибо"
```

JSONL:

```powershell
python -m semantic_ants train --dataset .\data\dataset.jsonl
```

Поведение `train`:

- `--text` разбивает вход на диалоговые пары, роли или последовательности;
- `--dataset` автоопределяет формат: raw dialogue JSONL, если записи похожи на `question/answer/relevance`; hierarchy JSONL, если у всех записей есть `hierarchy` или `text`; иначе обычный JSONL, который приводится к тексту;
- raw dialogue JSONL тренируется напрямую;
- дубликаты raw dialogue JSONL схлопываются через временное SQLite-хранилище пар;
- hierarchy JSONL создает гиперноды по всем префиксам hierarchy и тренирует текст внутри subgraph leaf-узла;
- token graph строится из токенов, пунктуации и переходов `next`;
- роли `[__user__]` и `[__assistant__]` нормализуются во внутренние `__user__` и `__assistant__`;
- checkpoint сохраняется в SQLite.

## Chat

```powershell
python -m semantic_ants chat --text "привет"
```

Ответ строится из:

- embedding текущего запроса;
- последних 3 сообщений сессии с экспоненциальным затуханием;
- текущего focus stack, если был `drill-down` в гиперноду;
- dense backpack-графа для скоринга.

Генерация идет пошаговым обходом графа от наиболее релевантных токенов. При активной гиперноде ответ может строиться из ее локального subgraph, а не из общего token graph.

Полезные команды:

```powershell
python -m semantic_ants graph --query "привет"
python -m semantic_ants serve --host 127.0.0.1 --port 8000
```

## API

- `GET /api/health`
- `GET /api/config`
- `POST /api/chat/message`
- `POST /api/chat/drill-down`
- `POST /api/chat/drill-up`
- `GET /api/chat/sessions`
- `DELETE /api/chat/sessions/{session_id}`
- `POST /api/train`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/graph`
- `GET /api/node/{node_id}`
- `POST /api/feedback`

Основные payload'ы:

- `POST /api/chat/message`: `text`, `session_id`, `backpack_limit`
- `POST /api/chat/drill-down`: `node_id`, `session_id`, `limit`
- `POST /api/chat/drill-up`: `session_id`, `limit`
- `POST /api/train`: `text` или `dataset_path`, `session_id`, `epochs`, `max_pairs`

## Checkpoint

`Checkpoint` сериализуется в SQLite и содержит:

- `version`
- `vector_dim`
- `tokens`
- `sessions`
- `results`
- `meta`

`meta` включает:

- `hypernodes`
- `backpack_stack`
- `transition_memory`
- `training_runs`

Старые форматы checkpoint читаются мягко: токены нормализуются, `transition_edge` приводится к `next`, subgraph payload'ы пересобираются в актуальный вид.
