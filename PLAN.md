# Декодер Search Tokens → Предложение

## Summary
- Добавить read-only диагностический декодер, обратный к `/api/understand`: `tokens/text -> роли SVO -> формы слов -> нормальная фраза`.
- Первый шаг покрывает русский и английский базовый SVO-паттерн: первый токен = субъект, второй = глагол, остальные = объекты.
- Целевой пример:
  - RU: `["кот", "есть", "рыба", "мясо"] -> "кот ест рыбу и мясо"`
  - EN: `["cat", "eat", "fish", "meat"] -> "cat eats fish and meat"`
- Декодер не пишет в checkpoint, не подключается к чату и не меняет `/api/analyze`.

## Public API / Types
- Добавить `POST /api/decode`.
- Request:
  ```json
  {
    "text": "кот есть рыба мясо",
    "tokens": ["кот", "есть", "рыба", "мясо"],
    "lang": "auto",
    "session_id": "optional",
    "turn_id": "optional"
  }
  ```
- `tokens` имеет приоритет над `text`; если `tokens` пустой, токены берутся из `text`.
- Response:
  ```json
  {
    "input_text": "...",
    "input_tokens": ["..."],
    "lang": "ru",
    "sentence": "кот ест рыбу и мясо",
    "pattern": "svo",
    "session_id": "...",
    "turn_id": "...",
    "tokens": [
      {
        "input_token": "кот",
        "normalized_token": "кот",
        "role": "subject",
        "surface": "кот",
        "concept_uri": "/c/ru/кот",
        "transform_status": "inflected",
        "morphology": {}
      }
    ],
    "summary": {
      "total_tokens": 4,
      "used_tokens": 4,
      "objects": 2,
      "fallbacks": 0
    }
  }
  ```

## Implementation Changes
- Создать backend-модуль `semantic_ants/decoding` с функцией `decode_words(...)`.
- RU-логика:
  - использовать уже обязательный `pymorphy3`;
  - субъект инфлектить в `nomn sing`;
  - глагол инфлектить в `3per sing pres`;
  - объекты инфлектить в `accs sing`;
  - список объектов соединять через `и`.
- EN-логика:
  - субъект оставить как surface;
  - глагол привести к 3rd-person singular для одиночного субъекта: `eat -> eats`, `go -> goes`, `have -> has`, `be -> is`, `do -> does`;
  - объекты соединять через `and`.
- Добавить `DecodeRequest` в backend schemas, `EngineService.decode`, endpoint `/api/decode`.
- Добавить frontend-страницу `/decode` “Декодер”:
  - textarea для `text`;
  - optional поле `tokens`, разделение по пробелам/запятым;
  - `lang`, `session_id`, `turn_id`;
  - результат sentence и таблица ролей/форм.
- Добавить клиентский метод `api.decode(...)`, TS-типы `DecodeResponse`, пункт меню и route.
- Добавить `docs/decoding.md` и ссылку из README рядом с `docs/understanding.md`.

## Test Plan
- Unit:
  - RU `["кот", "есть", "рыба", "мясо"]` возвращает `кот ест рыбу и мясо`;
  - EN `["cat", "eat", "fish", "meat"]` возвращает `cat eats fish and meat`;
  - `text`-вход работает без `tokens`;
  - при наличии `tokens` они переопределяют `text`;
  - пустой вход возвращает пустой `sentence`, `pattern="empty"`, без ошибки.
- API:
  - `/api/decode` возвращает sentence, роли и summary;
  - endpoint не пишет checkpoint, results и chat_sessions;
  - `session_id`/`turn_id` проходят через request-response.
- Frontend:
  - `/decode` доступен из меню;
  - отправка вызывает `api.decode`;
  - страница показывает sentence и таблицу role → surface.

## Assumptions
- Первый декодер детерминированный, без LLM и без обучения.
- В v1 порядок слов обязателен: `subject verb object...`.
- Выходная `sentence` возвращается без точки, чтобы совпадать с примером.
- Свободный порядок слов, несколько субъектов и сложное согласование остаются следующим этапом.
