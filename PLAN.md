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



# Усиление декодера через существующую муравьиную сеть

## Summary
- Декодер остается диагностическим и не пишет в checkpoint во время `/api/decode`.
- Новую сеть не создаем: используем существующие `SemanticGraph`, `Checkpoint`, `AntColony`, феромоны концептов и феромоны ребер.
- Декодер v2 работает как ранжировщик гипотез: строит несколько грамматических вариантов из токенов, прогоняет их через граф/феромоны и выбирает лучший.
- Целевой пример: `компьютер, код, писать, программист` должен выбрать не `программист пишет компьютер и код`, а `программист пишет код на компьютере`, если сеть обучена связи `писать -> код` как объект и `писать -> компьютер` как инструмент/место.

## Key Changes
- В `decode_words(...)` добавить опциональные зависимости `checkpoint` и `graph`/`learned_edges`, но сохранить старый вызов без них.
- Backend service для `/api/decode` передает `self.engine.checkpoint`; endpoint остается read-only.
- Русский декодер разбить на 3 шага:
  - `understand/token analysis`: нормализация, морфология, concept_uri для каждого токена.
  - `candidate generation`: несколько вариантов ролей `subject`, `verb`, `object`, `instrument`, `location`, `modifier`, `complement`.
  - `ant scoring`: оценка кандидатов через существующие феромоны концептов/ребер и грамматические штрафы.
- Добавить поддерживаемые обучаемые relations в общий checkpoint:
  - `CanDo`: субъект/агент может выполнять действие.
  - `TakesObject`: глагол обычно берет объект.
  - `UsesInstrument`: действие использует инструмент.
  - `AtLocation` или `UsesLocation`: действие связано с местом/средой.
  - `HasProperty`: субъект/объект имеет признак.
- Для генерации фразы добавить русские шаблоны:
  - `subject verb object`
  - `subject verb object instrument` → `программист пишет код на компьютере`
  - `modifier subject verb complement` → `осенью лист становится жёлтым`
  - несколько объектов соединять через `и` только если оба реально имеют роль `object`.

## Ant Scoring
- Для каждого кандидата строится score:
  - базовая грамматика: есть глагол, есть субъект, роли совместимы с POS.
  - морфология: выбранные формы успешно инфлектятся.
  - concept score: `checkpoint.concept_pheromone_for(uri)` для токенов.
  - edge score: феромоны relations между выбранными ролями, например `писать TakesObject код`.
  - ant route score: короткий обход `AntColony` по графу от глагола к кандидатам ролей; короткие высокоферомонные маршруты дают бонус.
  - penalties: неодушевленный субъект у явно агентного действия, два конкурирующих объекта без доказательства, инструмент как прямой объект при наличии `UsesInstrument`.
- Если checkpoint не содержит полезных связей, декодер падает обратно на текущие морфологические эвристики.

## Training Data
- Отдельный decoder checkpoint не нужен.
- Обучение идет через существующий checkpoint/training format с `positive_edges`.
- Минимальный обучающий пример для целевого кейса:
  - `/c/ru/программист --CanDo--> /c/ru/писать`
  - `/c/ru/писать --TakesObject--> /c/ru/код`
  - `/c/ru/писать --UsesInstrument--> /c/ru/компьютер`
- Документация должна показать, что правильная фраза обучается не строкой напрямую, а связями ролей и смыслов.

## Public API / Types
- `POST /api/decode` сохраняет текущий request/response shape.
- В response можно добавить необязательные диагностические поля без ломки UI:
  - `pattern`: например `semantic_svo_instrument`.
  - `tokens[].role`: теперь может быть `instrument`, `location`, `modifier`, `complement`.
  - `tokens[].score` или `tokens[].evidence` опционально для диагностики, если удобно.
  - `summary.fallbacks` остается; добавить `summary.candidates` и `summary.ant_score` можно только если frontend готов показывать их компактно.
- UI `/decode` должен показывать новые роли и итоговую фразу; отдельный экран обучения не добавлять в этой задаче.

## Test Plan
- Unit:
  - без checkpoint: старые кейсы продолжают работать.
  - с обученными edges: `компьютер, код, писать, программист` → `программист пишет код на компьютере`.
  - случайный порядок того же набора токенов дает ту же фразу.
  - `кот, есть, рыба, мясо` остается `кот ест рыбу и мясо`.
  - `осень, лист, становиться, желтый` остается `осенью лист становится жёлтым`.
  - если edge `писать TakesObject компьютер` усилен сильнее, декодер выбирает объектом `компьютер`, то есть обучение реально влияет на ранжирование.
- API:
  - `/api/decode` использует checkpoint для чтения, но не меняет файл checkpoint, results, chat_sessions.
  - response корректно сериализует новые роли.
- Regression:
  - английский декодер не ломается.
  - пустой ввод остается `pattern="empty"`.

## Assumptions
- Выбран общий `Checkpoint`, без отдельной decoder-сети.
- `/api/decode` остается read-only; обучение делается существующим training/feedback механизмом через `positive_edges`.
- В этой задаче не трогаем пониматель, чатовый ответчик и общий analyze pipeline, кроме передачи checkpoint в декодер.
- Первый этап покрывает русские простые фразы с одним глаголом; сложные предложения, несколько глаголов и предлоги кроме базового `на/в/с` остаются следующим этапом.
