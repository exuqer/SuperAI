# Первый Модуль «Пониматель»: Лемматизирующий Токенизатор

## Summary
- Сделать отдельный backend API и вкладку `/understand` для первого шага понимания: `текст -> raw tokens -> леммы -> search tokens -> совпадения в памяти`.
- Использовать `pymorphy3` как обязательную зависимость для русской лемматизации и сохранения грамматических тегов. Источник пакета: https://pypi.org/project/pymorphy3/
- Не подключать модуль к текущему чату и `/api/analyze` на этом шаге. Он только диагностирует вход и не пишет в checkpoint.

## Key Changes
- Добавить `pymorphy3>=2,<3` в зависимости проекта.
- Создать backend-модуль `semantic_ants/understanding`, который возвращает для каждого токена:
  - `raw_token`, `lemma`, `search_token`, `concept_uri`;
  - `match_status`: `found_as_alias`, `found_as_lemma`, `found_as_raw`, `candidate`, `partial_root_match`, `edit_distance_match`, `ignored_stop_word`;
  - `is_stop_word`;
  - `morphology`: `POS`, `case`, `number`, `gender`, `tense`, `person`, если теги доступны.
- Добавить `POST /api/understand`:
  - request: `{ "text": string, "lang": "auto" | "ru" | "en", "session_id"?: string, "turn_id"?: string }`
  - response: `{ "input_text", "lang", "session_id", "turn_id", "tokens", "summary" }`
- Добавить фронт-вкладку «Пониматель»:
  - textarea для текста;
  - поля `session_id` и `turn_id` как необязательные диагностические метки;
  - таблица `raw -> lemma -> search token -> concept uri -> status -> morphology`;
  - отдельное выделение стоп-слов и candidate/partial matches.

## Behavior
- Для `котики едят` модуль должен показать рабочие токены `кот` и `есть`, например:
  - `котики -> кот -> /c/ru/кот`;
  - `едят -> есть -> /c/ru/есть`.
- Стоп-слова, частицы, союзы, предлоги и междометия получают `ignored_stop_word`, не получают `concept_uri` и не идут в будущий муравьиный старт.
- Грамматические теги сохраняются, но не участвуют в поисковом ключе первого шага. Они нужны как задел для будущей Марковской генерации.
- Если точного совпадения нет, модуль возвращает `candidate`. Затем пробует легкие локальные эвристики без эмбеддингов:
  - edit distance для опечаток;
  - общий корень/подстрока по известным alias/search tokens.
- `session_id` и `turn_id` только возвращаются в ответе и подготавливают контракт для будущей памяти диалога. В этом шаге они не создают историю.

## Test Plan
- Unit tests:
  - `котики едят` дает search tokens `кот`, `есть`;
  - `котиками` сохраняет lemma `кот` и morphology с множественным числом/падежом, если `pymorphy3` отдает эти теги;
  - `эй, ну и как там мой кот?` помечает шумовые слова как `ignored_stop_word`, а `кот` остается рабочим токеном;
  - неизвестное слово получает `candidate`, а близкая опечатка может получить `edit_distance_match`.
- API tests:
  - `/api/understand` возвращает tokens с raw/lemma/search/status/morphology;
  - endpoint не пишет в checkpoint, results и chat_sessions;
  - `session_id`/`turn_id` проходят через request-response без побочных эффектов.
- Frontend tests:
  - `/understand` доступна из меню;
  - отправка текста вызывает новый API;
  - таблица показывает леммы, статусы, стоп-слова и grammar tags.

## Assumptions
- Первый модуль готовит слова к поиску в памяти, но не генерирует ответ.
- Эмбеддинги, FastText, векторный bridge и Марковская цепь остаются следующим этапом.
- Существующий чат не меняется до отдельного шага подключения search tokens к общему муравьиному анализу.
