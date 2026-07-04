# semantic_ants

`semantic_ants` — исследовательский прототип языковой модели, где фраза проходит путь:

`слова -> верхние абстракции -> ConceptNet/WordNet-концепты -> муравьиные маршруты -> смысловой вектор -> короткий ответ`.

Прототип не является LLM. Он нужен как проверяемая основа для будущего смыслового слоя: памяти, роутинга, объяснимых маршрутов, обучения по уровням и интерпретации готовых смысловых векторов.

## Быстрый запуск

```powershell
python -m semantic_ants analyze "яблоко упало на голову" --trace
python -m semantic_ants analyze "яблоко упало" --strength-vector 3 --trace --json
python -m semantic_ants chat
python -m semantic_ants train data/examples.jsonl --epochs 3
python -m semantic_ants learn data/top_layer_curriculum.jsonl --epochs 5 --strength-vector 3
python -m semantic_ants download-dataset spc --split train --limit 2000 --output data/spc_dialogues.jsonl
python -m semantic_ants learn-dialogues data/spc_dialogues.jsonl --epochs 1
python -m semantic_ants interpret-vector vector.json
python -m semantic_ants feedback --last --score 5
python -m semantic_ants eval data/examples.jsonl --json
```

Веб UI:

```powershell
python -m pip install -e ".[web]"
python scripts/run_web.py --no-cache-refresh
```

Подробнее: `docs/web.md`.

## Режим чата

```powershell
python -m semantic_ants chat
```

Команды выхода: `/exit`, `/quit`, `выход`, `пока`.

Одно сообщение без интерактивного цикла:

```powershell
python -m semantic_ants chat --once "кто ты"
python -m semantic_ants chat --once "покажи русский алфавит" --no-cache-refresh
python -m semantic_ants chat --once "частые русские слова"
python -m semantic_ants chat --session-id default --mode hybrid
```

При первом запуске движок автоматически загружает встроенную базу: русский и английский алфавиты, частые слова, базовые смыслы и графовые связи. Готовые диалоговые реплики не загружаются как кодовые правила.

## Что обучается

ConceptNet и WordNet-derived данные остаются внешним read-only источником. Локально меняется только checkpoint:

- верхний слой абстракций (`/m/top/object`, `/m/top/action`, `/m/top/person` и т. д.);
- феромоны маршрутов;
- пользовательские связи;
- подавленные ложные концепты;
- память ответов;
- история последних результатов для feedback.
- история чат-сессий;
- PyTorch-словарь и метаданные диалогового генератора;
- встроенные seed-данные для алфавитов, частых слов и базовых смыслов.

Файл состояния по умолчанию: `.semantic_ants/checkpoints/model.json`.
Веса диалоговой модели по умолчанию: `.semantic_ants/models/dialogue.pt`.

## Верхний слой и сила

Новая схема добавляет слой `0`: крупные абстрактные области, похожие на материки смысловой карты. Первый элемент `--strength-vector` задает, сколько шагов сигнал может сделать по этому верхнему слою:

```powershell
python -m semantic_ants analyze "яблоко упало" --strength-vector 3 --trace --json
```

Результат содержит `semantic_vector` и `signal_trace`. `semantic_vector` можно отдельно декодировать в понятную фразу:

```powershell
python -m semantic_ants analyze "яблоко" --strength-vector 3 --json > result.json
python -c "import json; d=json.load(open('result.json')); print(json.dumps(d['semantic_vector'], ensure_ascii=False))" > vector.json
python -m semantic_ants interpret-vector vector.json
```

Подробнее: `docs/training.md`.

## Ограничения

- Токенизация и определение языка простые, без морфологии.
- Диалоговая генерация требует обученных пар реплик; без них используется смысловое резюме маршрутов.
- При недоступном ConceptNet используется слабая fallback-связь, чтобы CLI не падал полностью.
- Встроенный чат использует графовые маршруты, контекст сессии, обученную память и PyTorch-навигацию ответа.
- PyTorch нужен только для обучения генеративного декодера; без него используется детерминированная интерпретация смыслового вектора.
- Для реальной LLM этот проект стоит рассматривать как прототип semantic memory/router, а не как готовую генеративную модель.
