# semantic_ants

`semantic_ants` — исследовательский прототип языковой модели, где фраза проходит путь:

`слова -> ConceptNet/WordNet-концепты -> муравьиные маршруты -> смысловое резюме -> короткий ответ`.

Прототип не является LLM. Он нужен как проверяемая основа для будущего смыслового слоя: памяти, роутинга, объяснимых маршрутов и базового обучения.

## Быстрый запуск

```powershell
python -m semantic_ants analyze "яблоко упало на голову" --trace
python -m semantic_ants chat
python -m semantic_ants train data/examples.jsonl --epochs 3
python -m semantic_ants download-dataset spc --split train --limit 2000 --output data/spc_dialogues.jsonl
python -m semantic_ants learn-dialogues data/spc_dialogues.jsonl --epochs 1
python -m semantic_ants feedback --last --score 5
python -m semantic_ants eval data/examples.jsonl --json
```

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

## Ограничения

- Токенизация и определение языка простые, без морфологии.
- Диалоговая генерация требует обученных пар реплик; без них используется смысловое резюме маршрутов.
- При недоступном ConceptNet используется слабая fallback-связь, чтобы CLI не падал полностью.
- Встроенный чат использует графовые маршруты, контекст сессии, обученную память и PyTorch-навигацию ответа.
- Для реальной LLM этот проект стоит рассматривать как прототип semantic memory/router, а не как готовую генеративную модель.
