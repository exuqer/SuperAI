# Примеры

## Анализ

```powershell
python -m semantic_ants analyze "яблоко упало на голову" --trace
python -m semantic_ants analyze "яблоко упало" --strength-vector 3 --trace --json
```

Выводит короткий ответ, смысловое резюме, `result_id`, `semantic_vector`, `signal_trace` и несколько маршрутов муравьев.

## Чат

```powershell
python -m semantic_ants chat
```

Примеры сообщений:

```text
привет
кто ты
что ты умеешь
покажи русский алфавит
частые русские слова
что такое яблоко
```

Для автоматической проверки без интерактивного режима:

```powershell
python -m semantic_ants chat --once "кто ты" --no-cache-refresh
```

## Обучение

```powershell
python -m semantic_ants train data/examples.jsonl --epochs 5
python -m semantic_ants learn data/top_layer_curriculum.jsonl --epochs 5 --strength-vector 3
```

После обучения checkpoint усилит связи между яблоком, головой и Ньютоном.
Второй пример обучает верхний слой абстрактных доменов.

## Интерпретация вектора

```powershell
python -m semantic_ants interpret-vector vector.json
```

Команда читает JSON `semantic_vector` и строит понятное предложение. Без обученной Torch-модели используется fallback по главному домену и сильнейшим понятиям.

## Feedback

```powershell
python -m semantic_ants feedback --last --score 5
python -m semantic_ants feedback --last --score 1 --corrected-concepts "/c/en/newton" --corrected-response "Это ассоциация с Ньютоном."
```

Feedback применяет подкрепление к последнему сохраненному результату.
