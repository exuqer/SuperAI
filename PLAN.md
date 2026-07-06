# План Нормальных Ответов Без Дублей Понятий

## Summary

Сделать единый слой идентичности понятий: слова на разных языках остаются surface-формами, а смысл хранится в одном canonical concept. Tatoeba/translation не исключать, а использовать как межъязыковое evidence. Ответ выбирается гибридно: память, контекст, смысловое объяснение, переводческие связи, локальный генератор, fallback.

## Key Changes

- Добавить в `Checkpoint` v5:
  - `canonical_concepts`: canonical URI -> labels, aliases, langs, source URIs, quality.
  - `concept_redirects`: legacy/surface URI -> canonical URI.
  - `surface_forms`: canonical URI -> language-specific forms.
- Ввести `CanonicalResolver` и использовать его во всех write-path:
  - `add_custom_edge`
  - `reinforce_edge`
  - `reinforce_concept`
  - `remember_response`
  - `remember_accepted_answer`
  - trainers, dataset converters, feedback.
- Для новых пользовательских понятий использовать `/m/concept/<slug>`.
- `/m/top/...` оставить только для настоящих верхних доменов: `object`, `action`, `dialogue`, `mind` и т. д.
- Текущие learned topics `superai`, `graph`, `memory`, `checkpoint`, `semantic_vector`, `feedback`, `learning` мигрировать в `/m/concept/...`.
- Legacy URI вроде `/c/ru/superai`, `/c/en/superai`, `/m/top/superai` не удалять физически, а редиректить в `/m/concept/superai`.
- При обучении, если canonical concept уже есть, не создавать новый узел, а усиливать существующие связи и добавлять surface-form/alias.
- Tatoeba оставить:
  - русские и английские токены связывать через canonical concepts.
  - `TranslationEquivalent` использовать как evidence для surface/language mapping.
  - обычный чат может использовать переводческие связи, но не должен выбирать случайную Tatoeba-фразу как ответ без высокого intent-score.
- Сделать `ResponseSelector`:
  - exact QA memory
  - contextual follow-up
  - concept definition/explanation
  - bilingual/translation evidence
  - local tiny Torch candidate
  - semantic fallback.
- Язык ответа выбирать автоматически по запросу, контексту и quality-score; разрешить mixed ответ, если английский термин сильнее русского.
- Исправить subject extraction: `что такое граф` должно фокусироваться на `граф`, а не на `что/такое`.
- Добавить `context_focus` с decay по последним user-turn concepts.
- Починить `mode="hybrid"` и ошибку `ACOTrainer._evaporate` с несуществующим `response_lang`.

## Migration

- Добавить команду:
  - `python -m semantic_ants migrate-memory --dry-run`
  - `python -m semantic_ants migrate-memory --apply --backup`
- Dry-run должен показать группы дублей: canonical target, merged URIs, affected edges/responses.
- Apply:
  - создает backup checkpoint.
  - строит canonical groups по aliases, labels, normalized URI token, translation edges.
  - переписывает concepts в accepted/response/negative memory.
  - переписывает custom_edges/learned_bridges через canonical URI.
  - суммирует pheromones/concept_pheromones на canonical URI.
  - сохраняет redirects для старых URI.
- Для `SuperAI` итог должен быть один canonical concept: `/m/concept/superai`.

## Public Interfaces

- В result добавить:
  - `response_source`
  - `response_lang`
  - `response_candidates`
  - `semantic_vector.context_focus`
  - `canonical_concepts`
- В graph API показывать canonical node и surface/legacy aliases отдельно.
- В UI диагностику добавить: выбранный canonical concept, redirects, response candidate scores.

## Test Plan

- `SuperAI`, `superai`, `СуперAI` резолвятся в один canonical concept.
- Обучение повторного `SuperAI` усиливает связи, а не добавляет новый concept.
- `что такое граф` отвечает про граф.
- `как дела?` возвращает диалоговый ответ.
- `SuperAI` -> `что это?` использует контекст SuperAI.
- `переведи graph` использует Tatoeba/translation evidence.
- Обычный чат не выбирает случайную Tatoeba-фразу без переводческого intent.
- Миграция сохраняет валидный checkpoint и не теряет accepted answers.
- Запустить тесты: understanding, chat, engine training, simple training, dialogue dataset, server API, graph API.
