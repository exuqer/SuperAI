# Normalized API V2

`GET /api/v2/field` и `GET /api/v2/spaces/{id}` возвращают:

```json
{"space": {}, "clouds": {}, "placements": [], "stats": {}}
```

`clouds` индексирован по `cloud_id`. Каждый элемент `placements` принадлежит одному `space_id`.

`GET /api/v2/scenes/{id}` возвращает `scene.components`; поля `words` и `word_forms` отсутствуют.

`GET /api/v2/clouds/{id}/structure` отдельно возвращает `structure_space`, ordered `components` и словарь дочерних clouds.

`POST /api/v2/training/learn` возвращает `training_run_id`, созданные и усиленные clouds, spaces, placements, structures, activations и reused scenes.
