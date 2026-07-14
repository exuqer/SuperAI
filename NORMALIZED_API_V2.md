# Normalized API V2

Field и space endpoints возвращают clouds и placements раздельно. Scene endpoint возвращает только scene metadata и component references. Структура слова загружается отдельным запросом при выборе или зуме.

Публичная идентичность визуального объекта: `(cloud_id, placement_id, space_id)`.
