# Cloud / Space / Placement V2

- `Cloud` хранит идентичность и накопленные свойства без координат.
- `Space` задаёт изолированную систему координат.
- `Placement` хранит положение одного cloud внутри одного space.
- Один cloud может иметь global, scene и hive placements.
- Physics tick выбирает placements только одного `space_id`.
