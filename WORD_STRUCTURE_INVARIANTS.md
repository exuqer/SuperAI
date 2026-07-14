# Word Structure Invariants

- На word-form существует один `word_structure_space`.
- На каждый индекс символа существует один `structural_component`.
- Повторяющийся символ использует тот же character cloud в разных индексах.
- Количество компонентов равно длине normalized word-form.
- Повторное обучение не создаёт structure space или components.
- Character clouds не имеют global placements.
