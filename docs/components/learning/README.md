# ExperienceCompiler и GenomeRegistry

Переработка начинается только для успешной trace-backed задачи. Компост —
приватный производный artifact с source trace/artifact refs, access scope,
verification status и reconstruction pointer. Его нужно отдельно validate и
явно integrate; гипотеза остаётся в карантине и не становится verified claim.

Компилятор навыков требует нескольких успешных разнообразных training traces,
одинакового saved procedure graph и непересекающегося holdout. Переходы навыка:
`candidate → validated → shadow → active`; promotion требует положительной
quality/utility оценки, а rollback target сохраняется.

Геном описывает только versioned компонент и его параметры. Состояние Улья и
горячая память в него не попадают. EvolutionEngine лишь строит Pareto frontier
по уже измеренным candidate metrics и не исполняет произвольный код.
