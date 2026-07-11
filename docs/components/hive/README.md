# HiveManager

Улей — ограниченное рабочее пространство задачи, не предметная база знаний.
`HiveManager` выбирает continue/create/restore по conversation, project и
лексическим topic terms и записывает кандидатов и score в trace.

Горячая память содержит `GoalStore`, `WorkingContextStore`, `EvidenceStore` и
`IntermediateResultStore`. Каждый entry хранит размер, relevance, protected,
source, стоимость восстановления и policy. `CapacityController` сначала
вытесняет cache/низкорелевантные неприкреплённые entries в тёплый слой с
reason-code; protected goals, constraints и evidence не пропадают молча.

Freeze создаёт согласованный `ModuleSnapshot`; restore проверяет tenant и hash
снимка и использует тёплый контекст без полного чтения архива.
