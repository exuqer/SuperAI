# Операции обучения

Цикл компоста: task → `/compost` → `/validate` → явный `/integrate`. Цикл
навыка: `/skills` → validate с benchmark metrics → shadow → activate. Любой
шаг, нарушающий disjoint train/holdout или baseline, отвергается. В production
не включайте автопромоут без отдельной политики и benchmark manifest.
