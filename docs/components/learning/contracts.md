# Контракты обучения

`CompostRecord` ссылается на private artifact и trace, сохраняет access scope.
`SkillManifest` содержит процедуру, train/holdout refs, metrics, lifecycle и
rollback target. `GenomeManifest` содержит исполняемый компонент и hash, но не
сессионное состояние.
