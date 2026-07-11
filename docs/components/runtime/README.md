# CommandRuntime

`superai.runtime.CommandRuntime` хранит `WorkItem` в SQLite, выбирает
запланированную работу по priority/scheduled_at и исполняет зарегистрированный
handler в том же процессе. Состояния: `queued → running → succeeded | failed |
cancelled | dead_letter`.

Вход — versioned `WorkItem`; выход — его новое сохранённое состояние, spans и
domain events. Повторная команда с тем же `(tenant_id, idempotency_key)`
возвращает исходный item. После рестарта `running` возвращается в `queued`, а
успешные items не проигрываются.

`RuntimeContext.checkpoint()` проверяет step/event/deadline budget и отмену.
Непредвиденные или исчерпавшие retry ошибки становятся dead letters; budget
ошибка — terminal failed task. Полные payload и секреты не пишутся в trace.
