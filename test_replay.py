from superai.service import ServiceConfig, SuperAIService

service = SuperAIService(ServiceConfig.from_environment())

# Test replay
task_id = 'task_a085d0fa601944eb8c7f30be47a7b554'
task = service.task(task_id, 'local')
print(f'Task: {task.task_id}')
print(f'Trace ID: {task.trace_id}')
print(f'Status: {task.status}')
if task.answer:
    print(f'Answer: {task.answer.answer[:200]}')
else:
    print('Answer: None')

trace = service.trace(task.trace_id, 'local')
print(f'Trace has {len(trace["spans"])} spans')
print(f'Trace has {len(trace["events"])} events')

service.close()