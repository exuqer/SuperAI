from superai.service import ServiceConfig, SuperAIService

service = SuperAIService(ServiceConfig.from_environment())
db = service.database

# Check tasks from benchmark runs
tasks = db.all('SELECT task_id, trace_id, status, created_at FROM tasks WHERE tenant_id = "local" ORDER BY created_at DESC LIMIT 10')
for t in tasks:
    print(f'Task: {t["task_id"]}, Trace: {t["trace_id"]}, Status: {t["status"]}, Created: {t["created_at"]}')

# Check benchmark runs
runs = db.all('SELECT run_id, mode, quality, latency_ms, cost, status FROM benchmark_runs ORDER BY created_at DESC LIMIT 5')
for r in runs:
    print(f'Run: {r["run_id"]}, Mode: {r["mode"]}, Quality: {r["quality"]}, Latency: {r["latency_ms"]}ms, Cost: {r["cost"]}, Status: {r["status"]}')

service.close()