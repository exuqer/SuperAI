from superai.service import ServiceConfig, SuperAIService

service = SuperAIService(ServiceConfig.from_environment())
db = service.database

# Check tasks table for benchmark tasks
tasks = db.all('SELECT task_id, status, trace_id, hive_id FROM tasks WHERE task_id LIKE "task_%" ORDER BY created_at DESC LIMIT 10')
for t in tasks:
    print(f'Task: {t["task_id"]}, Status: {t["status"]}, Trace: {t["trace_id"]}, Hive: {t["hive_id"]}')

# Check benchmark runs
runs = db.all('SELECT run_id, mode, quality, latency_ms, cost, status, manifest_id FROM benchmark_runs ORDER BY created_at DESC LIMIT 10')
for r in runs:
    print(f'Run: {r["run_id"]}, Mode: {r["mode"]}, Quality: {r["quality"]}, Latency: {r["latency_ms"]}ms, Cost: {r["cost"]}, Status: {r["status"]}, Manifest: {r["manifest_id"]}')

service.close()