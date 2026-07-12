from superai.service import ServiceConfig, SuperAIService

service = SuperAIService(ServiceConfig.from_environment())
db = service.database
runs = db.all('SELECT run_id, quality, latency_ms, cost, status FROM benchmark_runs ORDER BY created_at DESC LIMIT 5')
for r in runs:
    print(f'Run: {r["run_id"]}, Quality: {r["quality"]}, Latency: {r["latency_ms"]}ms, Cost: {r["cost"]}, Status: {r["status"]}')
service.close()