from superai.service import ServiceConfig, SuperAIService

service = SuperAIService(ServiceConfig.from_environment())
db = service.database

# Check the latest benchmark tasks
tasks = db.all('SELECT task_id, status, answer_json FROM tasks WHERE task_id LIKE "task_%" ORDER BY created_at DESC LIMIT 5')
for t in tasks:
    import json
    ans = json.loads(t['answer_json']) if t['answer_json'] else None
    print(f'Task: {t["task_id"]}, Status: {t["status"]}')
    if ans:
        print(f'  Answer: {ans.get("answer", "N/A")[:200]}')
    print()

service.close()