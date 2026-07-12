from superai.service import ServiceConfig, SuperAIService
from fastapi.testclient import TestClient
from superai.api import create_app

service = SuperAIService(ServiceConfig.from_environment())

# Test the benchmark API endpoints
app = create_app(ServiceConfig.from_environment())
client = TestClient(app)

# List benchmark runs
response = client.get('/api/v1/benchmarks', headers={'X-Tenant-Id': 'local'})
print('List benchmark runs:', response.status_code)
if response.status_code == 200:
    for run in response.json():
        print(f'  Run: {run["run_id"]}, Mode: {run["mode"]}, Quality: {run["quality"]}')

# Get a specific benchmark run
if response.status_code == 200 and response.json():
    run_id = response.json()[0]['run_id']
    response2 = client.get(f'/api/v1/benchmarks/{run_id}', headers={'X-Tenant-Id': 'local'})
    print(f'Get benchmark run {run_id}:', response2.status_code)
    if response2.status_code == 200:
        print(f'  Run: {response2.json()}')

service.close()