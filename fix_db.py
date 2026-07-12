from superai.service import ServiceConfig, SuperAIService

service = SuperAIService(ServiceConfig.from_environment())
db = service.database

# Add missing columns
db.connection.execute('ALTER TABLE benchmark_runs ADD COLUMN mode TEXT NOT NULL DEFAULT "baseline"')
db.connection.execute('ALTER TABLE benchmark_runs ADD COLUMN concept_id TEXT')
db.connection.execute('CREATE INDEX IF NOT EXISTS idx_benchmark_runs_mode ON benchmark_runs(mode)')
db.connection.commit()

# Verify
cols = db.all('PRAGMA table_info(benchmark_runs)')
for c in cols:
    print(c)

service.close()