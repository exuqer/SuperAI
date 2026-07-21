import sys, os, tempfile, atexit
from pathlib import Path
sys.path.insert(0, '.')

# Isolate database like conftest does
tmp_path = Path(tempfile.mkdtemp())
db_path = tmp_path / "state.sqlite"
import server.database as database
database.DB_PATH = db_path
database.init_db()

def cleanup():
    import shutil
    shutil.rmtree(tmp_path, ignore_errors=True)
atexit.register(cleanup)

from server.v2.graph_repository import GraphRepository
from server.v2.graph_service import GraphDialogueService, GraphTrainingService

repository = GraphRepository()
training = GraphTrainingService(repository)
dialogue = GraphDialogueService(repository)

training.train('Механик дал роботу болт.', independent_key='debug-what3')
hive_id = dialogue.create()['hive']['id']

r1 = dialogue.query(hive_id, 'Кому и кто дал болт?')
print('R1:', r1['answer']['surface'])

r2 = dialogue.query(hive_id, 'Кому?')
print('R2:', r2['answer']['surface'])

r3 = dialogue.query(hive_id, 'Кто?')
print('R3:', r3['answer']['surface'])

r4 = dialogue.query(hive_id, 'Что?')
print('R4:', r4['answer']['surface'])

print()
print('=== RELEASE DIAGNOSTICS ===')
for d in r4.get('release_diagnostics', []):
    for c in d.get('candidates', []):
        print(f"  candidate: lemma={c.get('resolved_surface')}, final_score={c.get('final_score'):.3f}, "
              f"animacy_score={c.get('animacy_score'):.3f}, animacy_conflict={c.get('animacy_conflict'):.3f}, "
              f"question_family_match={c.get('question_family_match'):.3f}")
    print(f"  decision: {d.get('decision')}, reason: {d.get('decision_reason')}")
    print(f"  selected_score: {d.get('selected_score')}, second_score: {d.get('second_score')}")
    print(f"  release_margin: {d.get('release_margin')}")