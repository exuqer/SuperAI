import sys, os, tempfile, atexit
from pathlib import Path
sys.path.insert(0, '.')

# Isolate database
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

training.train('Механик дал роботу болт.', independent_key='debug-what4')
hive_id = dialogue.create()['hive']['id']

r1 = dialogue.query(hive_id, 'Кому и кто дал болт?')
print('R1:', r1['answer']['surface'])
print('  R1 keys:', sorted(r1.keys()))

print()
print('=== FRAME after R1 ===')
for fid, frame in dialogue._frames.items():
    print(f"  frame {fid}: event_id={frame.event_id}")
    for p in frame.participants:
        print(f"    participant: lemma={p.resolved_lemma}, replaceable={p.replaceable}, morph={p.morphology_profile}")

r2 = dialogue.query(hive_id, 'Кому?')
print()
print('R2:', r2['answer']['surface'])
print('  R2 keys:', sorted(r2.keys()))
if 'release_diagnostics' in r2:
    for d in r2['release_diagnostics']:
        print(f'  diag: {d["decision"]} {d["decision_reason"]}')

r3 = dialogue.query(hive_id, 'Кто?')
print()
print('R3:', r3['answer']['surface'])
print('  R3 keys:', sorted(r3.keys()))
if 'release_diagnostics' in r3:
    for d in r3['release_diagnostics']:
        print(f'  diag: {d["decision"]} {d["decision_reason"]}')

r4 = dialogue.query(hive_id, 'Что?')
print()
print('R4:', r4['answer']['surface'])
print('R4 keys:', sorted(r4.keys()))
for d in r4.get('release_diagnostics', []):
    for c in d.get('candidates', []):
        print(f"  candidate: lemma={c.get('resolved_surface')}, score={c.get('final_score'):.3f}, "
              f"anim={c.get('animacy_score'):.3f}, conflict={c.get('animacy_conflict'):.3f}, "
              f"family={c.get('question_family_match'):.3f}")
    print(f"  decision: {d.get('decision')}, reason: {d.get('decision_reason')}")

print()
print('=== FRAME after R4 ===')
for fid, frame in dialogue._frames.items():
    print(f"  frame {fid}: event_id={frame.event_id}")
    for p in frame.participants:
        print(f"    participant: lemma={p.resolved_lemma}, replaceable={p.replaceable}")
        print(f"      observed: {[o.question_surface for o in p.observed_question_profiles]}")
        print(f"      compatible: {p.compatible_question_profiles}")