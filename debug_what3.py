import sys, os, tempfile, atexit
from pathlib import Path
sys.path.insert(0, '.')

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

training.train('Механик дал роботу болт.', independent_key='debug-what5')
hive_id = dialogue.create()['hive']['id']

# Check initial frames
with repository.transaction() as conn:
    rows = conn.execute("SELECT * FROM event_binding_frames").fetchall()
    print(f"INITIAL frames in DB: {len(rows)}")
    dc = conn.execute("SELECT active_event_binding_frame_id FROM dialogue_context_states").fetchone()
    print(f"INITIAL DC frame_id: {dc['active_event_binding_frame_id'] if dc else 'NO DC'}")

r1 = dialogue.query(hive_id, 'Кому и кто дал болт?')
print('R1:', r1['answer']['surface'])

with repository.transaction() as conn:
    rows = conn.execute("SELECT id, event_id, status FROM event_binding_frames").fetchall()
    print(f"AFTER R1 frames: {len(rows)}")
    for r in rows:
        print(f"  frame: {r['id'][-20:]}, event={r['event_id'][-20:]}, status={r['status']}")
    dc = conn.execute("SELECT active_event_binding_frame_id, last_resolved_turn_id FROM dialogue_context_states").fetchone()
    print(f"AFTER R1 DC: frame_id={dc['active_event_binding_frame_id']}, resolved={dc['last_resolved_turn_id']}")
    
    # Check participants
    if rows:
        parts = conn.execute("SELECT * FROM event_binding_frame_participants WHERE frame_id=?", (rows[0]['id'],)).fetchall()
        print(f"  participants: {len(parts)}")

r2 = dialogue.query(hive_id, 'Кому?')
print('R2:', r2['answer']['surface'])

with repository.transaction() as conn:
    rows = conn.execute("SELECT id, event_id, status FROM event_binding_frames").fetchall()
    print(f"AFTER R2 frames: {len(rows)}")
    dc = conn.execute("SELECT active_event_binding_frame_id, last_resolved_turn_id FROM dialogue_context_states").fetchone()
    print(f"AFTER R2 DC: frame_id={dc['active_event_binding_frame_id']}, resolved={dc['last_resolved_turn_id']}")