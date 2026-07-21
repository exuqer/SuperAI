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

training.train('Механик дал роботу болт.', independent_key='debug-what6')
hive_id = dialogue.create()['hive']['id']

r1 = dialogue.query(hive_id, 'Кому и кто дал болт?')
print('R1:', r1['answer']['surface'])

r2 = dialogue.query(hive_id, 'Кому?')
print('R2:', r2['answer']['surface'])

r3 = dialogue.query(hive_id, 'Кто?')
print('R3:', r3['answer']['surface'])

# Debug: what does gap release give us?
from server.v2.graph_repository import GraphRepository
from server.v2.gap_release import GapReleaseSelector, GapReleaseDiagnostic
from server.v2.event_binding_frame import EventBindingFrame, FrameStatus, ParticipantOrigin, EventBindingFrameParticipant, ObservedQuestionProfile
import json

# Get the state before the "Что?" query
hive_state = dialogue.get(hive_id)
print(f"PREV_STATE bindings: {hive_state.get('selected_bindings')}")

with repository.transaction() as conn:
    frames = conn.execute("SELECT * FROM event_binding_frames WHERE status='ACTIVE'").fetchall()
    for fr in frames:
        participants = conn.execute("SELECT * FROM event_binding_frame_participants WHERE frame_id=?", (fr['id'],)).fetchall()
        print(f"FRAME {fr['id']}")
        for p in participants:
            print(f"  participant: node_id={p['participant_node_id']}, lemma={p['resolved_lemma']}, latest_source_binding_id={p['latest_source_binding_id']}")
        
        # Check accepted bindings from last turn
        last_turn = conn.execute("SELECT * FROM dialogue_turns ORDER BY turn_index DESC LIMIT 1").fetchone()
        if last_turn:
            accepted = conn.execute("SELECT * FROM candidate_bindings WHERE query_graph_id=? OR 1=0 ORDER BY total_score DESC", (last_turn['query_graph_id'],)).fetchall()
            print(f"ACCEPTED bindings for last turn: {len(accepted)}")
            for a in accepted:
                print(f"  binding: resolved_node_id={a['resolved_node_id']}, resolved_surface={a['resolved_surface']}")

r4 = dialogue.query(hive_id, 'Что?')
print('R4:', r4['answer']['surface'])
for d in r4.get('release_diagnostics', []):
    print(f"  decision={d['decision']}, reason={d['decision_reason']}")
    for c in d.get('candidates', []):
        print(f"    {c['resolved_surface']}: score={c['final_score']:.3f}")

# Check debug trace for Что?
print(f"TRACE gap_release_diagnostic: {r4['trace'].get('gap_release_diagnostic')}")

print()

with repository.transaction() as conn:
    rows = conn.execute("SELECT id, event_id, status FROM event_binding_frames").fetchall()
    print(f"TOTAL frames: {len(rows)}")
    for r in rows:
        print(f"  frame: {r['id'][-20:]}, event={r['event_id'][-20:]}, status={r['status']}")
        parts = conn.execute("SELECT resolved_lemma FROM event_binding_frame_participants WHERE frame_id=?", (r['id'],)).fetchall()
        for p in parts:
            print(f"    participant: {p['resolved_lemma']}")
    
    dc = conn.execute("SELECT * FROM dialogue_context_states").fetchone()
    print(f"DC: frame_id={dc['active_event_binding_frame_id']}, resolved={dc['last_resolved_turn_id']}")
    
    print()
    rel = conn.execute("SELECT * FROM gap_release_diagnostics").fetchall()
    print(f"gap_release_diagnostics: {len(rel)}")
    
    scores = conn.execute("SELECT diagnostic_id, resolved_surface, final_score FROM gap_release_candidate_scores ORDER BY rank").fetchall()
    print(f"gap_release_candidate_scores: {len(scores)}")
    for s in scores:
        print(f"  diag={s['diagnostic_id'][-20:]}, surface={s['resolved_surface']}, score={s['final_score']}")
    
    hyp = conn.execute("SELECT interpretation_type, total_score, selected FROM query_interpretation_hypotheses").fetchall()
    print(f"query_interpretation_hypotheses: {len(hyp)}")
    for h in hyp:
        print(f"  {h['interpretation_type']}: score={h['total_score']}, selected={h['selected']}")