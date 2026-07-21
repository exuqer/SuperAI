import json
from tests.test_dialogue_context_v29 import services

_, train, dialogue = services()
train.train("Механик дал роботу болт.", independent_key="scenario-a")
hid = dialogue.create()["hive"]["id"]

r1 = dialogue.query(hid, "Кому и кто дал болт?")
print("Turn 1:", r1["answer"]["surface"])

# After turn 1, check if frame exists - use the dialogue's own repository
with dialogue.repository.transaction() as conn:
    frames = conn.execute("SELECT id, event_id, status FROM event_binding_frames LIMIT 10").fetchall()
    print(f"After turn 1: {len(frames)} event_binding_frames")
    for f in frames:
        print(f"  frame: {f['id']}, event: {f['event_id']}, status: {f['status']}")
        participants = conn.execute("SELECT id, resolved_lemma, replaceable, last_released_turn FROM event_binding_frame_participants WHERE frame_id=?", (f['id'],)).fetchall()
        for p in participants:
            print(f"    participant: {p['resolved_lemma']} replaceable={p['replaceable']} last_released={p['last_released_turn']}")
    
    # Check dialogue context  
    ctx = conn.execute("SELECT * FROM dialogue_context_states LIMIT 10").fetchall()
    print(f"  dialogue_context_states: {len(ctx)}")
    for c in ctx:
        print(f"    ctx: conversation_id={c['conversation_id']} last_resolved={c['last_resolved_turn_id']} active_frame={c['active_event_binding_frame_id']}")
print()

# Now turn 2
r2 = dialogue.query(hid, "Кому?")
print("Turn 2:", r2["answer"]["surface"])
tr2 = r2.get("trace", {})
print("  has gap_release_diagnostic:", "gap_release_diagnostic" in tr2)
print("  trace keys:", list(tr2.keys())[:10])
print("  structural_signature:", tr2.get("structural_signature", "N/A"))
print()

# Check frames again - use the dialogue's own repository
with dialogue.repository.transaction() as conn:
    frames = conn.execute("SELECT id, event_id, status FROM event_binding_frames LIMIT 10").fetchall()
    print(f"After turn 2: {len(frames)} event_binding_frames")
    for f in frames:
        print(f"  frame: {f['id']}, event: {f['event_id']}, status: {f['status']}")
        participants = conn.execute("SELECT id, resolved_lemma, replaceable, last_released_turn FROM event_binding_frame_participants WHERE frame_id=?", (f['id'],)).fetchall()
        for p in participants:
            print(f"    participant: {p['resolved_lemma']} replaceable={p['replaceable']} last_released={p['last_released_turn']}")
    
    # Check gap_release_diagnostics
    diags = conn.execute("SELECT * FROM gap_release_diagnostics").fetchall()
    print(f"  gap_release_diagnostics: {len(diags)}")
    for d in diags:
        print(f"    diagnostic: {d['id']} decision={d['decision']} reason={d['decision_reason']}")
    
    scores = conn.execute("SELECT * FROM gap_release_candidate_scores").fetchall()
    print(f"  gap_release_candidate_scores: {len(scores)}")
    for s in scores:
        print(f"    participant={s['resolved_surface']} score={s['final_score']} rank={s['rank']} accepted={s['accepted']}")