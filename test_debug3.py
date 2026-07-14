from server.training import TrainingManager, TrainingConfig
from server.services.lexeme import lexeme_service
import sqlite3

# Clear DB
conn = sqlite3.connect('.superai/state.sqlite')
conn.execute('DELETE FROM context_vectors')
conn.execute('DELETE FROM concept_centroids')
conn.execute('DELETE FROM lexeme_concept_membership')
conn.execute('DELETE FROM clouds WHERE layer_id IN (SELECT id FROM layers WHERE name IN ("lexeme", "concept"))')
conn.commit()

# Train with multiple sentences
config = TrainingConfig(
    min_word_observations=1, 
    min_concept_observations=1, 
    enable_concept_layer=True, 
    enable_lexeme_layer=True,
    enable_scene_layer=True
)
manager = TrainingManager(config)

# Train multiple times with overlapping sentences
print("=== Training ===")
for epoch in range(3):
    print(f'Epoch {epoch+1}')
    manager.learn('круглый мяч')
    manager.learn('круглый стол') 
    manager.learn('бросить мяч')
    manager.learn('играть мяч')

# Check context vectors after training
print('\n=== Checking context vectors ===')
conn = sqlite3.connect('.superai/state.sqlite')
conn.row_factory = sqlite3.Row

print('\n=== Context Vectors ===')
for row in conn.execute('SELECT * FROM context_vectors'):
    print(dict(row))

# Check lexeme context vectors manually
print('\n=== Manual get_context_vector ===')
l1 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="круглый"').fetchone()
l2 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="мяч"').fetchone()
l3 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="стол"').fetchone()
l4 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="бросить"').fetchone()
l5 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="играть"').fetchone()

print(f'Lexeme IDs: круглый={l1["id"]}, мяч={l2["id"]}, стол={l3["id"]}, бросить={l4["id"]}, играть={l5["id"]}')

if l1 and l2:
    vec = lexeme_service.get_context_vector(l1["id"])
    print(f'Context vector for "круглый" (id={l1["id"]}): {vec}')
    
    vec = lexeme_service.get_context_vector(l2["id"])
    print(f'Context vector for "мяч" (id={l2["id"]}): {vec}')

# Check if there are any sentence lexeme IDs being tracked
print('\n=== Check sentences ===')
for row in conn.execute('SELECT * FROM scenes'):
    print(dict(row))

# Now test accumulate_context directly
print('\n=== Test accumulate_context directly ===')
conn.execute('DELETE FROM context_vectors')
conn.commit()

# Use the actual lexeme IDs
lexeme_ids = [l1["id"], l2["id"]]
print(f'Calling accumulate_context with {lexeme_ids}')
lexeme_service.accumulate_context(lexeme_ids)

print('\n=== Context Vectors after direct call ===')
for row in conn.execute('SELECT * FROM context_vectors'):
    print(dict(row))