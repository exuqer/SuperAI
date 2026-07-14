from server.services.lexeme import lexeme_service
from server.training import TrainingManager, TrainingConfig
import sqlite3

# Clear DB
conn = sqlite3.connect('.superai/state.sqlite')
conn.execute('DELETE FROM context_vectors')
conn.execute('DELETE FROM concept_centroids')
conn.execute('DELETE FROM lexeme_concept_membership')
conn.execute('DELETE FROM clouds WHERE layer_id IN (SELECT id FROM layers WHERE name IN ("lexeme", "concept"))')
conn.commit()

# Train with multiple overlapping sentences
config = TrainingConfig(min_word_observations=1, min_concept_observations=1, enable_concept_layer=True, enable_lexeme_layer=True)
manager = TrainingManager(config)

# Train ONE sentence and debug
result = manager.learn('круглый мяч')
print('Result:', result)

# Check if lexemes were created
conn = sqlite3.connect('.superai/state.sqlite')
conn.row_factory = sqlite3.Row

print('\n=== Lexemes ===')
for row in conn.execute('SELECT * FROM lexemes'):
    print(dict(row))

print('\n=== Word Form to Lexeme ===')
for row in conn.execute('SELECT * FROM word_form_to_lexeme'):
    print(dict(row))

# Get the lexeme IDs from the sentence
print('\n=== Calling accumulate_context manually ===')
l1 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="круглый"').fetchone()
l2 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="мяч"').fetchone()
print(f'Lexeme 1: {l1["id"] if l1 else None}')
print(f'Lexeme 2: {l2["id"] if l2 else None}')

if l1 and l2:
    lexeme_service.accumulate_context([l1["id"], l2["id"]])
    print('\n=== Context Vectors after manual call ===')
    for row in conn.execute('SELECT * FROM context_vectors'):
        print(dict(row))