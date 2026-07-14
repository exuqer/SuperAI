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

# Train with multiple overlapping sentences
config = TrainingConfig(min_word_observations=1, min_concept_observations=1, enable_concept_layer=True, enable_lexeme_layer=True)
manager = TrainingManager(config)

# Test just one learn call
result = manager.learn('круглый мяч')
print('Result:', result)

# Check if lexemes were created
conn = sqlite3.connect('.superai/state.sqlite')
conn.row_factory = sqlite3.Row

print('\n=== Lexeme Clouds ===')
lex_layer = conn.execute('SELECT id FROM layers WHERE name="lexeme"').fetchone()
if lex_layer:
    for row in conn.execute('SELECT * FROM clouds WHERE layer_id=?', (lex_layer['id'],)):
        print(dict(row))

print('\n=== Context Vectors ===')
for row in conn.execute('SELECT * FROM context_vectors'):
    print(dict(row))