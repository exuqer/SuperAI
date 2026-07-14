from server.training import TrainingManager, TrainingConfig
import sqlite3

# Clear DB
conn = sqlite3.connect('.superai/state.sqlite')
conn.execute('DELETE FROM context_vectors')
conn.execute('DELETE FROM concept_centroids')
conn.execute('DELETE FROM lexeme_concept_membership')
conn.execute('DELETE FROM clouds WHERE layer_id IN (SELECT id FROM layers WHERE name IN ("lexeme", "concept"))')
conn.commit()

config = TrainingConfig(min_word_observations=1, min_concept_observations=1, enable_concept_layer=True, enable_lexeme_layer=True, enable_scene_layer=True)
manager = TrainingManager(config)

# Add debug to see what's happening
print('Calling learn...')
result = manager.learn('круглый мяч')
print('Result created_clouds:', [c['layer'] for c in result['details']['created_clouds']])

# Check lexemes
conn = sqlite3.connect('.superai/state.sqlite')
conn.row_factory = sqlite3.Row
print('\nLexeme clouds:')
lex_layer = conn.execute('SELECT id FROM layers WHERE name="lexeme"').fetchone()
if lex_layer:
    for row in conn.execute('SELECT * FROM clouds WHERE layer_id=?', (lex_layer['id'],)):
        print(dict(row))

print('\nContext Vectors:')
for row in conn.execute('SELECT * FROM context_vectors'):
    print(dict(row))