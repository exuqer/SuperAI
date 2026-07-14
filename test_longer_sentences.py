from server.training import TrainingManager, TrainingConfig
import sqlite3

# Clear DB
conn = sqlite3.connect('.superai/state.sqlite')
conn.execute('DELETE FROM context_vectors')
conn.execute('DELETE FROM concept_centroids')
conn.execute('DELETE FROM lexeme_concept_membership')
conn.execute('DELETE FROM clouds WHERE layer_id IN (SELECT id FROM layers WHERE name IN ("lexeme", "concept"))')
conn.commit()

# Train with LONGER sentences
config = TrainingConfig(
    min_word_observations=1, 
    min_concept_observations=1, 
    enable_concept_layer=True, 
    enable_lexeme_layer=True,
    enable_scene_layer=True
)
manager = TrainingManager(config)

# Train with longer sentences that have overlapping words
for epoch in range(5):
    print(f'Epoch {epoch+1}')
    manager.learn('круглый мяч летит быстро')
    manager.learn('круглый стол стоит в углу') 
    manager.learn('бросить мяч в корзину')
    manager.learn('играть мяч на поле')

# Check results
conn = sqlite3.connect('.superai/state.sqlite')
conn.row_factory = sqlite3.Row

print('\n=== Context Vectors ===')
for row in conn.execute('SELECT * FROM context_vectors'):
    print(dict(row))

print('\n=== Concept Clouds ===')
cl = conn.execute('SELECT id FROM layers WHERE name="concept"').fetchone()
if cl:
    for row in conn.execute('SELECT * FROM clouds WHERE layer_id=?', (cl['id'],)):
        print(dict(row))

print('\n=== Concept Centroids ===')
for row in conn.execute('SELECT * FROM concept_centroids'):
    print(dict(row))

print('\n=== Lexeme Concept Membership ===')
for row in conn.execute('SELECT * FROM lexeme_concept_membership'):
    print(dict(row))