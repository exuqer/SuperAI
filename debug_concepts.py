from server.training import TrainingManager, TrainingConfig
from server.services.lexeme import lexeme_service
import sqlite3

config = TrainingConfig(min_word_observations=1, min_concept_observations=1, enable_concept_layer=True)
manager = TrainingManager(config)

# Train
for _ in range(3):
    manager.learn('круглый мяч')
    manager.learn('круглый стол')
    manager.learn('бросить мяч')
    manager.learn('играть мяч')

# Use database directly
conn = sqlite3.connect('.superai/state.sqlite')
conn.row_factory = sqlite3.Row

print('=== Lexemes ===')
for row in conn.execute('SELECT * FROM lexemes'):
    print(dict(row))

print('\n=== Context Vectors ===')
for row in conn.execute('SELECT * FROM context_vectors'):
    print(dict(row))

print('\n=== Clouds in concept layer ===')
concept_layer = conn.execute('SELECT id FROM layers WHERE name="concept"').fetchone()
if concept_layer:
    for row in conn.execute('SELECT * FROM clouds WHERE layer_id=?', (concept_layer['id'],)):
        print(dict(row))
else:
    print('No concept layer')

print('\n=== Concept Centroids ===')
for row in conn.execute('SELECT * FROM concept_centroids'):
    print(dict(row))

print('\n=== Lexeme Concept Membership ===')
for row in conn.execute('SELECT * FROM lexeme_concept_membership'):
    print(dict(row))