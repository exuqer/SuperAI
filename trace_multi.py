from server.services.lexeme import lexeme_service
import sqlite3
import math
from collections import Counter

conn = sqlite3.connect('.superai/state.sqlite')
conn.row_factory = sqlite3.Row

# Clear context vectors
conn.execute('DELETE FROM context_vectors')
conn.commit()

# Get lexemes
l1 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="круглый"').fetchone()
l2 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="мяч"').fetchone()
l3 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="стол"').fetchone()
l4 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="бросить"').fetchone()
l5 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="играть"').fetchone()

print(f'Lexemes: круглый={l1["id"]}, мяч={l2["id"]}, стол={l3["id"]}, бросить={l4["id"]}, играть={l5["id"]}')

# Multiple overlapping sentences
sentences = [
    [l1["id"], l2["id"]],  # круглый мяч
    [l1["id"], l3["id"]],  # круглый стол
    [l4["id"], l2["id"]],  # бросить мяч
    [l5["id"], l2["id"]],  # играть мяч
]

# Train multiple times to accumulate statistics
for epoch in range(10):
    print(f'\nEpoch {epoch+1}')
    for sent in sentences:
        lexeme_service.accumulate_context(sent)

# Check context vectors
print('\n=== Context Vectors ===')
for row in conn.execute('SELECT * FROM context_vectors'):
    print(dict(row))

# Check context vectors for key lexemes
print('\n=== Context vector for "мяч" (lexeme 2) ===')
vec = lexeme_service.get_context_vector(l2["id"])
print(vec)

print('\n=== Context vector for "круглый" (lexeme 1) ===')
vec = lexeme_service.get_context_vector(l1["id"])
print(vec)