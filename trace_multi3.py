from server.services.lexeme import lexeme_service
import sqlite3
import math
from collections import Counter
from datetime import datetime

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

# Multiple overlapping sentences - this should produce positive PPMI
sentences = [
    [l1["id"], l2["id"]],  # круглый мяч
    [l1["id"], l3["id"]],  # круглый стол
    [l4["id"], l2["id"]],  # бросить мяч
    [l5["id"], l2["id"]],  # играть мяч
]

# Use the actual service method which should work correctly
for epoch in range(5):
    print(f'\nEpoch {epoch+1}')
    for sent in sentences:
        lexeme_service.accumulate_context(sent)

# Check context vectors
print('\n=== Context Vectors ===')
for row in conn.execute('SELECT * FROM context_vectors'):
    print(dict(row))