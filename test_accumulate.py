from server.services.lexeme import lexeme_service
import sqlite3

conn = sqlite3.connect('.superai/state.sqlite')
conn.row_factory = sqlite3.Row

# Clear context vectors
conn.execute('DELETE FROM context_vectors')
conn.commit()

# Get lexemes that were created
l1 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="круглый"').fetchone()
l2 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="мяч"').fetchone()

print(f'Lexeme 1: {l1}')
print(f'Lexeme 2: {l2}')

if l1 and l2:
    # Call accumulate_context directly
    print(f'Calling accumulate_context with [{l1["id"]}, {l2["id"]}]')
    lexeme_service.accumulate_context([l1["id"], l2["id"]])
    
    # Check context vectors
    print('\n=== Context Vectors ===')
    for row in conn.execute('SELECT * FROM context_vectors'):
        print(dict(row))