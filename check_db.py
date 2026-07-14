import sqlite3
conn = sqlite3.connect('.superai/state.sqlite')
conn.row_factory = sqlite3.Row

# Check lexeme clouds
cur = conn.execute('SELECT * FROM clouds WHERE layer_id = (SELECT id FROM layers WHERE name="lexeme")')
print('Lexeme clouds:', [dict(row) for row in cur.fetchall()])

cur = conn.execute('SELECT * FROM word_form_to_lexeme')
print('Word form to lexeme:', [dict(row) for row in cur.fetchall()])

# Check scenes
cur = conn.execute('SELECT * FROM scenes')
print('Scenes:', [dict(row) for row in cur.fetchall()])

# Check word_form clouds
cur = conn.execute('SELECT * FROM clouds WHERE layer_id = (SELECT id FROM layers WHERE name="word_form")')
print('Word form clouds:', [dict(row) for row in cur.fetchall()])