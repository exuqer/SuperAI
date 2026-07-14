import sqlite3
conn = sqlite3.connect(".superai/state.sqlite")
conn.row_factory = sqlite3.Row
conn.execute('UPDATE layers SET order_index=4 WHERE name="concept"')
conn.execute('UPDATE layers SET order_index=5 WHERE name="scene"')
conn.execute('UPDATE layers SET order_index=6 WHERE name="context"')
conn.commit()
cur = conn.execute("SELECT * FROM layers ORDER BY order_index")
print([dict(row) for row in cur.fetchall()])