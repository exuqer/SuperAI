from server.services.lexeme import lexeme_service
import sqlite3
import math
from collections import Counter

# Let's trace through the actual accumulate_context function step by step
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

# Simulate what accumulate_context does for ONE sentence
sentence_lexeme_ids = [l1["id"], l2["id"]]  # "круглый мяч"
window_size = 5

print(f'\nProcessing sentence: {sentence_lexeme_ids}')

cooccur_counts = Counter()
lexeme_totals = Counter()

for i, lexeme_id in enumerate(sentence_lexeme_ids):
    lexeme_totals[lexeme_id] += 1
    print(f'  i={i}, lexeme_id={lexeme_id}, lexeme_totals[{lexeme_id}]={lexeme_totals[lexeme_id]}')
    # Look at neighbors within window
    for j in range(max(0, i - window_size), min(len(sentence_lexeme_ids), i + window_size + 1)):
        if i == j:
            continue
        context_id = sentence_lexeme_ids[j]
        distance = abs(i - j)
        weight = 1.0 / (1.0 + distance)
        pair = tuple(sorted([lexeme_id, context_id]))
        cooccur_counts[pair] += weight
        lexeme_totals[context_id] += 1
        print(f'    j={j}, context_id={context_id}, distance={distance}, weight={weight}, pair={pair}, cooccur_counts[{pair}]={cooccur_counts[pair]}, lexeme_totals[{context_id}]={lexeme_totals[context_id]}')

print(f'\nCo-occurrence counts: {dict(cooccur_counts)}')
print(f'Lexeme totals: {dict(lexeme_totals)}')

total_windows = sum(lexeme_totals.values())
print(f'Total windows: {total_windows}')

# Now try to insert into context_vectors
with conn:
    for (lex_a, lex_b), cooccur in cooccur_counts.items():
        p_ab = cooccur / total_windows
        p_a = lexeme_totals[lex_a] / total_windows
        p_b = lexeme_totals[lex_b] / total_windows
        print(f'Pair ({lex_a},{lex_b}): p_ab={p_ab:.6f}, p_a={p_a:.6f}, p_b={p_b:.6f}')
        if p_a > 0 and p_b > 0:
            pmi = math.log(p_ab / (p_a * p_b))
            ppmi = max(0.0, pmi)
            print(f'  PMI={pmi:.6f}, PPMI={ppmi:.6f}')
            if ppmi > 0:
                print(f'  -> Inserting/Updating context_vectors')
                for l1_id, l2_id in [(lex_a, lex_b), (lex_b, lex_a)]:
                    row = conn.execute(
                        "SELECT weight, count FROM context_vectors WHERE lexeme_id = ? AND context_lexeme_id = ?",
                        (l1_id, l2_id)
                    ).fetchone()
                    print(f'  Existing row for ({l1_id},{l2_id}): {row}')
                    
                    if row:
                        new_count = row["count"] + 1
                        new_weight = row["weight"] * 0.9 + ppmi * 0.1
                        print(f'  Updating: weight={new_weight:.6f}, count={new_count}')
                        conn.execute(
                            """UPDATE context_vectors SET
                            weight = ?, count = ?, updated_at = ?
                            WHERE lexeme_id = ? AND context_lexeme_id = ?""",
                            (new_weight, new_count, "2024-01-01T00:00:00", l1_id, l2_id)
                        )
                    else:
                        print(f'  Inserting: weight={ppmi:.6f}')
                        conn.execute(
                            """INSERT INTO context_vectors
                            (lexeme_id, context_lexeme_id, weight, count, updated_at)
                            VALUES (?, ?, ?, 1, ?)""",
                            (l1_id, l2_id, ppmi, "2024-01-01T00:00:00")
                        )
    conn.commit()

# Check results
print('\n=== Context Vectors after manual insert ===')
for row in conn.execute('SELECT * FROM context_vectors'):
    print(dict(row))