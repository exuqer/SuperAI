from server.services.lexeme import lexeme_service
import sqlite3

conn = sqlite3.connect('.superai/state.sqlite')
conn.row_factory = sqlite3.Row

# Clear context vectors
conn.execute('DELETE FROM context_vectors')
conn.commit()

# Get lexemes
l1 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="круглый"').fetchone()
l2 = conn.execute('SELECT * FROM lexemes WHERE canonical_form="мяч"').fetchone()

print(f'Lexeme 1: {dict(l1)}')
print(f'Lexeme 2: {dict(l2)}')

if l1 and l2:
    # Call accumulate_context directly with debug
    print(f'\nCalling accumulate_context with [{l1["id"]}, {l2["id"]}]')
    sentence_lexeme_ids = [l1["id"], l2["id"]]
    window_size = 5
    
    from collections import Counter
    
    # Count co-occurrences
    cooccur_counts = Counter()
    lexeme_totals = Counter()
    
    for i, lexeme_id in enumerate(sentence_lexeme_ids):
        lexeme_totals[lexeme_id] += 1
        print(f'  i={i}, lexeme_id={lexeme_id}')
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
            print(f'    j={j}, context_id={context_id}, distance={distance}, weight={weight}, pair={pair}')
    
    print(f'\nCo-occurrence counts: {dict(cooccur_counts)}')
    print(f'Lexeme totals: {dict(lexeme_totals)}')
    
    total_windows = sum(lexeme_totals.values())
    print(f'Total windows: {total_windows}')
    
    import math
    for (lex_a, lex_b), cooccur in cooccur_counts.items():
        p_ab = cooccur / total_windows
        p_a = lexeme_totals[lex_a] / total_windows
        p_b = lexeme_totals[lex_b] / total_windows
        print(f'Pair ({lex_a},{lex_b}): p_ab={p_ab:.4f}, p_a={p_a:.4f}, p_b={p_b:.4f}')
        if p_a > 0 and p_b > 0:
            pmi = math.log(p_ab / (p_a * p_b))
            ppmi = max(0.0, pmi)
            print(f'  PMI={pmi:.4f}, PPMI={ppmi:.4f}')
            if ppmi > 0:
                print('  Would insert into context_vectors!')