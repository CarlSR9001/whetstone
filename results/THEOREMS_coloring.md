# Machine-validated conjectures: coloring

Claim template: for every graph satisfying the predicate, degree-descending greedy coloring uses exactly the chromatic number.

Nothing here is proved. Each entry survived: exhaustive verification on all
graphs with n <= 6, a hostile stress pool at n in (7, 8, 10) (random,
deterministic adversary families, adversary library), and simulated-annealing
counterexample search inside its own predicate class.

| predicate | origin | small-n support | stress matches | anneal budget |
| --- | --- | --- | --- | --- |
| `(is_bipartite) and (m <= 4)` | repair | 2038 | 86 | 8x800 |
| `(is_triangle_free) and (m <= 4)` | repair | 2038 | 86 | 8x800 |
| `(is_bipartite and is_triangle_free) and (m <= 4)` | repair | 2038 | 86 | 8x800 |
| `(is_bipartite and not is_regular) and (m <= 4)` | repair | 2010 | 86 | 8x800 |
| `(is_triangle_free and not is_regular) and (m <= 4)` | repair | 2010 | 86 | 8x800 |
| `(is_forest) and (m <= 4)` | repair | 1975 | 85 | 8x800 |
| `(is_forest and not is_regular) and (m <= 4)` | repair | 1950 | 85 | 8x800 |
| `(max_degree_le_2) and (m <= 4)` | repair | 1544 | 72 | 8x800 |
| `(max_degree_le_2 and not is_regular) and (m <= 4)` | repair | 1515 | 72 | 8x800 |
| `(is_bipartite and max_degree_le_2) and (m <= 4)` | repair | 1439 | 68 | 8x800 |
| `(is_bipartite and max_degree >= 2) and (m <= 4)` | repair | 1919 | 68 | 8x800 |
| `(is_triangle_free and max_degree >= 2) and (m <= 4)` | repair | 1919 | 68 | 8x800 |
| `(is_triangle_free and max_degree_le_2) and (m <= 4)` | repair | 1439 | 68 | 8x800 |
| `(is_forest and max_degree_le_2) and (m <= 4)` | repair | 1376 | 67 | 8x800 |
| `(is_forest and max_degree >= 2) and (m <= 4)` | repair | 1856 | 67 | 8x800 |
| `(is_bipartite and m >= 3) and (m <= 4)` | repair | 1829 | 61 | 8x800 |
| `(is_triangle_free and m >= 3) and (m <= 4)` | repair | 1829 | 61 | 8x800 |
| `(is_forest and m >= 3) and (m <= 4)` | repair | 1766 | 60 | 8x800 |
| `(max_degree_le_2 and m >= 3) and (m <= 4)` | repair | 1335 | 47 | 8x800 |
| `(is_bipartite and m >= 4) and (m <= 4)` | repair | 1268 | 42 | 8x800 |
| `(is_triangle_free and m >= 4) and (m <= 4)` | repair | 1268 | 42 | 8x800 |
| `(is_forest and m >= 4) and (m <= 4)` | repair | 1205 | 41 | 8x800 |
| `(is_tree and m >= 5) and (has_universal_vertex)` | repair | 6 | 14 | 8x800 |
| `(is_forest and m >= 5) and (has_universal_vertex)` | repair | 6 | 14 | 8x800 |
| `(is_triangle_free and m >= 5) and (girth == 5)` | repair | 444 | 12 | 8x800 |
| `(is_connected and is_triangle_free) and (girth == 5)` | repair | 372 | 6 | 8x800 |
| `(is_triangle_free and is_connected) and (girth == 5)` | repair | 372 | 6 | 8x800 |
| `(is_tree) and (m <= 4)` | repair | 146 | 0 (vacuous) | 8x800 |
| `(is_connected and is_bipartite) and (n <= 5)` | repair | 219 | 0 (vacuous) | 8x800 |
| `(is_triangle_free and is_regular) and (m <= 5)` | repair | 40 | 0 (vacuous) | 8x800 |
| `(is_connected and max_degree_le_2) and (n <= 5)` | repair | 93 | 0 (vacuous) | 8x800 |
| `(is_forest and not has_isolated_vertex) and (m <= 4)` | repair | 523 | 0 (vacuous) | 8x800 |
| `(is_bipartite and not has_isolated_vertex) and (m <= 4)` | repair | 526 | 0 (vacuous) | 8x800 |
| `(is_triangle_free and not has_isolated_vertex) and (m <= 4)` | repair | 526 | 0 (vacuous) | 8x800 |
| `(is_tree and is_connected) and (m <= 4)` | repair | 146 | 0 (vacuous) | 8x800 |
| `is_tree and has_isolated_vertex` | candidate | 1 | 0 (vacuous) | 8x800 |
| `(is_tree and not has_isolated_vertex) and (m <= 4)` | repair | 145 | 0 (vacuous) | 8x800 |
| `is_tree and is_regular` | candidate | 2 | 0 (vacuous) | 8x800 |
| `(is_tree and not is_regular) and (m <= 4)` | repair | 144 | 0 (vacuous) | 8x800 |
| `(is_tree and m >= 3) and (m <= 4)` | repair | 141 | 0 (vacuous) | 8x800 |
| `(is_tree and m >= 4) and (m <= 4)` | repair | 125 | 0 (vacuous) | 8x800 |
| `(is_tree and max_degree >= 2) and (m <= 4)` | repair | 144 | 0 (vacuous) | 8x800 |
| `(is_tree and max_degree_le_2) and (m <= 4)` | repair | 77 | 0 (vacuous) | 8x800 |
| `(is_forest and is_connected) and (m <= 4)` | repair | 146 | 0 (vacuous) | 8x800 |
| `is_forest and is_regular` | candidate | 25 | 0 (vacuous) | 8x800 |
| `(is_bipartite and is_connected) and (n <= 5)` | repair | 219 | 0 (vacuous) | 8x800 |
| `(is_bipartite and is_regular) and (m <= 4)` | repair | 28 | 0 (vacuous) | 8x800 |
| `(is_bipartite and m >= 5) and (n <= 5)` | repair | 70 | 0 (vacuous) | 8x800 |
| `(max_degree_le_2 and is_connected) and (n <= 5)` | repair | 93 | 0 (vacuous) | 8x800 |
| `(max_degree_le_2 and not has_isolated_vertex) and (m <= 4)` | repair | 408 | 0 (vacuous) | 8x800 |
| `(max_degree_le_2 and is_regular) and (m <= 5)` | repair | 41 | 0 (vacuous) | 8x800 |
