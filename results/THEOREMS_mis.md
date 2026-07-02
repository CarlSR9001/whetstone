# Machine-validated conjectures: mis

Claim template: for every graph satisfying the predicate, degree-ascending greedy independent set reaches the independence number.

Nothing here is proved. Each entry survived: exhaustive verification on all
graphs with n <= 6, a hostile stress pool at n in (7, 8, 10) (random,
deterministic adversary families, adversary library), and simulated-annealing
counterexample search inside its own predicate class.

| predicate | origin | small-n support | stress matches | anneal budget |
| --- | --- | --- | --- | --- |
| `(is_bipartite) and (m <= 4)` | repair | 2038 | 86 | 8x1200 |
| `(is_triangle_free) and (m <= 4)` | repair | 2038 | 86 | 8x1200 |
| `(is_bipartite and is_triangle_free) and (m <= 4)` | repair | 2038 | 86 | 8x1200 |
| `(is_bipartite and not is_connected) and (m <= 4)` | repair | 1889 | 86 | 8x1200 |
| `(is_bipartite and has_isolated_vertex) and (m <= 4)` | repair | 1512 | 86 | 8x1200 |
| `(is_bipartite and not is_regular) and (m <= 4)` | repair | 2010 | 86 | 8x1200 |
| `(is_triangle_free and not is_connected) and (m <= 4)` | repair | 1889 | 86 | 8x1200 |
| `(is_triangle_free and has_isolated_vertex) and (m <= 4)` | repair | 1512 | 86 | 8x1200 |
| `(is_triangle_free and not is_regular) and (m <= 4)` | repair | 2010 | 86 | 8x1200 |
| `(max_degree_le_2) and (m <= 5)` | repair | 2093 | 85 | 8x1200 |
| `(is_triangle_free and max_degree_le_2) and (m <= 5)` | repair | 1928 | 81 | 8x1200 |
| `(is_bipartite and max_degree_le_2) and (m <= 5)` | repair | 1844 | 80 | 8x1200 |
| `(is_bipartite and max_degree >= 2) and (m <= 4)` | repair | 1919 | 68 | 8x1200 |
| `(is_triangle_free and max_degree >= 2) and (m <= 4)` | repair | 1919 | 68 | 8x1200 |
| `(is_bipartite and m >= 3) and (m <= 4)` | repair | 1829 | 61 | 8x1200 |
| `(is_triangle_free and m >= 3) and (m <= 4)` | repair | 1829 | 61 | 8x1200 |
| `(max_degree_le_2 and m >= 3) and (m <= 5)` | repair | 1884 | 60 | 8x1200 |
| `(is_bipartite and m >= 4) and (m <= 4)` | repair | 1268 | 42 | 8x1200 |
| `(is_triangle_free and m >= 4) and (m <= 4)` | repair | 1268 | 42 | 8x1200 |
| `is_tree and max_degree >= 3` | candidate | 1005 | 33 | 8x1200 |
| `(is_bipartite and m >= 5) and (has_universal_vertex)` | repair | 6 | 14 | 8x1200 |
| `(is_triangle_free and m >= 5) and (has_universal_vertex)` | repair | 6 | 14 | 8x1200 |
| `(max_degree_le_2 and not has_isolated_vertex) and (m <= 5)` | repair | 885 | 3 | 8x1200 |
| `(is_connected and is_bipartite) and (m <= 4)` | repair | 149 | 0 (vacuous) | 8x1200 |
| `(is_triangle_free and is_regular) and (m <= 5)` | repair | 40 | 0 (vacuous) | 8x1200 |
| `(is_connected and max_degree_le_2) and (m <= 5)` | repair | 453 | 0 (vacuous) | 8x1200 |
| `(is_bipartite and not has_isolated_vertex) and (m <= 4)` | repair | 526 | 0 (vacuous) | 8x1200 |
| `is_tree and has_isolated_vertex` | candidate | 1 | 0 (vacuous) | 8x1200 |
| `is_tree and is_regular` | candidate | 2 | 0 (vacuous) | 8x1200 |
| `is_forest and is_regular` | candidate | 25 | 0 (vacuous) | 8x1200 |
| `(is_bipartite and is_connected) and (m <= 4)` | repair | 149 | 0 (vacuous) | 8x1200 |
| `(is_bipartite and is_regular) and (m <= 4)` | repair | 28 | 0 (vacuous) | 8x1200 |
| `(max_degree_le_2 and is_connected) and (m <= 5)` | repair | 453 | 0 (vacuous) | 8x1200 |
| `(max_degree_le_2 and is_regular) and (m <= 5)` | repair | 41 | 0 (vacuous) | 8x1200 |
