"""Code tasks: the first non-toy exam domain, graded by hidden property checks.

The whetstone rule, applied to code: the system under exam sees a task
description; the checker it must satisfy stays private. Checkers verify
PROPERTIES — round-trips, invariants, brute-force oracles on small inputs —
so there is no canonical solution string anywhere in the system. A topological
sort task has exponentially many correct answers; all of them pass, none of
them are stored.

Submitted code runs in an isolated subprocess (``python -I``, no inherited
site-packages or environment, hard wall-clock timeout). That is process
isolation for honest measurement, not a security sandbox for hostile code —
production deployments should run graders inside their existing containment.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from bcv.examiner import ExamItem


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str  # everything the system under exam sees
    checker_source: str  # private: defines check(namespace) raising AssertionError


CODE_TASKS: tuple[CodeTask, ...] = (
    CodeTask(
        task_id="rle_roundtrip",
        prompt=(
            "Define two functions. rle_encode(s: str) -> list[tuple[str, int]] compresses a "
            "string into maximal (character, run_length) pairs. rle_decode(pairs) -> str inverts it."
        ),
        checker_source="""
import random
def check(ns):
    encode, decode = ns["rle_encode"], ns["rle_decode"]
    rng = random.Random(7)
    cases = ["", "a", "aaabbbcccd", "abababab", "zzzzzzzzzz"]
    cases += ["".join(rng.choice("abc") for _ in range(rng.randrange(0, 40))) for _ in range(30)]
    for s in cases:
        pairs = encode(s)
        assert decode(pairs) == s, f"round trip failed for {s!r}"
        assert all(count >= 1 for _, count in pairs)
        assert all(pairs[i][0] != pairs[i + 1][0] for i in range(len(pairs) - 1)), "runs not maximal"
""",
    ),
    CodeTask(
        task_id="merge_intervals",
        prompt=(
            "Define merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]] that "
            "merges overlapping or touching closed integer intervals and returns them sorted."
        ),
        checker_source="""
import random
def check(ns):
    merge = ns["merge_intervals"]
    rng = random.Random(11)
    for _ in range(40):
        raw = [(a, a + rng.randrange(0, 6)) for a in (rng.randrange(0, 30) for _ in range(rng.randrange(0, 8)))]
        merged = merge(list(raw))
        assert merged == sorted(merged), "output not sorted"
        assert all(a <= b for a, b in merged)
        assert all(merged[i + 1][0] > merged[i][1] + 1 for i in range(len(merged) - 1)), "not fully merged"
        covered = {x for a, b in raw for x in range(a, b + 1)}
        covered_out = {x for a, b in merged for x in range(a, b + 1)}
        assert covered == covered_out, "coverage changed"
""",
    ),
    CodeTask(
        task_id="balanced_brackets",
        prompt=(
            "Define balanced(s: str) -> bool for strings of ()[]{} deciding whether every bracket "
            "is closed by the matching kind in the right order."
        ),
        checker_source="""
import random
def check(ns):
    balanced = ns["balanced"]
    def oracle(s):
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        for ch in s:
            if ch in '([{':
                stack.append(ch)
            elif not stack or stack.pop() != pairs[ch]:
                return False
        return not stack
    rng = random.Random(3)
    cases = ["", "()", "([]{})", "(]", "([)]", "((("]
    cases += ["".join(rng.choice("()[]{}") for _ in range(rng.randrange(0, 14))) for _ in range(60)]
    for s in cases:
        assert balanced(s) == oracle(s), f"disagrees with oracle on {s!r}"
""",
    ),
    CodeTask(
        task_id="topo_order",
        prompt=(
            "Define topo_order(n: int, edges: list[tuple[int, int]]) -> list[int] | None returning "
            "any topological order of nodes 0..n-1 for directed edges (u, v) meaning u before v, "
            "or None when the graph has a cycle."
        ),
        checker_source="""
import random
def check(ns):
    topo = ns["topo_order"]
    rng = random.Random(5)
    for _ in range(30):
        n = rng.randrange(1, 9)
        nodes = list(range(n)); rng.shuffle(nodes)
        dag = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.3]
        order = topo(n, list(dag))
        assert order is not None and sorted(order) == list(range(n)), "not a permutation"
        position = {node: k for k, node in enumerate(order)}
        assert all(position[u] < position[v] for u, v in dag), "edge violated"
    assert topo(2, [(0, 1), (1, 0)]) is None, "cycle not detected"
    assert topo(3, [(0, 1), (1, 2), (2, 0)]) is None, "cycle not detected"
""",
    ),
    CodeTask(
        task_id="mode_stable",
        prompt=(
            "Define mode_stable(xs: list[int]) -> int | None returning the most frequent value; "
            "ties break toward the value whose FIRST occurrence is earliest. Empty list returns None."
        ),
        checker_source="""
import random
def check(ns):
    mode = ns["mode_stable"]
    def oracle(xs):
        if not xs:
            return None
        counts = {}
        for x in xs:
            counts[x] = counts.get(x, 0) + 1
        best = max(counts.values())
        for x in xs:
            if counts[x] == best:
                return x
    rng = random.Random(13)
    cases = [[], [4], [1, 2, 2, 1], [3, 1, 3, 1, 2]]
    cases += [[rng.randrange(0, 5) for _ in range(rng.randrange(0, 20))] for _ in range(50)]
    for xs in cases:
        assert mode(list(xs)) == oracle(xs), f"wrong on {xs}"
""",
    ),
    CodeTask(
        task_id="base_digits",
        prompt=(
            "Define base_digits(n: int, base: int) -> list[int] returning the digits of a "
            "non-negative integer in the given base (2..16), most significant first, with no "
            "leading zeros; zero is [0]."
        ),
        checker_source="""
import random
def check(ns):
    digits_fn = ns["base_digits"]
    rng = random.Random(17)
    cases = [(0, 2), (1, 2), (255, 16), (100, 10)]
    cases += [(rng.randrange(0, 10**6), rng.randrange(2, 17)) for _ in range(60)]
    for n, base in cases:
        digits = digits_fn(n, base)
        assert digits and all(0 <= d < base for d in digits), f"digit range wrong for {n} base {base}"
        assert digits == [0] or digits[0] != 0, "leading zero"
        value = 0
        for d in digits:
            value = value * base + d
        assert value == n, f"value mismatch for {n} base {base}"
""",
    ),
    CodeTask(
        task_id="dedupe_stable",
        prompt=(
            "Define dedupe_stable(xs: list) -> list removing duplicates while keeping the first "
            "occurrence of each value in the original order. Values may be unhashable-free "
            "(assume hashable)."
        ),
        checker_source="""
import random
def check(ns):
    dedupe = ns["dedupe_stable"]
    rng = random.Random(19)
    cases = [[], [1], [1, 1, 1], [2, 1, 2, 3, 1]]
    cases += [[rng.randrange(0, 6) for _ in range(rng.randrange(0, 25))] for _ in range(50)]
    for xs in cases:
        out = dedupe(list(xs))
        assert len(set(out)) == len(out), "still has duplicates"
        assert set(out) == set(xs), "value set changed"
        seen = []
        for x in xs:
            if x not in seen:
                seen.append(x)
        assert out == seen, f"order not first-occurrence-stable on {xs}"
""",
    ),
    CodeTask(
        task_id="longest_common_prefix",
        prompt=(
            "Define longest_common_prefix(strs: list[str]) -> str returning the longest string "
            "that is a prefix of every input; empty list returns ''."
        ),
        checker_source="""
import random
def check(ns):
    lcp = ns["longest_common_prefix"]
    rng = random.Random(23)
    cases = [[], ["abc"], ["abc", "abd"], ["", "a"], ["same", "same"]]
    cases += [["pre" + "".join(rng.choice("xy") for _ in range(rng.randrange(0, 5))) for _ in range(rng.randrange(1, 5))] for _ in range(40)]
    for strs in cases:
        out = lcp(list(strs))
        assert all(s.startswith(out) for s in strs), f"not a common prefix on {strs}"
        if strs:
            longer_exists = all(len(s) > len(out) for s in strs) and len({s[len(out)] for s in strs}) == 1
            assert not longer_exists, f"prefix not maximal on {strs}"
        else:
            assert out == ""
""",
    ),
    CodeTask(
        task_id="rotate_right",
        prompt=(
            "Define rotate_right(xs: list, k: int) -> list returning a new list rotated right by k "
            "positions. Negative k rotates left; empty input returns an empty list. Do not mutate xs."
        ),
        checker_source="""
import random
def check(ns):
    rotate = ns["rotate_right"]
    rng = random.Random(29)
    cases = [([], 0), ([1], 17), ([1, 2, 3], 1), ([1, 2, 3], -1), ([1, 2, 3], 7)]
    cases += [([rng.randrange(0, 20) for _ in range(rng.randrange(0, 15))], rng.randrange(-40, 41)) for _ in range(50)]
    for xs, k in cases:
        original = list(xs)
        out = rotate(xs, k)
        expected = [] if not xs else xs[-(k % len(xs)):] + xs[:-(k % len(xs))] if k % len(xs) else list(xs)
        assert out == expected, f"wrong rotation for {xs}, {k}"
        assert xs == original, "mutated input"
""",
    ),
    CodeTask(
        task_id="first_index",
        prompt=(
            "Define first_index(xs: list[int], target: int) -> int for a nondecreasing sorted list. "
            "Return the first index containing target, or -1 if target is absent."
        ),
        checker_source="""
import random
def check(ns):
    first = ns["first_index"]
    rng = random.Random(31)
    cases = [([], 0), ([1], 1), ([1, 1, 1], 1), ([1, 2, 4], 3)]
    for _ in range(60):
        xs = sorted(rng.randrange(0, 12) for _ in range(rng.randrange(0, 35)))
        cases.append((xs, rng.randrange(0, 12)))
    for xs, target in cases:
        expected = xs.index(target) if target in xs else -1
        assert first(list(xs), target) == expected, f"wrong first index for {xs}, {target}"
""",
    ),
    CodeTask(
        task_id="transpose_rect",
        prompt=(
            "Define transpose_rect(matrix: list[list]) -> list[list] for a rectangular matrix. "
            "Return its transpose; both [] and matrices with zero columns transpose to []. Do not mutate matrix."
        ),
        checker_source="""
import random
def check(ns):
    transpose = ns["transpose_rect"]
    rng = random.Random(37)
    cases = [[], [[]], [[], []], [[1, 2, 3]], [[1], [2], [3]]]
    for _ in range(40):
        rows, cols = rng.randrange(0, 7), rng.randrange(0, 7)
        cases.append([[rng.randrange(-9, 10) for _ in range(cols)] for _ in range(rows)])
    for matrix in cases:
        original = [list(row) for row in matrix]
        expected = [] if not matrix or not matrix[0] else [list(column) for column in zip(*matrix)]
        assert transpose(matrix) == expected, f"wrong transpose for {matrix}"
        assert matrix == original, "mutated input"
""",
    ),
    CodeTask(
        task_id="sliding_max",
        prompt=(
            "Define sliding_max(xs: list[int], width: int) -> list[int] returning the maximum of each "
            "contiguous window of width. Return [] when width is nonpositive or larger than len(xs)."
        ),
        checker_source="""
import random
def check(ns):
    sliding = ns["sliding_max"]
    rng = random.Random(41)
    cases = [([], 1), ([1], 1), ([1], 2), ([1, 3, 2, 5, 4], 3), ([2, 2, 2], 2)]
    cases += [([rng.randrange(-20, 21) for _ in range(rng.randrange(0, 30))], rng.randrange(-2, 34)) for _ in range(60)]
    for xs, width in cases:
        expected = [] if width <= 0 or width > len(xs) else [max(xs[i:i + width]) for i in range(len(xs) - width + 1)]
        assert sliding(list(xs), width) == expected, f"wrong sliding max for {xs}, {width}"
""",
    ),
    CodeTask(
        task_id="is_prime",
        prompt="Define is_prime(n: int) -> bool. It returns True exactly for prime integers greater than one.",
        checker_source="""
import random
def check(ns):
    prime = ns["is_prime"]
    def oracle(n):
        if n < 2: return False
        divisor = 2
        while divisor * divisor <= n:
            if n % divisor == 0: return False
            divisor += 1
        return True
    rng = random.Random(43)
    cases = list(range(-5, 100)) + [rng.randrange(0, 10000) for _ in range(80)]
    for n in cases:
        assert prime(n) == oracle(n), f"wrong primality for {n}"
""",
    ),
    CodeTask(
        task_id="stable_partition_even",
        prompt=(
            "Define stable_partition_even(xs: list[int]) -> tuple[list[int], list[int]] returning the even "
            "values then odd values, preserving original order within each list. Do not mutate xs."
        ),
        checker_source="""
import random
def check(ns):
    partition = ns["stable_partition_even"]
    rng = random.Random(47)
    cases = [[], [1], [2], [3, 2, 4, 1, 6]]
    cases += [[rng.randrange(-20, 21) for _ in range(rng.randrange(0, 30))] for _ in range(60)]
    for xs in cases:
        original = list(xs)
        expected = ([x for x in xs if x % 2 == 0], [x for x in xs if x % 2 != 0])
        assert partition(xs) == expected, f"wrong stable partition for {xs}"
        assert xs == original, "mutated input"
""",
    ),
    CodeTask(
        task_id="word_counts_ascii",
        prompt=(
            "Define word_counts_ascii(text: str) -> dict[str, int]. A word is one or more ASCII letters; "
            "count words case-insensitively and ignore all other characters."
        ),
        checker_source="""
import random, re
def check(ns):
    counts = ns["word_counts_ascii"]
    def oracle(text):
        out = {}
        for word in re.findall(r"[A-Za-z]+", text.lower()): out[word] = out.get(word, 0) + 1
        return out
    rng = random.Random(53)
    alphabet = "AbC xyz-123_!?"
    cases = ["", "Hello, hello!", "can't STOP 42 times", "a_A a"]
    cases += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 100))) for _ in range(50)]
    for text in cases:
        assert counts(text) == oracle(text), f"wrong word counts for {text!r}"
""",
    ),
    CodeTask(
        task_id="top_k_frequent",
        prompt=(
            "Define top_k_frequent(xs: list[int], k: int) -> list[int] returning up to k distinct values "
            "ordered by decreasing frequency; ties break by the value's first occurrence in xs. Nonpositive k returns []."
        ),
        checker_source="""
import random
def check(ns):
    top = ns["top_k_frequent"]
    def oracle(xs, k):
        if k <= 0: return []
        counts, first = {}, {}
        for index, value in enumerate(xs):
            counts[value] = counts.get(value, 0) + 1
            first.setdefault(value, index)
        return sorted(counts, key=lambda value: (-counts[value], first[value]))[:k]
    rng = random.Random(59)
    cases = [([], 2), ([1, 2, 2, 1, 3], 2), ([4, 4, 4], 0)]
    cases += [([rng.randrange(0, 8) for _ in range(rng.randrange(0, 35))], rng.randrange(-2, 12)) for _ in range(60)]
    for xs, k in cases:
        assert top(list(xs), k) == oracle(xs, k), f"wrong top-k for {xs}, {k}"
""",
    ),
)

# A second tranche deliberately targets algorithms that require more than a
# familiar one-liner.  Their test generators and oracles remain private.
CODE_TASKS += (
    CodeTask(
        task_id="binary_search_left",
        prompt=(
            "Define binary_search_left(xs: list[int], target: int) -> int for a sorted nondecreasing list. "
            "Return the first position where target can be inserted while preserving order (from 0 through len(xs))."
        ),
        checker_source="import bisect, random\ndef check(ns):\n    search = ns['binary_search_left']\n    rng = random.Random(61)\n    cases = [([], 0), ([1], 0), ([1], 1), ([1, 1, 1], 1), ([1, 3, 5], 4)]\n    cases += [(sorted(rng.randrange(-10, 11) for _ in range(rng.randrange(0, 45))), rng.randrange(-12, 13)) for _ in range(70)]\n    for xs, target in cases:\n        assert search(list(xs), target) == bisect.bisect_left(xs, target), f'wrong insertion position for {xs}, {target}'\n",
    ),
    CodeTask(
        task_id="find_all_anagrams",
        prompt=(
            "Define find_all_anagrams(text: str, pattern: str) -> list[int] returning every starting index "
            "of a substring of text that is an anagram of pattern. Overlapping matches count; an empty pattern returns []."
        ),
        checker_source="import random\ndef check(ns):\n    find = ns['find_all_anagrams']\n    def oracle(text, pattern):\n        if not pattern: return []\n        target = sorted(pattern)\n        return [i for i in range(len(text) - len(pattern) + 1) if sorted(text[i:i + len(pattern)]) == target]\n    rng = random.Random(67)\n    cases = [('cbaebabacd', 'abc'), ('abab', 'ab'), ('aaaa', 'aa'), ('', 'a'), ('abc', '')]\n    cases += [(''.join(rng.choice('abcd') for _ in range(rng.randrange(0, 35))), ''.join(rng.choice('abcd') for _ in range(rng.randrange(0, 8)))) for _ in range(60)]\n    for text, pattern in cases:\n        assert find(text, pattern) == oracle(text, pattern), f'wrong anagram positions for {text!r}, {pattern!r}'\n",
    ),
    CodeTask(
        task_id="shortest_path_length",
        prompt=(
            "Define shortest_path_length(n: int, edges: list[tuple[int, int]], start: int, goal: int) -> int for "
            "an undirected graph with nodes 0..n-1. Return the fewest edges from start to goal, or -1 if unreachable."
        ),
        checker_source="import collections, random\ndef check(ns):\n    shortest = ns['shortest_path_length']\n    def oracle(n, edges, start, goal):\n        adjacency = [[] for _ in range(n)]\n        for a, b in edges: adjacency[a].append(b); adjacency[b].append(a)\n        queue = collections.deque([(start, 0)]); seen = {start}\n        while queue:\n            node, distance = queue.popleft()\n            if node == goal: return distance\n            for nxt in adjacency[node]:\n                if nxt not in seen: seen.add(nxt); queue.append((nxt, distance + 1))\n        return -1\n    rng = random.Random(73)\n    cases = [(1, [], 0, 0), (3, [(0, 1)], 0, 2), (4, [(0, 1), (1, 2), (2, 3)], 0, 3)]\n    for _ in range(55):\n        n = rng.randrange(1, 10)\n        edges = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < .25]\n        cases.append((n, edges, rng.randrange(n), rng.randrange(n)))\n    for n, edges, start, goal in cases:\n        assert shortest(n, list(edges), start, goal) == oracle(n, edges, start, goal), f'wrong path length for {n}, {edges}, {start}, {goal}'\n",
    ),
    CodeTask(
        task_id="min_coin_change",
        prompt=(
            "Define min_coin_change(coins: list[int], target: int) -> int returning the fewest coins needed to sum "
            "to a nonnegative target using each positive denomination any number of times, or -1 when impossible. Target 0 returns 0."
        ),
        checker_source="import random\ndef check(ns):\n    change = ns['min_coin_change']\n    def oracle(coins, target):\n        best = [0] + [target + 1] * target\n        for amount in range(1, target + 1):\n            best[amount] = min((best[amount - coin] + 1 for coin in coins if 0 < coin <= amount), default=target + 1)\n        return -1 if best[target] > target else best[target]\n    rng = random.Random(79)\n    cases = [([], 0), ([], 5), ([1, 2, 5], 11), ([2], 3), ([2, 4], 8)]\n    cases += [([rng.randrange(1, 10) for _ in range(rng.randrange(1, 6))], rng.randrange(0, 60)) for _ in range(60)]\n    for coins, target in cases:\n        assert change(list(coins), target) == oracle(coins, target), f'wrong coin change for {coins}, {target}'\n",
    ),
)

CODE_TASKS += (
    CodeTask(
        task_id="spiral_order",
        prompt=(
            "Define spiral_order(matrix: list[list[int]]) -> list[int] for a rectangular matrix. Return the "
            "elements clockwise from the top-left; [] and matrices with zero columns return []. Do not mutate matrix."
        ),
        checker_source="import random\ndef check(ns):\n    spiral = ns['spiral_order']\n    def oracle(matrix):\n        if not matrix or not matrix[0]: return []\n        top, bottom, left, right, out = 0, len(matrix)-1, 0, len(matrix[0])-1, []\n        while top <= bottom and left <= right:\n            out.extend(matrix[top][left:right+1]); top += 1\n            for row in range(top, bottom+1): out.append(matrix[row][right])\n            right -= 1\n            if top <= bottom: out.extend(reversed(matrix[bottom][left:right+1])); bottom -= 1\n            if left <= right:\n                for row in range(bottom, top-1, -1): out.append(matrix[row][left])\n                left += 1\n        return out\n    rng = random.Random(101)\n    cases = [[], [[]], [[], []], [[1]], [[1,2,3]], [[1],[2],[3]], [[1,2],[4,5],[7,8]]]\n    cases += [[[rng.randrange(-9,10) for _ in range(cols)] for _ in range(rows)] for rows, cols in [(rng.randrange(0,7), rng.randrange(0,7)) for _ in range(45)]]\n    for matrix in cases:\n        original = [list(row) for row in matrix]\n        assert spiral(matrix) == oracle(matrix), f'wrong spiral for {matrix}'\n        assert matrix == original, 'mutated input'\n",
    ),
    CodeTask(
        task_id="normalize_posix_path",
        prompt=(
            "Define normalize_posix_path(path: str) -> str. Split on '/', remove empty and '.' segments, resolve '..' "
            "without moving above root for absolute paths, preserve leading '/' when absolute, and return '.' for an empty relative result."
        ),
        checker_source="import random\ndef check(ns):\n    normalize = ns['normalize_posix_path']\n    def oracle(path):\n        absolute = path.startswith('/'); parts = []\n        for segment in path.split('/'):\n            if not segment or segment == '.': continue\n            if segment == '..':\n                if parts and parts[-1] != '..': parts.pop()\n                elif not absolute: parts.append(segment)\n            else: parts.append(segment)\n        result = '/'.join(parts)\n        return '/' + result if absolute else result or '.'\n    rng = random.Random(103); atoms = ['', '.', '..', 'a', 'b', 'tmp']\n    cases = ['', '.', 'a/./b', 'a/../../b', '/a/../../b', '/', '//a///b/', '/a/../b']\n    cases += [('' if rng.random() < .5 else '/') + '/'.join(rng.choice(atoms) for _ in range(rng.randrange(0,9))) for _ in range(70)]\n    for path in cases: assert normalize(path) == oracle(path), f'wrong normalized path for {path!r}'\n",
    ),
    CodeTask(
        task_id="count_islands",
        prompt=(
            "Define count_islands(grid: list[str]) -> int. Each equal-length row contains only '0' and '1'; count "
            "4-directionally connected groups of '1'. An empty grid has zero islands. Do not mutate grid."
        ),
        checker_source="import random\ndef check(ns):\n    count = ns['count_islands']\n    def oracle(grid):\n        if not grid: return 0\n        cells = {(r,c) for r,row in enumerate(grid) for c,value in enumerate(row) if value == '1'}; groups = 0\n        while cells:\n            groups += 1; stack = [cells.pop()]\n            while stack:\n                r,c = stack.pop()\n                for nxt in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):\n                    if nxt in cells: cells.remove(nxt); stack.append(nxt)\n        return groups\n    rng = random.Random(107)\n    cases = [[], ['0'], ['1'], ['11000','11000','00100','00011']]\n    cases += [[''.join(rng.choice('01') for _ in range(cols)) for _ in range(rows)] for rows, cols in [(rng.randrange(0,8), rng.randrange(1,9)) for _ in range(60)]]\n    for grid in cases:\n        original = list(grid)\n        assert count(grid) == oracle(grid), f'wrong island count for {grid}'\n        assert grid == original, 'mutated input'\n",
    ),
    CodeTask(
        task_id="is_valid_sudoku",
        prompt=(
            "Define is_valid_sudoku(board: list[str]) -> bool for a 9x9 partial Sudoku board using digits '1'..'9' "
            "and '.'. Return True exactly when no row, column, or 3x3 box repeats a digit."
        ),
        checker_source="import random\ndef check(ns):\n    valid = ns['is_valid_sudoku']\n    def oracle(board):\n        groups = list(board) + [''.join(row[c] for row in board) for c in range(9)]\n        groups += [''.join(board[r][c] for r in range(br,br+3) for c in range(bc,bc+3)) for br in range(0,9,3) for bc in range(0,9,3)]\n        return all(len([x for x in group if x != '.']) == len(set(x for x in group if x != '.')) for group in groups)\n    rng = random.Random(109)\n    good = ['53..7....','6..195...','.98....6.','8...6...3','4..8.3..1','7...2...6','.6....28.','...419..5','....8..79']\n    bad_row = ['553..7...','6..195...','.98....6.','8...6...3','4..8.3..1','7...2...6','.6....28.','...419..5','....8..79']\n    cases = [good, bad_row]\n    for _ in range(60):\n        board = [['.' for _ in range(9)] for _ in range(9)]\n        for _ in range(rng.randrange(0,30)): board[rng.randrange(9)][rng.randrange(9)] = str(rng.randrange(1,10))\n        cases.append([''.join(row) for row in board])\n    for board in cases: assert valid(list(board)) == oracle(board), f'wrong sudoku verdict for {board}'\n",
    ),
)

TASKS_BY_ID = {task.task_id: task for task in CODE_TASKS}


# ------------------------------------------------------------------ minting


def training_overlap(buffer_paths: list[str | Path], text: str) -> bool:
    """Row-level leakage: the exact task prompt appears in a training buffer."""
    for path in buffer_paths:
        path = Path(path)
        if path.exists() and text in path.read_text(encoding="utf-8"):
            return True
    return False


def mint_code_items(
    buffer_paths: list[str | Path] | None = None,
    max_items: int = 8,
    seed: int = 0,
) -> list[ExamItem]:
    """Mint a reproducible, seed-selected slice of the private task library.

    A fixed prefix silently makes successive "fresh" banks identical. The
    seed is therefore selection provenance, not cosmetic randomness: a later
    audit can recreate the task slice without exposing checkers to candidates.
    """
    buffer_paths = buffer_paths or []
    tasks = list(CODE_TASKS)
    # Seed zero is the documented canonical starter cohort, retained for
    # compatible CLI/MCP/ACP examples. Every explicit nonzero seed selects a
    # distinct deterministic slice instead of silently reusing that prefix.
    if seed:
        random.Random(seed).shuffle(tasks)
    items: list[ExamItem] = []
    for task in tasks[:max_items]:
        leaked = training_overlap(buffer_paths, task.prompt)
        item = ExamItem(
            item_id=f"code_{uuid.uuid4().hex[:8]}",
            domain="code",
            kind="code_task",
            payload={"task_id": task.task_id, "prompt": task.prompt, "selection_seed": seed},
            oracle="hidden_property_checks",
            source="task_library_v2",
            horizon="property_checks_v1",
            lineage=[task.task_id],
            leakage_risk=1.0 if leaked else 0.0,
        )
        if leaked:
            item.status = "quarantined"
        items.append(item)
    return items


# ------------------------------------------------------------------ grading


def code_item_prompt(item: ExamItem) -> str:
    return (
        f"{item.payload['prompt']}\n\n"
        "Reply with a single Python code block containing only the required definition(s). "
        "No commentary outside the block."
    )


def extract_code(raw: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        return max(fenced, key=len).strip()
    return raw.strip()


_HARNESS = """
import json, sys
namespace = {}
try:
    exec(compile(CANDIDATE_SOURCE, "<candidate>", "exec"), namespace)
    exec(compile(CHECKER_SOURCE, "<checker>", "exec"), namespace)
    namespace["check"](namespace)
except AssertionError as error:
    print(json.dumps({"passed": False, "reason": f"check failed: {error}"}))
except BaseException as error:
    print(json.dumps({"passed": False, "reason": f"{type(error).__name__}: {error}"}))
else:
    print(json.dumps({"passed": True, "reason": "all property checks passed"}))
"""


def run_checker(candidate_source: str, checker_source: str, timeout_seconds: float = 15.0) -> dict:
    """Run candidate + private checker in an isolated interpreter with a hard timeout."""
    program = (
        f"CANDIDATE_SOURCE = {candidate_source!r}\n"
        f"CHECKER_SOURCE = {checker_source!r}\n"
        f"{_HARNESS}"
    )
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "graded_run.py"
        path.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                timeout=timeout_seconds,
                cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": f"timed out after {timeout_seconds}s"}
    for line in reversed(completed.stdout.decode("utf-8", "replace").splitlines()):
        if line.strip().startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {
        "passed": False,
        "reason": f"no verdict emitted (exit {completed.returncode}): "
        f"{completed.stderr.decode('utf-8', 'replace')[:300]}",
    }


def grade_code_answer(item: ExamItem, raw_answer: str | None, timeout_seconds: float = 15.0) -> bool:
    if not raw_answer:
        return False
    task = TASKS_BY_ID.get(item.payload.get("task_id", ""))
    if task is None:
        return False
    source = extract_code(str(raw_answer))
    if not source:
        return False
    return bool(run_checker(source, task.checker_source, timeout_seconds).get("passed"))
