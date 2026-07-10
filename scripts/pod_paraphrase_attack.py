"""Adversarial paraphrase generation against the leakage-fingerprint tier.

Runs on the pod AFTER the ladder (the 32B is already cached). For each public
DSL expression, the 32B is asked for rewrites in two buckets: exactly
equivalent, and subtly different. The attack corpus comes home for local
evaluation: pairs that COLLIDE on the fingerprint corpus but DIVERGE on larger
graphs measure the fingerprint's corpus-size blind spot under a real
adversary, not mechanical rewrites.

Only public inputs are used (the DSL grammar and expression list are in the
public repo); no exam bank content is involved.

Run: cd /workspace/whetstone && python scripts/pod_paraphrase_attack.py > attack.log 2>&1
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "src")

from bcv.graph_repair_data import _candidate_expressions
from bcv.transformers_client import TransformersLocalClient, extract_json

OUT = "paraphrase_attack_corpus.jsonl"
MODEL = "Qwen/Qwen2.5-32B-Instruct"

PROMPT = """You are attacking a plagiarism-detection system for boolean predicates over graphs.
Grammar: python-like boolean expressions over these features: n, m, density, max_degree,
min_degree, is_connected, is_complete, is_forest, is_tree, is_bipartite, is_triangle_free,
max_degree_le_2, has_universal_vertex, has_isolated_vertex, is_regular, num_components,
clique_number, girth. Operators: and, or, not, comparisons (<=, >=, <, >, ==), parentheses.

Target predicate: `{expression}`

Produce EXACTLY this JSON and nothing else:
{{"equivalent": ["<3 rewrites that are LOGICALLY IDENTICAL on every graph, using different syntax>"],
 "near_miss": ["<3 rewrites that agree on almost all small graphs but differ on SOME graph>"]}}"""


def main() -> None:
    expressions = list(_candidate_expressions())[:60]
    client = TransformersLocalClient(model_name=MODEL, max_new_tokens=512)
    written = 0
    with open(OUT, "w", encoding="utf-8") as handle:
        for index, expression in enumerate(expressions):
            started = time.time()
            try:
                raw = client.generate_text(PROMPT.format(expression=expression), temperature=0.0)
                parsed = extract_json(raw)
            except Exception as error:
                print(json.dumps({"expression": expression, "error": str(error)[:200]}), flush=True)
                continue
            if not isinstance(parsed, dict):
                continue
            for bucket in ("equivalent", "near_miss"):
                for rewrite in (parsed.get(bucket) or [])[:3]:
                    if isinstance(rewrite, str) and rewrite.strip():
                        handle.write(json.dumps({
                            "original": expression,
                            "rewrite": rewrite.strip(),
                            "model_claim": bucket,
                        }) + "\n")
                        written += 1
            handle.flush()
            print(f"[{index + 1}/{len(expressions)}] {expression[:40]} "
                  f"({round(time.time() - started, 1)}s, total rows {written})", flush=True)
    print(f"ATTACK DONE rows={written}", flush=True)


if __name__ == "__main__":
    main()
