"""Mint a reviewable, leakage-audited private-bank extension in a fresh root.

This never mutates the incumbent bank. Grade the extension alongside the
incumbent first, then merge only items that add genuine paired discrimination.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from bcv.domains import DOMAINS
from bcv.examiner import ExaminerBank, mint_repair_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint a focused Whetstone private-bank extension.")
    parser.add_argument("--domain", choices=tuple(DOMAINS), default="mis")
    parser.add_argument("--root", default=".bcv_runs/bank_extension")
    parser.add_argument("--buffers", nargs="+", required=True, help="every training buffer the candidate could have seen")
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--stress-ns", type=int, nargs="+", default=(7, 8))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    bank = ExaminerBank(root)
    minted = mint_repair_items(
        DOMAINS[args.domain], args.buffers, args.max_items, args.max_n, tuple(args.stress_ns), args.seed
    )
    for item in minted:
        bank.add(item)
        if item.status == "candidate":
            bank.promote(item.item_id)
    bank.save()
    report = {
        "domain": args.domain,
        "minted": len(minted),
        "promoted": len(bank.promoted_items()),
        "quarantined": sum(item.status == "quarantined" for item in minted),
        "quarantine_reasons": {
            "row_identity": sum(item.leakage_match == "row_identity" for item in minted),
            "behavioral_fingerprint": sum(item.leakage_match == "behavioral_fingerprint" for item in minted),
        },
        "review_rule": "grade beside the incumbent; merge only items that add paired discrimination",
    }
    (root / "extension_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
