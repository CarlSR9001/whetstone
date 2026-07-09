"""Measure finite-horizon leakage-fingerprint collisions on the local oracle."""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "src")

from bcv.leakage_calibration import run_fingerprint_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate behavioral leakage fingerprints against exhaustive graphs.")
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--pairs-per-kind", type=int, default=48)
    parser.add_argument("--root", default=".bcv_runs/fingerprint_calibration")
    args = parser.parse_args()
    print(json.dumps(run_fingerprint_calibration(args.max_n, args.min_n, args.pairs_per_kind, args.root), indent=2))


if __name__ == "__main__":
    main()
