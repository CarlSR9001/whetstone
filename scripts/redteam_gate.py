"""Run Whetstone's local semantic-leakage and inflation attacks."""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "src")

from bcv.redteam import run_redteam


def main() -> None:
    parser = argparse.ArgumentParser(description="Red-team the Whetstone quarantine and promotion gate.")
    parser.add_argument("--root", default=".bcv_runs/redteam")
    args = parser.parse_args()
    print(json.dumps(run_redteam(args.root), indent=2))


if __name__ == "__main__":
    main()
