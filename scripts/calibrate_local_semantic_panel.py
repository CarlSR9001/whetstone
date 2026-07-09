"""Calibrate the local semantic support panel on a labeled JSONL corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from bcv.panel import calibrate_panel
from bcv.panel_semantic import LocalSemanticJudge, semantic_support_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate an in-boundary local semantic support panel.")
    parser.add_argument("--labeled", default="sample_docs/support_hard_calibration.jsonl")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--out", default=".bcv_runs/local_semantic_panel/calibration.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.labeled).read_text(encoding="utf-8").splitlines() if line.strip()]
    panel = semantic_support_panel(LocalSemanticJudge(args.api_base, args.model, args.max_tokens))
    calibration = calibrate_panel(panel, [(row["case"], row["answer"], bool(row["human_pass"])) for row in rows])
    report = {
        "label_sources": sorted({row.get("label_source", "unspecified") for row in rows}),
        "model": args.model,
        "api_base": args.api_base,
        "calibration": calibration.to_dict(),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
