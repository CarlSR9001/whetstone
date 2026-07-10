"""Exposure audit: cross-check exam banks against PUBLISHED engine logs.

The incident this tool exists for: KataGo writes gtp_logs/ transcripts of every
session, including the full `play` streams of milling games, and two such logs
from the 2026-07-02 mill were committed to the public repo. Worse than the
direct leak: low-visit engine mills are nearly deterministic, so a LATER mill
regenerates the same game trajectories — items minted on 2026-07-09 turned out
to be prefix-derivable from logs published a week earlier. Publication is not
an event that ends; it is a standing oracle an adversary can replay.

The audit reconstructs every board-position prefix present in a set of GTP
logs (play/undo/clear_board semantics) and burns any bank item whose move
history matches a published prefix. Receipts report counts and domains only —
auditing a leak must not widen it.

Lesson, encoded here as policy: mills that feed private banks must randomize
their openings, and engine transcripts are exam content, never repo content.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bcv.examiner import ExaminerBank

COMMAND = re.compile(
    r"Controller: (play (?:b|w) [A-Z]\d+|genmove (?:b|w)|undo|clear_board|boardsize \d+)"
)


def published_prefixes(log_paths: list[str | Path]) -> set[tuple[str, ...]]:
    """Every position (as an ordered vertex tuple) reconstructible from the logs.

    genmove pushes a move the log does not name; it is tracked as an unknown
    placeholder so the following undo pops IT, not a real play. Prefixes are
    only recorded while the state contains no unknowns."""
    prefixes: set[tuple[str, ...]] = set()
    unknown = object()
    for path in log_paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        state: list = []
        for match in COMMAND.finditer(text):
            command = match.group(1)
            if command == "undo":
                if state:
                    state.pop()
            elif command == "clear_board" or command.startswith("boardsize"):
                state = []
            elif command.startswith("genmove"):
                state.append(unknown)
            else:  # play <color> <vertex>
                state.append(command.split()[-1])
                if unknown not in state:
                    prefixes.add(tuple(state))
    return prefixes


def audit_bank(
    root: str | Path,
    prefixes: set[tuple[str, ...]],
    burn: bool,
    provider: str,
    reason: str,
) -> dict:
    """Find (and optionally burn) items whose move history is published."""
    bank = ExaminerBank(root)
    matches: list[str] = []
    for item in bank.items.values():
        if item.status not in ("candidate", "promoted"):
            continue
        moves = item.payload.get("moves")
        if moves and tuple(moves) in prefixes:
            matches.append(item.item_id)
    burned = 0
    if burn:
        for item_id in matches:
            bank.burn(item_id, provider=provider, reason=reason)
            burned += 1
        if burned:
            bank.save()
    reusable_go = sum(
        1 for item in bank.items.values() if item.status == "promoted" and item.payload.get("moves")
    )
    return {
        "bank": str(root),
        "exposed_items_found": len(matches),
        "burned": burned,
        "mode": "burn" if burn else "dry_run",
        "go_items_still_reusable": reusable_go,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit banks against published GTP logs; burn matches.")
    parser.add_argument("--logs", nargs="+", required=True, help="published engine log files")
    parser.add_argument("--banks", nargs="+", required=True, help="bank roots to audit")
    parser.add_argument("--dry-run-banks", nargs="*", default=[], help="banks to audit WITHOUT burning")
    parser.add_argument("--provider", default="github.com/CarlSR9001/whetstone (public repo)")
    parser.add_argument("--reason", default="position history derivable from committed gtp_logs")
    parser.add_argument("--receipt", default="results/gtp_log_exposure_receipt.json")
    args = parser.parse_args()

    prefixes = published_prefixes(args.logs)
    audits = [audit_bank(root, prefixes, True, args.provider, args.reason) for root in args.banks]
    audits += [audit_bank(root, prefixes, False, args.provider, args.reason) for root in args.dry_run_banks]

    receipt = {
        "evidence_scope": "exposure audit of exam banks vs published engine logs, "
        + datetime.now(timezone.utc).date().isoformat(),
        "incident": {
            "what": "two KataGo GTP transcripts from the 2026-07-02 mill were committed to the "
            "public repo; they contain full play streams of milling games",
            "amplifier": "low-visit mills are near-deterministic, so the 2026-07-09 mill "
            "regenerated trajectories already public — items minted AFTER the leak were still "
            "derivable from it",
            "published_position_prefixes": len(prefixes),
        },
        "remediation": {
            "audits": audits,
            "repo": "logs removed from the branch and gitignored; they remain in git history, "
            "so every position in them is treated as permanently public",
            "policy": "engine transcripts are exam content: logDir outputs stay untracked; "
            "future mills randomize openings so trajectories are not replayable",
        },
        "note": "Counts only; auditing a leak must not widen it.",
    }
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
