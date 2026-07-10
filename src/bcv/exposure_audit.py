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
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bcv.examiner import ExaminerBank

COMMAND = re.compile(
    r"Controller: (play (?:b|w) [A-Z]\d+|genmove (?:b|w)|undo|clear_board|boardsize \d+)"
)

# Transcripts known to be public forever (committed 2026-07-02; removing them
# from HEAD does not unpublish git history). Mint-time gates check against
# these so a collision is quarantined at birth, not found by a later audit.
KNOWN_PUBLISHED_LOGS = (
    "gtp_logs/20260702-135220-0A3CEC6E.log",
    "gtp_logs/20260702-135220-F796F8D1.log",
)

# SHA-256 commitments to the 22 ordered move prefixes reconstructed from the
# public logs. The raw transcripts stay out of HEAD, but the standing denylist
# must survive a fresh clone. Hashing canonical move arrays lets minting test a
# candidate without republishing the leaked trajectories.
KNOWN_PUBLISHED_PREFIX_HASHES = frozenset({
    "081ed460d04d48484e0bd30d49bf33639ba4b42d34b6ba4ff5a5bd6b769bc596",
    "169427cbcdab600b418a531c5b7b64aa01fd56feafc9e2247865232ccc123147",
    "205c8a0658430dfa7ba46874521edd0c414e8317b7278c2e591ade1eda384e43",
    "2976bc1ac697f256fa674f0a34bfa12caf383b38344960a613a3ca0101bcd954",
    "33edd2b33cdecc1dedc5c0342fc97b21841aab06ae20144c33651643e4499257",
    "486e18b44d03354d0c381dfa7ada2965404daeebf1b0509a7a36229ac6c31c18",
    "53a7117a5c9947958141180cc84a81ed2380e42e10cf459918e52767642cb33d",
    "5b61327e512d8709b0c53b9a657aec91184bbfac40f41f5e81d29913acd18d47",
    "5cb35ec2a15fd8472a167fc7d6cbe0135e2725a4f84daef9d6a17e21131e6a70",
    "6c921f7cc5ee17271de7147b1c271fc0203b9e93895bb195061c954758c017de",
    "79778b8363d03e43d56de93e51c3e90660fb95382dba2c2e7f860a7fe8c89c55",
    "9b8f749509397372c334e72627b70fa935d69b0ffe7b3f60a5418b598dcd3d2e",
    "9f31cc6caf66a77e5d38569e2f7f1b56cc2789d8ad26a931a4ed7298c4995e0a",
    "b777f5f1b5c592ea2278ea8130a2683c30074bee1f84a9eeae12c6af1b3946bf",
    "c31b4d7a173ce9368d26169c98e5e5c61b1af4b699937ffe92a5f1cf18206a7a",
    "d5c68419a3641cb61f4dd1a5a665b75f5efcb5dc65d7547c5d173657d5a8c1c8",
    "e770e8adc01c749e8b17df3f14ea85149612c7d0327ad975c3fc2869b7ec9d71",
    "ed6b4728ba2170b245f5450bb1a00e0f42ca09bafd336a535e85a84b12abbb5b",
    "f5585692eac6228b3f869b50f91f4b1d8a570c2b5239a4569ce8aa7a38de2741",
    "f8516613edd4674640fc1e5bcacccb5b3a770f40aa350567a0f7aa131ff52ca6",
    "fac2c957c287cb93c3141e09d3c2afb54b4ed447edcfdef567ef5ba1f354b271",
    "fe6b92ffb53c723d0eecc0aa7946003872fa2ddc1818e1f2fe051d080821a1c2",
})


def prefix_sha256(moves: tuple[str, ...] | list[str]) -> str:
    payload = json.dumps(list(moves), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_known_published_prefix(moves: tuple[str, ...] | list[str]) -> bool:
    return prefix_sha256(moves) in KNOWN_PUBLISHED_PREFIX_HASHES


def known_published_prefixes() -> set[tuple[str, ...]]:
    present = [path for path in KNOWN_PUBLISHED_LOGS if Path(path).exists()]
    return published_prefixes(present)


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
