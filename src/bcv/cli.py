"""whetstone: one command line for the promotion gate.

    whetstone init                      scaffold a bank root + whetstone.toml
    whetstone mint --domain code        mint exam items through the registry
    whetstone grade --system gen3 ...   grade any candidate (API, command, stored)
    whetstone gate --baseline a --candidate b   decide, report, exit-code the verdict
    whetstone status                    bank health: buckets, discrimination, metabolism
    whetstone burn --item ID ...        record an external exposure by hand
    whetstone calibrate-panel           measure panel agreement on labeled cases
    whetstone serve --port 8977         the examiner as a local JSON service

Exit codes are the CI contract: PASS=0, HOLD=2, BLOCK=3, usage/config errors=1.
A pipeline step `whetstone gate ...` therefore fails the build unless the
candidate cleared the gate — which is the product in one shell line.

Config: ``whetstone.toml`` in the working directory (or --config). Flags
override config; config overrides defaults. Secrets never go on argv — API
keys are named by environment variable (--api-key-env).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tomllib
from pathlib import Path

from bcv.examiner import ExaminerBank

EXIT_BY_VERDICT = {"PASS": 0, "HOLD": 2, "BLOCK": 3}
DEFAULT_CONFIG = "whetstone.toml"

CONFIG_TEMPLATE = """# whetstone configuration
[bank]
root = ".bcv_runs/examiner"

[policy]
min_gains = 1
max_regressions = 0
confidence_alpha = 0.05
require_retained_probe = false

[grading]
stress_ns = [7, 8]
seed = 0
"""


def load_config(path: str | None) -> dict:
    candidate = Path(path or DEFAULT_CONFIG)
    if not candidate.exists():
        if path:  # explicitly requested config must exist
            raise SystemExit(f"config not found: {candidate}")
        return {}
    return tomllib.loads(candidate.read_text(encoding="utf-8"))


def open_bank(args, config: dict) -> ExaminerBank:
    root = args.root or config.get("bank", {}).get("root") or ".bcv_runs/examiner"
    return ExaminerBank(root)


def build_candidate(args):
    from bcv.candidates import CommandCandidate, OpenAICompatibleCandidate, StoredAnswerCandidate

    chosen = [bool(args.answers), bool(args.api_base), bool(args.command), bool(args.acp)]
    if sum(chosen) != 1:
        raise SystemExit("choose exactly one of --answers, --api-base, --command, --acp")
    if args.answers:
        return StoredAnswerCandidate(args.answers)
    if args.api_base:
        if not args.model:
            raise SystemExit("--api-base requires --model")
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        return OpenAICompatibleCandidate(
            base_url=args.api_base, model=args.model, api_key=api_key,
            timeout_seconds=args.timeout,
        )
    if args.acp:
        from bcv.acp import ACPCandidate

        return ACPCandidate(shlex.split(args.acp), timeout_seconds=args.timeout)
    return CommandCandidate(shlex.split(args.command), timeout_seconds=args.timeout)


# ------------------------------------------------------------------ commands


def cmd_init(args, config: dict) -> int:
    root = Path(args.root or ".bcv_runs/examiner")
    root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config or DEFAULT_CONFIG)
    if config_path.exists() and not args.force:
        print(f"{config_path} already exists (use --force to overwrite)")
    else:
        config_path.write_text(CONFIG_TEMPLATE.replace(".bcv_runs/examiner", str(root).replace("\\", "/")), encoding="utf-8")
        print(f"wrote {config_path}")
    ExaminerBank(root).save()
    print(f"bank initialized at {root}")
    return 0


def cmd_mint(args, config: dict) -> int:
    from bcv.registry import MINTABLE_DOMAINS, mint_domain

    bank = open_bank(args, config)
    domains = args.domain or list(MINTABLE_DOMAINS)
    buffers = args.buffers or []
    reports = [
        mint_domain(bank, domain, buffers=buffers, max_items=args.max_items, seed=args.seed)
        for domain in domains
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


def cmd_grade(args, config: dict) -> int:
    from bcv.registry import grade_bank

    bank = open_bank(args, config)
    candidate = build_candidate(args)
    if candidate.is_external and args.allow_external_no_burn:
        print("WARNING: external endpoint graded WITHOUT burn accounting (explicitly overridden)")
    grading = config.get("grading", {})
    report = grade_bank(
        bank,
        system=args.system,
        candidate=candidate,
        max_items=args.max_items,
        stress_ns=tuple(grading.get("stress_ns", (7, 8))),
        seed=int(grading.get("seed", 0)),
        burn_external=not args.allow_external_no_burn,
    )
    if hasattr(candidate, "close"):
        candidate.close()
    summary = {key: value for key, value in report.items() if key != "results"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_gate(args, config: dict) -> int:
    from bcv.gate import GatePolicy, build_gate_report, latest_grade_event_results, write_gate_report

    bank = open_bank(args, config)
    policy_config = dict(config.get("policy", {}))
    policy = GatePolicy(
        min_gains=args.min_gains if args.min_gains is not None else int(policy_config.get("min_gains", 1)),
        max_regressions=args.max_regressions if args.max_regressions is not None else int(policy_config.get("max_regressions", 0)),
        confidence_alpha=args.alpha if args.alpha is not None else float(policy_config.get("confidence_alpha", 0.05)),
        require_retained_probe=bool(policy_config.get("require_retained_probe", False)) if args.retained_probe is None else True,
    )
    events_path = bank.root / "grade_events.jsonl"
    baseline_results = latest_grade_event_results(events_path, args.baseline)
    candidate_results = latest_grade_event_results(events_path, args.candidate)
    shared = set(baseline_results) & set(candidate_results)
    if not shared:
        raise SystemExit("baseline and candidate share no graded items")
    retained = json.loads(Path(args.retained_probe).read_text(encoding="utf-8")) if args.retained_probe else None
    report = build_gate_report(
        bank,
        baseline=args.baseline,
        candidate=args.candidate,
        baseline_results={item: baseline_results[item] for item in sorted(shared)},
        candidate_results={item: candidate_results[item] for item in sorted(shared)},
        retained_probe=retained,
        policy=policy,
    )
    out_dir = args.out or str(bank.root / f"gate_{args.candidate}")
    json_path, html_path = write_gate_report(report, out_dir)
    print(json.dumps({
        "verdict": report["verdict"],
        "reasons": report["reasons"],
        "gains": report["paired_evidence"]["gains"],
        "regressions": report["paired_evidence"]["regressions"],
        "p_value": report["paired_evidence"]["exact_mcnemar_two_sided_p"],
        "report_json": str(json_path),
        "report_html": str(html_path),
    }, indent=2, sort_keys=True))
    return EXIT_BY_VERDICT[report["verdict"]]


def cmd_status(args, config: dict) -> int:
    bank = open_bank(args, config)
    statuses: dict[str, int] = {}
    domains: dict[str, int] = {}
    for item in bank.items.values():
        statuses[item.status] = statuses.get(item.status, 0) + 1
        if item.status == "promoted":
            domains[item.domain] = domains.get(item.domain, 0) + 1
    discriminating = sum(1 for item in bank.promoted_items() if item.discrimination() > 0)
    graded_systems = sorted({system for item in bank.items.values() for system in item.graded})
    print(json.dumps({
        "root": str(bank.root),
        "buckets": dict(sorted(statuses.items())),
        "promoted_by_domain": dict(sorted(domains.items())),
        "discriminating_items": discriminating,
        "graded_systems": graded_systems,
        "metabolism": {
            "reusable": statuses.get("promoted", 0),
            "burned": statuses.get("burned", 0),
            "retired_to_regression": statuses.get("retired", 0),
            "quarantined": statuses.get("quarantined", 0),
        },
    }, indent=2, sort_keys=True))
    return 0


def cmd_burn(args, config: dict) -> int:
    bank = open_bank(args, config)
    bank.burn(args.item, provider=args.provider, reason=args.reason)
    bank.save()
    print(f"burned {args.item} ({args.provider}: {args.reason})")
    return 0


def cmd_calibrate_panel(args, config: dict) -> int:
    from bcv.panel import SUPPORT_PANEL, calibrate_panel, save_calibration

    labeled_path = Path(args.labeled)
    triples = []
    for line in labeled_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            triples.append((row["case"], row["answer"], bool(row["human_pass"])))
    calibration = calibrate_panel(SUPPORT_PANEL, triples)
    out = args.out or "results/support_panel_calibration.json"
    save_calibration(calibration, out)
    print(json.dumps(calibration.to_dict(), indent=2, sort_keys=True))
    print(f"saved to {out}")
    return 0


def cmd_serve(args, config: dict) -> int:
    from bcv.service import serve

    serve(
        root=args.root or config.get("bank", {}).get("root") or ".bcv_runs/examiner",
        port=args.port,
        token=args.token or os.environ.get("WHETSTONE_TOKEN"),
        config=config,
    )
    return 0


def cmd_sweep(args, config: dict) -> int:
    """Saturation sweep: retire items every graded system passes (after two
    consecutive saturated rounds) and report the downward flow."""
    bank = open_bank(args, config)
    retired = bank.sweep_saturation(min_systems=args.min_systems)
    bank.save()
    print(json.dumps({
        "retired_this_sweep": retired,
        "trainable_rows_available": len(bank.trainable_rows()),
        "promoted_remaining": len(bank.promoted_items()),
        "note": "items retire on the second consecutive saturated sweep; run after each grading round",
    }, indent=2, sort_keys=True))
    return 0


def cmd_demo(args, config: dict) -> int:
    from bcv.demo_investor import DemoConfig, run_demo

    run_demo(DemoConfig(seed=args.seed))
    return 0


def cmd_redteam(args, config: dict) -> int:
    """Hostile self-test of the quarantine and gate; nonzero exit on any escape."""
    from bcv.redteam import run_redteam

    report = run_redteam(root=args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    paraphrase_escaped = report["paraphrase_attack"]["promotion_allowed"]
    inflation_escaped = not report["inflation_attack"]["caught"]
    return 1 if (paraphrase_escaped or inflation_escaped) else 0


def cmd_mcp(args, config: dict) -> int:
    import bcv.whetstone_mcp as whetstone_mcp

    root = args.root or config.get("bank", {}).get("root")
    if root:
        whetstone_mcp.use_bank_impl(root)
    whetstone_mcp.mcp.run()
    return 0


# -------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whetstone", description="The private promotion gate for AI agents.")
    parser.add_argument("--config", help=f"config file (default {DEFAULT_CONFIG} when present)")
    parser.add_argument("--root", help="bank root (overrides config)")
    # dest must not be "command": the grade subcommand owns a --command flag,
    # and argparse would silently overwrite one with the other.
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_init = sub.add_parser("init", help="scaffold a bank and config")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(fn=cmd_init)

    p_mint = sub.add_parser("mint", help="mint exam items through the registry")
    p_mint.add_argument("--domain", action="append", help="repeatable; default: all mintable domains")
    p_mint.add_argument("--max-items", type=int, default=8)
    p_mint.add_argument("--seed", type=int, default=0)
    p_mint.add_argument("--buffers", nargs="*", help="training buffers for the leakage check")
    p_mint.set_defaults(fn=cmd_mint)

    p_grade = sub.add_parser("grade", help="grade a candidate on the promoted bank")
    p_grade.add_argument("--system", required=True, help="name recorded in the grade ledger")
    p_grade.add_argument("--answers", help="stored answers JSONL: {item_id, answer}")
    p_grade.add_argument("--api-base", help="OpenAI-compatible API root, e.g. http://localhost:1234/v1")
    p_grade.add_argument("--model", help="model name for --api-base")
    p_grade.add_argument("--api-key-env", help="environment variable holding the API key")
    p_grade.add_argument("--command", help="shell command: prompt on stdin, answer on stdout")
    p_grade.add_argument("--acp", help="Agent Client Protocol agent command; whetstone drives it as the ACP client")
    p_grade.add_argument("--max-items", type=int)
    p_grade.add_argument("--timeout", type=float, default=120.0)
    p_grade.add_argument(
        "--allow-external-no-burn", action="store_true",
        help="grade an external endpoint WITHOUT burning exposed items (on-the-record override)",
    )
    p_grade.set_defaults(fn=cmd_grade)

    p_gate = sub.add_parser("gate", help="promotion decision from recorded grades; exit code carries the verdict")
    p_gate.add_argument("--baseline", required=True)
    p_gate.add_argument("--candidate", required=True)
    p_gate.add_argument("--retained-probe", help="JSON file with base_verified/adapter_verified/eval_examples")
    p_gate.add_argument("--out", help="report directory")
    p_gate.add_argument("--alpha", type=float)
    p_gate.add_argument("--min-gains", type=int)
    p_gate.add_argument("--max-regressions", type=int)
    p_gate.set_defaults(fn=cmd_gate)

    p_status = sub.add_parser("status", help="bank health and metabolism")
    p_status.set_defaults(fn=cmd_status)

    p_burn = sub.add_parser("burn", help="record an external exposure by hand")
    p_burn.add_argument("--item", required=True)
    p_burn.add_argument("--provider", required=True)
    p_burn.add_argument("--reason", required=True)
    p_burn.set_defaults(fn=cmd_burn)

    p_cal = sub.add_parser("calibrate-panel", help="measure support-panel agreement on labeled cases")
    p_cal.add_argument("--labeled", default="sample_docs/support_calibration.jsonl")
    p_cal.add_argument("--out")
    p_cal.set_defaults(fn=cmd_calibrate_panel)

    p_serve = sub.add_parser("serve", help="run the examiner as a local JSON service")
    p_serve.add_argument("--port", type=int, default=8977)
    p_serve.add_argument("--token", help="shared secret; also WHETSTONE_TOKEN env var")
    p_serve.set_defaults(fn=cmd_serve)

    p_mcp = sub.add_parser("mcp", help="run the examiner as an MCP server (stdio)")
    p_mcp.set_defaults(fn=cmd_mcp)

    p_sweep = sub.add_parser("sweep", help="retire saturated items into the downward flow")
    p_sweep.add_argument("--min-systems", type=int, default=2)
    p_sweep.set_defaults(fn=cmd_sweep)

    p_demo = sub.add_parser("demo", help="the sixty-second investor walkthrough (toy bank, real machinery)")
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.set_defaults(fn=cmd_demo)

    p_redteam = sub.add_parser("redteam", help="hostile self-test of quarantine + gate; nonzero exit on escape")
    p_redteam.add_argument("--out", default=".bcv_runs/redteam")
    p_redteam.set_defaults(fn=cmd_redteam)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # `init --config path` is the bootstrap path for a fresh project/CI
    # workspace. Loading an explicitly named file before init would make it
    # impossible for init to create that file.
    config = {} if args.subcommand == "init" else load_config(args.config)
    return args.fn(args, config)


if __name__ == "__main__":
    sys.exit(main())
