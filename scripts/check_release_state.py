"""Validate deployment state without assertion tracebacks during polling."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable


def _read_payload() -> dict:
    try:
        payload = json.load(sys.stdin)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    return payload


def _value(payload: dict, path: str) -> object:
    value: object = payload
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"{path}: missing")
        value = value[component]
    return value


def _expect(payload: dict, path: str, expected: object) -> str | None:
    try:
        actual = _value(payload, path)
    except ValueError as error:
        return str(error)
    if actual != expected:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def _allow(payload: dict, path: str, allowed: set[object]) -> str | None:
    try:
        actual = _value(payload, path)
    except ValueError as error:
        return str(error)
    if actual not in allowed:
        choices = ", ".join(repr(item) for item in sorted(allowed, key=repr))
        return f"{path}: expected one of {choices}, got {actual!r}"
    return None


def validate_sessions(payload: dict, _args: argparse.Namespace) -> list[str]:
    return [error for error in [_expect(payload, "report_card.active_sessions", 0)] if error]


def validate_forge(payload: dict, args: argparse.Namespace) -> list[str]:
    checks = [
        _expect(payload, "status", "running"),
        _expect(payload, "version", args.version),
        _expect(payload, "build_commit", args.commit),
        _allow(payload, "library_sync", {"unchanged", "published", "source_absent"}),
        _expect(payload, "pid", args.pid),
    ]
    return [error for error in checks if error]


def validate_health(payload: dict, args: argparse.Namespace) -> list[str]:
    checks = [
        _expect(payload, "status", "ok"),
        _expect(payload, "version", args.version),
        _expect(payload, "build_commit", args.commit),
        _expect(payload, "tools", 8),
        _expect(payload, "stateless", True),
        _expect(payload, "private_bank_loaded", False),
        _expect(payload, "report_card.error", None),
        _expect(payload, "report_card.ready", True),
        _expect(payload, "open_bench.ready", True),
        _expect(payload, "open_bench.publication_ledger_configured", True),
        _expect(payload, "open_bench.publication_ledger_parent_writable", True),
        _expect(payload, "open_bench.raw_tasks_persisted", False),
        _expect(payload, "open_bench.raw_answers_persisted", False),
        _expect(payload, "receipt_signing.configured", True),
        _expect(payload, "receipt_signing.ready", True),
    ]
    return [error for error in checks if error]


VALIDATORS: dict[str, Callable[[dict, argparse.Namespace], list[str]]] = {
    "sessions": validate_sessions,
    "forge": validate_forge,
    "health": validate_health,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=VALIDATORS)
    parser.add_argument("--commit")
    parser.add_argument("--version")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    required = {
        "forge": ("commit", "version", "pid"),
        "health": ("commit", "version"),
    }.get(args.mode, ())
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        build_parser().error(f"{args.mode} requires " + ", ".join(f"--{name}" for name in missing))
    try:
        errors = VALIDATORS[args.mode](_read_payload(), args)
    except ValueError as error:
        errors = [str(error)]
    if errors and not args.quiet:
        print("state mismatch: " + "; ".join(errors), file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
