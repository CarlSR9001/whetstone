"""Run the private trajectory-disjoint Gen-4 engine-student experiment.

Raw FENs, Go histories, bank items, model outputs, and checkpoints stay under
the ignored run root. Only a sanitized aggregate evaluation receipt is written
under results/, and it is called a promotion receipt only after a strict PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

from bcv.examiner import ExaminerBank
from bcv.gate import GatePolicy, bank_hash, build_gate_report, file_sha256, write_gate_report
from bcv.gen4 import (
    canonical_sha256,
    load_jsonl,
    prepare_engine_data,
    trajectory_id,
    write_engine_split,
)


DEFAULT_ROOT = Path(".bcv_runs/gen4_engine_student")
BASE_SYSTEM = "fastcontext_4b_engine_base_gen4"
CANDIDATE_SYSTEM = "fastcontext_4b_engine_adapter_gen4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _resume_mill(path: Path, target: int, batch_size: int, runner, label: str) -> list[dict]:
    rows = load_jsonl(path) if path.exists() else []
    if len(rows) > target:
        raise SystemExit(f"{path} already has {len(rows)} rows, above target {target}")
    known_ids = {trajectory_id(row) for row in rows}
    batch_number = 0
    while len(rows) < target:
        count = min(batch_size, target - len(rows))
        started = time.perf_counter()
        new_rows = runner(count, batch_number)
        new_ids = {trajectory_id(row) for row in new_rows}
        if known_ids & new_ids:
            raise SystemExit(f"{label} mill reused a trajectory id")
        _append_jsonl(path, new_rows)
        rows.extend(new_rows)
        known_ids.update(new_ids)
        batch_number += 1
        print(
            json.dumps(
                {
                    "phase": "mill",
                    "domain": label,
                    "rows": len(rows),
                    "target": target,
                    "batch_seconds": round(time.perf_counter() - started, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return rows


def mill(args) -> dict:
    from bcv.baduk import mill_go_positions
    from bcv.grandmaster import mill_positions

    raw = args.root / "raw"
    chess_path = raw / "chess.jsonl"
    go_path = raw / "go.jsonl"
    chess_rows = _resume_mill(
        chess_path,
        args.chess_target,
        args.chess_batch,
        lambda count, batch: mill_positions(
            count,
            seed=args.seed + batch,
            engine_path=args.stockfish,
        ),
        "chess",
    )
    go_rows = _resume_mill(
        go_path,
        args.go_target,
        args.go_batch,
        lambda count, batch: mill_go_positions(
            count,
            seed=args.seed + 10_000 + batch,
            katago_dir=args.katago_dir,
            opening_plies=(4, 8),
        ),
        "go",
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "new_rows": {"chess": len(chess_rows), "go": len(go_rows)},
        "trajectories": {
            "chess": len({trajectory_id(row) for row in chess_rows}),
            "go": len({trajectory_id(row) for row in go_rows}),
            "randomized_non_replayable": True,
        },
        "files": {
            "chess_sha256": _sha256(chess_path),
            "go_sha256": _sha256(go_path),
        },
        "engines": {
            "chess": "Stockfish d12 oracle versus d2 shallow, one thread",
            "go": "KataGo v48 oracle versus v2 shallow, 4-8-stone SystemRandom opening",
        },
        "raw_positions_public": False,
    }
    path = args.root / "mill_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def prepare(args) -> dict:
    bank = ExaminerBank(args.bank)
    if not bank.promoted_items():
        raise SystemExit(f"no promoted items in retained bank {args.bank}")
    split = prepare_engine_data(
        load_jsonl(args.root / "raw" / "chess.jsonl"),
        load_jsonl(args.root / "raw" / "go.jsonl"),
        bank,
        holdout_percent=args.holdout_percent,
    )
    paths = write_engine_split(split, args.root / "data")
    report = {"paths": paths, "manifest": split.manifest}
    print(json.dumps({"phase": "prepare", **split.manifest}, sort_keys=True), flush=True)
    return report


def train(args) -> dict:
    from bcv.graph_lora import train_graph_adapter

    data_path = args.root / "data" / "train.jsonl"
    data_manifest = json.loads((args.root / "data" / "data_manifest.json").read_text(encoding="utf-8"))
    rows = load_jsonl(data_path)
    if canonical_sha256(rows) != data_manifest["train"]["sft_sha256"]:
        raise SystemExit("training buffer no longer matches its data commitment")
    result = train_graph_adapter(
        dataset_path=data_path,
        output_dir=args.root / "training",
        max_train_examples=len(rows),
        heldout_examples=0,
        epochs=args.epochs,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        mask_prompt_loss=True,
        checkpoint_every_steps=args.checkpoint_every,
        resume=True,
    )
    print(json.dumps({"phase": "train", **asdict(result)}, sort_keys=True), flush=True)
    if not result.accepted:
        raise SystemExit(result.failure or "training failed")
    return asdict(result)


def _grade_events(path: Path, system: str) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "grade" and event.get("system") == system:
            events.append(event)
    return events


def _complete_fresh_loads(bank: ExaminerBank, system: str, repeats: int, adapter: str | None, max_tokens: int) -> None:
    from bcv.registry import grade_bank
    from bcv.transformers_client import TransformersLocalClient

    events_path = bank.root / "grade_events.jsonl"
    completed = len(_grade_events(events_path, system))
    while completed < repeats:
        client = TransformersLocalClient(adapter_path=adapter, max_new_tokens=max_tokens)
        client.infrastructure = "local_rtx_5060"
        try:
            run = grade_bank(bank, system, client, burn_external=False)
        finally:
            client.unload()
        completed += 1
        print(
            json.dumps(
                {
                    "phase": "grade",
                    "system": system,
                    "fresh_load": completed,
                    "score": f"{run['passed']}/{run['items']}",
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _event_summary(events: list[dict]) -> dict:
    scores = [sum(event["results"].values()) for event in events]
    vector_hashes = [canonical_sha256(event["results"]) for event in events]
    manifests = [event.get("run_manifest", {}) for event in events]
    return {
        "runs": len(events),
        "scores": scores,
        "unique_item_outcome_vectors": len(set(vector_hashes)),
        "outcome_vector_sha256": vector_hashes,
        "fresh_model_load_per_run": True,
        "run_manifests": manifests,
    }


def _gate_summary(report: dict) -> dict:
    evidence = report["paired_evidence"]
    return {
        "verdict": report["verdict"],
        "reasons": report["reasons"],
        "gains": evidence["gains"],
        "regressions": evidence["regressions"],
        "ties": evidence["ties"],
        "exact_mcnemar_two_sided_p": evidence["exact_mcnemar_two_sided_p"],
        "resolution": evidence["resolution"],
        "regression_classifications": [
            {
                "domain": row["domain"],
                "classification": row["classification"],
                "flip_rate": (row.get("reliability") or {}).get("flip_rate"),
            }
            for row in evidence.get("regression_reliability", [])
        ],
    }


def _hardware_name() -> str:
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return process.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unavailable"


def _pregrade_identity(root: Path, bank: ExaminerBank) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "grade_identity.json"
    item_set_sha256 = canonical_sha256(sorted(item.item_id for item in bank.promoted_items()))
    if path.is_file():
        identity = json.loads(path.read_text(encoding="utf-8"))
    else:
        legacy_receipt = root / "evaluation_receipt.json"
        before_hash = None
        if legacy_receipt.is_file():
            legacy = json.loads(legacy_receipt.read_text(encoding="utf-8"))
            before_hash = (legacy.get("bank") or {}).get("before_grading_sha256")
        if not isinstance(before_hash, str) or len(before_hash) != 64:
            before_hash = bank_hash(bank)
        identity = {
            "schema_version": 1,
            "before_grading_sha256": before_hash,
            "promoted_items": len(bank.promoted_items()),
            "item_set_sha256": item_set_sha256,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    if identity.get("promoted_items") != len(bank.promoted_items()):
        raise SystemExit("retained bank item count changed after Gen-4 grading began")
    if identity.get("item_set_sha256") != item_set_sha256:
        raise SystemExit("retained bank item set changed after Gen-4 grading began")
    before_hash = identity.get("before_grading_sha256")
    if not isinstance(before_hash, str) or len(before_hash) != 64:
        raise SystemExit("Gen-4 pre-grade bank commitment is malformed")
    return identity


def grade(args) -> tuple[dict, int]:
    from bcv.gate import latest_grade_event_results
    from bcv.lora_smoke import find_fastcontext_snapshot
    from bcv.transformers_client import TransformersLocalClient

    bank = ExaminerBank(args.bank)
    adapter_path = args.root / "training" / "fastcontext_graph_repair_lora"
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise SystemExit(f"adapter not found: {adapter_path}")
    bank_before = _pregrade_identity(args.root, bank)["before_grading_sha256"]
    _complete_fresh_loads(bank, BASE_SYSTEM, args.repeats, None, args.max_new_tokens)
    _complete_fresh_loads(bank, CANDIDATE_SYSTEM, args.repeats, str(adapter_path), args.max_new_tokens)

    events_path = bank.root / "grade_events.jsonl"
    base_results = latest_grade_event_results(events_path, BASE_SYSTEM)
    candidate_results = latest_grade_event_results(events_path, CANDIDATE_SYSTEM)
    policies = {
        "strict": GatePolicy(require_retained_probe=False, regression_policy="strict"),
        "reliability_aware": GatePolicy(require_retained_probe=False, regression_policy="reliability_aware"),
    }
    reports = {}
    for name, policy in policies.items():
        report = build_gate_report(
            bank,
            BASE_SYSTEM,
            CANDIDATE_SYSTEM,
            base_results,
            candidate_results,
            policy=policy,
        )
        write_gate_report(report, args.root / "gates" / name)
        reports[name] = report

    base_events = _grade_events(events_path, BASE_SYSTEM)[-args.repeats :]
    candidate_events = _grade_events(events_path, CANDIDATE_SYSTEM)[-args.repeats :]
    adapter_sha = TransformersLocalClient._adapter_sha256(str(adapter_path))
    data_manifest = json.loads((args.root / "data" / "data_manifest.json").read_text(encoding="utf-8"))
    mill_manifest = json.loads((args.root / "mill_manifest.json").read_text(encoding="utf-8"))
    train_result = json.loads((args.root / "training" / "train_result.json").read_text(encoding="utf-8"))
    strict = reports["strict"]
    is_promotion = strict["verdict"] == "PASS"
    receipt = {
        "artifact_type": "promotion_receipt" if is_promotion else "evaluation_receipt",
        "evidence_scope": "trajectory-disjoint private Gen-4 engine student",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": _hardware_name(),
        "model": {
            "base": "microsoft/FastContext-1.0-4B-RL",
            "base_snapshot_commit": find_fastcontext_snapshot().name,
            "adapter_sha256": adapter_sha,
            "adapter_config_sha256": _sha256(adapter_path / "adapter_config.json"),
        },
        "milling": mill_manifest,
        "training_data": data_manifest,
        "training": {
            key: train_result[key]
            for key in (
                "accepted",
                "train_examples",
                "heldout_examples",
                "epochs",
                "final_loss",
                "device",
                "skipped_steps",
                "steps_completed",
                "resumed_from_step",
                "checkpoints_written",
            )
        },
        "bank": {
            "promoted_items": len(bank.promoted_items()),
            "before_grading_sha256": bank_before,
            "after_grading_sha256": bank_hash(bank),
            "grade_events_sha256": file_sha256(events_path),
        },
        "repeated_grading": {
            "baseline": _event_summary(base_events),
            "candidate": _event_summary(candidate_events),
        },
        "gate_strict": _gate_summary(strict),
        "gate_reliability_aware": _gate_summary(reports["reliability_aware"]),
        "promotion_claim": is_promotion,
        "sanitization": "counts and commitments only; no item ids, FENs, Go histories, raw outputs, or local paths",
    }
    private_receipt = args.root / "evaluation_receipt.json"
    private_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public_path = args.promotion_receipt if is_promotion else args.evaluation_receipt
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return receipt, {"PASS": 0, "HOLD": 2, "BLOCK": 3}[strict["verdict"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("mill", "prepare", "train", "grade", "all"), default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bank", type=Path, default=DEFAULT_ROOT / "bank")
    parser.add_argument("--stockfish", type=Path, default=Path("tools/stockfish/stockfish-windows-x86-64-avx2.exe"))
    parser.add_argument("--katago-dir", type=Path, default=Path("tools/katago"))
    parser.add_argument("--chess-target", type=int, default=1000)
    parser.add_argument("--go-target", type=int, default=500)
    parser.add_argument("--chess-batch", type=int, default=50)
    parser.add_argument("--go-batch", type=int, default=25)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--holdout-percent", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--promotion-receipt",
        type=Path,
        default=Path("results/gen4_engine_student_promotion_receipt.json"),
    )
    parser.add_argument(
        "--evaluation-receipt",
        type=Path,
        default=Path("results/gen4_engine_student_evaluation_receipt.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    phases = ("mill", "prepare", "train", "grade") if args.phase == "all" else (args.phase,)
    exit_code = 0
    for phase in phases:
        if phase == "mill":
            mill(args)
        elif phase == "prepare":
            prepare(args)
        elif phase == "train":
            train(args)
        else:
            _, exit_code = grade(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
