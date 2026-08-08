"""Trajectory-disjoint preparation for the private Gen-4 engine student.

Raw engine positions and the retained promotion bank stay in ignored local
state. This module emits SFT rows plus aggregate commitments, and makes the two
non-negotiable invariants executable: no promoted position enters training and
no trajectory crosses the train/holdout boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from bcv.examiner import ExamItem, ExaminerBank, game_item_prompt


INFERENCE_SYSTEM = (
    "You are a careful combinatorics research assistant. Follow the output format exactly."
)
_TRAJECTORY_ID = re.compile(r"^(?:chess|go)_[0-9a-f]{32}$")


class Gen4DataError(ValueError):
    """Engine data violates a disjointness or schema invariant."""


@dataclass(frozen=True)
class EngineDataSplit:
    train_examples: tuple[dict, ...]
    holdout_rows: tuple[dict, ...]
    manifest: dict


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Gen4DataError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise Gen4DataError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict] | tuple[dict, ...]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def position_key(row: dict) -> tuple:
    game = row.get("game")
    if game == "chess":
        fen = row.get("fen")
        if not isinstance(fen, str) or not fen:
            raise Gen4DataError("chess row requires a non-empty fen")
        return ("chess", fen)
    if game == "go9":
        moves = row.get("moves")
        to_move = row.get("to_move")
        if not isinstance(moves, list) or not all(isinstance(move, str) for move in moves):
            raise Gen4DataError("go row requires a string moves list")
        if to_move not in {"black", "white"}:
            raise Gen4DataError("go row requires to_move black or white")
        return ("go", tuple(moves), to_move)
    raise Gen4DataError(f"unsupported engine row game: {game!r}")


def bank_position_keys(bank: ExaminerBank) -> set[tuple]:
    keys: set[tuple] = set()
    for item in bank.promoted_items():
        if item.domain == "chess" and isinstance(item.payload.get("fen"), str):
            keys.add(("chess", item.payload["fen"]))
        elif item.domain == "go" and isinstance(item.payload.get("moves"), list):
            keys.add(("go", tuple(item.payload["moves"]), item.payload.get("to_move")))
    return keys


def trajectory_id(row: dict) -> str:
    identifier = row.get("trajectory_id")
    if not isinstance(identifier, str) or not _TRAJECTORY_ID.fullmatch(identifier):
        raise Gen4DataError("every Gen-4 row requires a fresh opaque trajectory_id")
    return identifier


def trajectory_bucket(identifier: str, holdout_percent: int) -> str:
    if not 1 <= holdout_percent <= 50:
        raise Gen4DataError("holdout_percent must be between 1 and 50")
    sample = int.from_bytes(hashlib.sha256(identifier.encode("ascii")).digest()[:8], "big") % 100
    return "holdout" if sample < holdout_percent else "train"


def row_to_sft(row: dict) -> dict:
    key = position_key(row)
    oracle_move = row.get("oracle_move")
    if not isinstance(oracle_move, str) or not oracle_move:
        raise Gen4DataError("engine row requires a non-empty oracle_move")
    if key[0] == "chess":
        domain = "chess"
        payload = {"rules": {"game": "chess"}, "fen": row["fen"], "acceptable": [[oracle_move]]}
    else:
        domain = "go"
        payload = {
            "rules": {"game": "go9"},
            "moves": row["moves"],
            "to_move": row["to_move"],
            "acceptable": [[oracle_move]],
        }
    item = ExamItem(
        item_id="private_training_row",
        domain=domain,
        kind="game_move",
        payload=payload,
        oracle="private_engine_trajectory",
        source="gen4_train",
        horizon="training_only",
        lineage=[],
    )
    return {
        "messages": [
            {"role": "system", "content": INFERENCE_SYSTEM},
            {"role": "user", "content": game_item_prompt(item)},
            {"role": "assistant", "content": json.dumps({"move": oracle_move})},
        ]
    }


def prepare_engine_data(
    chess_rows: list[dict],
    go_rows: list[dict],
    bank: ExaminerBank,
    *,
    holdout_percent: int = 10,
) -> EngineDataSplit:
    source_rows = [*chess_rows, *go_rows]
    if not source_rows:
        raise Gen4DataError("no engine rows supplied")
    promoted_keys = bank_position_keys(bank)
    unique: dict[tuple, dict] = {}
    duplicate_positions = 0
    bank_collisions = 0
    for row in source_rows:
        trajectory_id(row)
        key = position_key(row)
        if key in promoted_keys:
            bank_collisions += 1
            continue
        if key in unique:
            duplicate_positions += 1
            continue
        unique[key] = row

    train_rows: list[dict] = []
    holdout_rows: list[dict] = []
    for row in unique.values():
        destination = (
            holdout_rows
            if trajectory_bucket(trajectory_id(row), holdout_percent) == "holdout"
            else train_rows
        )
        destination.append(row)
    train_ids = {trajectory_id(row) for row in train_rows}
    holdout_ids = {trajectory_id(row) for row in holdout_rows}
    overlap = train_ids & holdout_ids
    if overlap:
        raise Gen4DataError(f"trajectory split overlap: {len(overlap)}")
    if not train_rows or not holdout_rows:
        raise Gen4DataError("trajectory split must produce non-empty train and holdout sets")
    if {position_key(row) for row in train_rows} & promoted_keys:
        raise Gen4DataError("promoted bank position reached training")

    train_examples = tuple(row_to_sft(row) for row in train_rows)
    domain_count = lambda rows, game: sum(row.get("game") == game for row in rows)
    manifest = {
        "schema_version": 1,
        "source": {
            "rows": len(source_rows),
            "chess_rows": len(chess_rows),
            "go_rows": len(go_rows),
            "sha256": canonical_sha256(source_rows),
        },
        "deduplication": {
            "duplicate_positions_removed": duplicate_positions,
            "promoted_bank_collisions_removed": bank_collisions,
        },
        "train": {
            "rows": len(train_rows),
            "chess_rows": domain_count(train_rows, "chess"),
            "go_rows": domain_count(train_rows, "go9"),
            "trajectories": len(train_ids),
            "sft_sha256": canonical_sha256(train_examples),
        },
        "holdout": {
            "rows": len(holdout_rows),
            "chess_rows": domain_count(holdout_rows, "chess"),
            "go_rows": domain_count(holdout_rows, "go9"),
            "trajectories": len(holdout_ids),
            "positions_sha256": canonical_sha256(holdout_rows),
        },
        "retained_bank": {
            "promoted_items": len(bank.promoted_items()),
            "engine_position_commitment": canonical_sha256(sorted(map(repr, promoted_keys))),
        },
        "invariants": {
            "train_holdout_trajectory_overlap": 0,
            "train_promoted_exact_position_overlap": 0,
            "raw_positions_public": False,
        },
    }
    return EngineDataSplit(train_examples, tuple(holdout_rows), manifest)


def write_engine_split(split: EngineDataSplit, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.jsonl"
    holdout_path = output / "holdout_positions.jsonl"
    manifest_path = output / "data_manifest.json"
    write_jsonl(train_path, split.train_examples)
    write_jsonl(holdout_path, split.holdout_rows)
    manifest_path.write_text(json.dumps(split.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "train": str(train_path),
        "holdout": str(holdout_path),
        "manifest": str(manifest_path),
    }
