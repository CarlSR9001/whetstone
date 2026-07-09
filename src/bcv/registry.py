"""The verifier registry: every exam domain as a plug with the same three verbs.

A domain plugin knows how to MINT items at its frontier, render the PROMPT a
system under exam sees, and GRADE a raw answer against the item's checker
spec. The registry routes by ``ExamItem.domain``, so a single bank can mix
graph repairs, game moves, code tasks, and panel cases, and a single
``grade_bank`` call grades a candidate across all of them.

``grade_bank`` is the product-side sibling of ``examiner.grade_system``: it
accepts any adapter from bcv.candidates (including stored answers, which have
no free-prompt ability), records grades through the bank's append-only event
trail, and — when the candidate is external — burns every exposed item via the
bank's burn accounting. The leakage rule is enforced here, in code, not in a
deployment checklist.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from bcv.examiner import (
    ExamItem,
    ExaminerBank,
    game_item_prompt,
    grade_game_answer,
    grade_repair_answer,
    mint_game_items,
    mint_repair_items,
    repair_item_prompt,
)


class Candidate(Protocol):
    provider: str
    is_external: bool

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str: ...


@dataclass(frozen=True)
class DomainPlugin:
    name: str
    mint: Callable[..., list[ExamItem]]  # (buffers, max_items, seed) -> items
    prompt: Callable[[ExamItem], str]
    # grade(item, raw_text_or_answer, context) -> bool
    grade: Callable[[ExamItem, object, dict], bool]


# --------------------------------------------------------------- graph repair


def _mint_graph(domain_name: str):
    def mint(buffers: list[str | Path], max_items: int, seed: int) -> list[ExamItem]:
        from bcv.domains import DOMAINS

        return mint_repair_items(DOMAINS[domain_name], buffers, max_items=max_items, seed=seed)

    return mint


def _grade_repair(item: ExamItem, answer: object, context: dict) -> bool:
    from bcv.transformers_client import extract_json

    if isinstance(answer, str):
        parsed = extract_json(answer)
        answer = parsed.get("repair_expression") if isinstance(parsed, dict) else answer.strip() or None
    pools = context.setdefault("stress_pools", {})
    if item.domain not in pools:
        from bcv.domains import DOMAINS
        from bcv.refinery import _stress_pool

        pools[item.domain] = _stress_pool(
            DOMAINS[item.domain],
            context.get("stress_ns", (7, 8)),
            40,
            context.get("seed", 0),
            Path(context.get("scratch", ".bcv_runs/registry_tmp")) / "no_library.jsonl",
        )
    return grade_repair_answer(item, answer if isinstance(answer, str) else None, pool=pools[item.domain])


# --------------------------------------------------------------------- games


def _mint_games(buffers: list[str | Path], max_items: int, seed: int) -> list[ExamItem]:
    return mint_game_items(max_items=max_items, seed=seed)


def _grade_game(item: ExamItem, answer: object, context: dict) -> bool:
    if isinstance(answer, str):
        from bcv.transformers_client import extract_json

        parsed = extract_json(answer)
        answer = parsed.get("move") if isinstance(parsed, dict) else None
    return grade_game_answer(item, answer)


# ---------------------------------------------------------------------- code


def _mint_code(buffers: list[str | Path], max_items: int, seed: int) -> list[ExamItem]:
    from bcv.code_domain import mint_code_items

    return mint_code_items(buffers, max_items=max_items, seed=seed)


def _grade_code(item: ExamItem, answer: object, context: dict) -> bool:
    from bcv.code_domain import grade_code_answer

    return grade_code_answer(item, str(answer) if answer is not None else None,
                             timeout_seconds=context.get("code_timeout", 15.0))


def _prompt_code(item: ExamItem) -> str:
    from bcv.code_domain import code_item_prompt

    return code_item_prompt(item)


# ------------------------------------------------------------------- support


def _mint_support(buffers: list[str | Path], max_items: int, seed: int) -> list[ExamItem]:
    from bcv.panel import mint_support_items

    corpus = Path("sample_docs/support_tickets.jsonl")
    if not corpus.exists():
        return []
    tickets = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    return mint_support_items(tickets, max_items=max_items)


def _grade_support(item: ExamItem, answer: object, context: dict) -> bool:
    from bcv.panel import grade_support_answer

    return grade_support_answer(item, str(answer) if answer is not None else None)


def _prompt_support(item: ExamItem) -> str:
    from bcv.panel import support_item_prompt

    return support_item_prompt(item)


REGISTRY: dict[str, DomainPlugin] = {
    "coloring": DomainPlugin("coloring", _mint_graph("coloring"), repair_item_prompt, _grade_repair),
    "mis": DomainPlugin("mis", _mint_graph("mis"), repair_item_prompt, _grade_repair),
    "playground": DomainPlugin("playground", _mint_games, game_item_prompt, _grade_game),
    "arcade": DomainPlugin("arcade", _mint_games, game_item_prompt, _grade_game),
    "chess": DomainPlugin("chess", _mint_games, game_item_prompt, _grade_game),
    "go": DomainPlugin("go", _mint_games, game_item_prompt, _grade_game),
    "code": DomainPlugin("code", _mint_code, _prompt_code, _grade_code),
    "support": DomainPlugin("support", _mint_support, _prompt_support, _grade_support),
}

MINTABLE_DOMAINS = ("coloring", "mis", "playground", "code", "support")


def plugin_for(item: ExamItem) -> DomainPlugin:
    plugin = REGISTRY.get(item.domain)
    if plugin is None:
        raise KeyError(f"no registered verifier for domain {item.domain!r}")
    return plugin


# ------------------------------------------------------------------ minting


def mint_domain(
    bank: ExaminerBank,
    domain: str,
    buffers: list[str | Path] | None = None,
    max_items: int = 8,
    seed: int = 0,
) -> dict:
    plugin = REGISTRY[domain]
    minted = plugin.mint(buffers or [], max_items, seed)
    promoted = quarantined = 0
    for item in minted:
        bank.add(item)
        if item.status == "quarantined":
            quarantined += 1
        elif bank.promote(item.item_id):
            promoted += 1
    bank.save()
    return {"domain": domain, "minted": len(minted), "promoted": promoted, "quarantined": quarantined}


def candidate_run_manifest(
    candidate: Candidate,
    max_items: int | None,
    stress_ns: tuple[int, ...],
    seed: int,
    burn_external: bool,
) -> dict:
    """Non-secret receipt for a grading run.

    The ledger must be reproducible without copying prompts, answers, API keys,
    or a command line that might itself contain credentials.
    """
    manifest: dict = {
        "adapter": type(candidate).__name__,
        "backend": str(getattr(candidate, "backend", "unknown")),
        "external": bool(candidate.is_external),
        "max_items": max_items,
        "stress_ns": list(stress_ns),
        "seed": seed,
        "temperature": 0.0,
        "burn_external": burn_external,
    }
    if hasattr(candidate, "model") and hasattr(candidate, "host"):
        manifest.update({
            "endpoint_host": str(candidate.host),
            "model": str(candidate.model),
            "max_tokens": int(getattr(candidate, "max_tokens", 0)),
            "timeout_seconds": float(getattr(candidate, "timeout_seconds", 0.0)),
        })
    elif hasattr(candidate, "argv"):
        command = "\0".join(str(part) for part in candidate.argv)
        manifest["command_sha256"] = hashlib.sha256(command.encode("utf-8")).hexdigest()
        manifest["timeout_seconds"] = float(getattr(candidate, "timeout_seconds", 0.0))
    elif hasattr(candidate, "path"):
        manifest["answers_file"] = Path(candidate.path).name
    return manifest


# ------------------------------------------------------------------ grading


def grade_bank(
    bank: ExaminerBank,
    system: str,
    candidate: Candidate,
    max_items: int | None = None,
    stress_ns: tuple[int, ...] = (7, 8),
    seed: int = 0,
    burn_external: bool = True,
) -> dict:
    """Grade a candidate on every promoted item, whatever its domain.

    External candidates trigger burn accounting: each exposed item is
    permanently removed from the reusable pools after grading. Set
    ``burn_external=False`` only if you have decided, on the record, that the
    endpoint is inside your trust boundary despite its host.
    """
    from bcv.candidates import StoredAnswerCandidate

    started = time.perf_counter()
    context: dict = {"stress_ns": stress_ns, "seed": seed, "scratch": str(bank.root / "registry_tmp")}
    items = bank.promoted_items()
    if max_items:
        items = items[:max_items]
    results: dict[str, bool] = {}
    exposed: list[str] = []
    for item in items:
        plugin = plugin_for(item)
        if isinstance(candidate, StoredAnswerCandidate):
            answer: object = candidate.answer_for(item.item_id)
        else:
            answer = candidate.generate_text(plugin.prompt(item), temperature=0.0)
            if candidate.is_external:
                exposed.append(item.item_id)
        results[item.item_id] = plugin.grade(item, answer, context)
    manifest = candidate_run_manifest(candidate, max_items, stress_ns, seed, burn_external)
    manifest["items_graded"] = len(results)
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    bank.record_grades(system, results, run_manifest=manifest)
    burned: list[str] = []
    if exposed and burn_external:
        for item_id in exposed:
            bank.burn(item_id, provider=candidate.provider, reason="graded via external endpoint")
            burned.append(item_id)
    bank.save()
    return {
        "system": system,
        "provider": candidate.provider,
        "items": len(results),
        "passed": sum(results.values()),
        "external": bool(exposed),
        "burned": burned,
        "run_manifest": manifest,
        "results": results,
    }
