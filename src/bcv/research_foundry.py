from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bcv.discovery import RuleResult
from bcv.graph_agent import (
    ProposalEvaluation,
    ProposedRule,
    _load_feedback,
    evaluate_proposals,
    run_model_conjecture_loop,
)
from bcv.local_model import LocalModelClient, LocalModelError, auto_local_client
from bcv.lora_smoke import LoraSmokeResult, run_lora_smoke_from_dataset
from bcv.schema import Event, TestResult
from bcv.store import CognitiveStore


@dataclass(frozen=True)
class FoundryRound:
    condition: str
    round_index: int
    proposed: int
    accepted: int
    rejected: int
    invalid: int
    repairs: int
    unique_accepted_total: int
    unique_repair_total: int
    evaluation_path: str
    scale_falsified: int = 0
    novel_proposed: int = 0
    novel_accepted: int = 0
    resumed: bool = False
    learner_new_examples: int = -1
    learner_probe_accepted: int = -1
    learner_promoted: bool = False


@dataclass(frozen=True)
class FoundryRun:
    condition: str
    rounds: tuple[FoundryRound, ...]
    accepted_expressions: tuple[str, ...]
    repair_expressions: tuple[str, ...]
    sft_path: str
    sft_examples: int


@dataclass(frozen=True)
class FoundryComparison:
    max_n: int
    rounds: int
    mode: str
    stateless: FoundryRun
    git_feedback: FoundryRun
    adapter: LoraSmokeResult | None
    stress_ns: tuple[int, ...] = ()
    stress_survived: int = 0
    stress_survived_vacuously: int = 0
    stress_falsified: int = 0
    stress_report_path: str | None = None


def run_research_foundry(
    root: str | Path = ".bcv_runs/research_foundry",
    max_n: int = 6,
    rounds: int = 3,
    max_rules: int = 4,
    mode: str = "scripted",
    train_adapter: bool = False,
    client: LocalModelClient | None = None,
    stress_ns: tuple[int, ...] = (),
    stress_feedback_ns: tuple[int, ...] = (),
    library_path: str | Path | None = None,
    learn: bool = False,
    proposer_model: str | None = None,
    max_new_tokens: int = 384,
) -> FoundryComparison:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if client is None:
        if mode == "model":
            client = auto_local_client()
        elif mode == "fastcontext":
            from bcv.transformers_client import TransformersLocalClient

            client = TransformersLocalClient(model_name=proposer_model, max_new_tokens=max_new_tokens)

    stress_observations = None
    judge = None
    if stress_feedback_ns:
        from bcv.graph_generalize import build_observation_pool
        from bcv.novelty import NoveltyJudge

        stress_observations, _, _ = build_observation_pool(
            ns=stress_feedback_ns,
            samples_per_np=80,
            library_path=library_path,
        )
        judge = NoveltyJudge(max_n=max_n)

    if learn:
        # A/B for continual learning: both arms get full verifier + stress
        # feedback; only the learning arm updates its adapter between rounds.
        from bcv.continual_proposer import ProposerLearner
        from bcv.model_zoo import FASTCONTEXT

        learner = ProposerLearner(
            model_name=proposer_model or FASTCONTEXT,
            root=root / "learner",
            max_n=max_n,
            max_rules=max_rules,
            max_new_tokens=max_new_tokens,
        )
        stateless = _run_condition(
            root=root / "frozen",
            condition="frozen",
            max_n=max_n,
            rounds=rounds,
            max_rules=max_rules,
            mode=mode,
            use_feedback=True,
            client=client,
            stress_observations=stress_observations,
            judge=judge,
        )
        if hasattr(client, "unload"):
            client.unload()
        git_feedback = _run_condition(
            root=root / "learning",
            condition="learning",
            max_n=max_n,
            rounds=rounds,
            max_rules=max_rules,
            mode=mode,
            use_feedback=True,
            client=client,
            stress_observations=stress_observations,
            judge=judge,
            learner=learner,
        )
    else:
        stateless = _run_condition(
            root=root / "stateless",
            condition="stateless",
            max_n=max_n,
            rounds=rounds,
            max_rules=max_rules,
            mode=mode,
            use_feedback=False,
            client=client,
            stress_observations=stress_observations,
            judge=judge,
        )
        git_feedback = _run_condition(
            root=root / "git_feedback",
            condition="git_feedback",
            max_n=max_n,
            rounds=rounds,
            max_rules=max_rules,
            mode=mode,
            use_feedback=True,
            client=client,
            stress_observations=stress_observations,
            judge=judge,
        )
    sft_path = _merge_sft(root / "foundry_sft.jsonl", (stateless.sft_path, git_feedback.sft_path))
    adapter = (
        run_lora_smoke_from_dataset(
            output_dir=root / "adapter",
            dataset_path=sft_path,
        )
        if train_adapter
        else None
    )
    stress = _stress_check(root, stateless, git_feedback, stress_ns) if stress_ns else None
    result = FoundryComparison(
        max_n=max_n,
        rounds=rounds,
        mode=mode,
        stateless=stateless,
        git_feedback=git_feedback,
        adapter=adapter,
        stress_ns=tuple(stress_ns),
        stress_survived=stress.survived if stress else 0,
        stress_survived_vacuously=stress.survived_vacuously if stress else 0,
        stress_falsified=stress.falsified if stress else 0,
        stress_report_path=str(root / "generalization" / "generalization_report.json") if stress else None,
    )
    (root / "comparison.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _record_foundry_ledger(root / "ledger", result)
    if stress:
        _record_scale_falsifications(root / "ledger", stress)
    return result


def _stress_check(root: Path, stateless: FoundryRun, git_feedback: FoundryRun, stress_ns: tuple[int, ...]):
    from bcv.graph_generalize import run_generalization

    expressions: dict[str, str] = {}
    for run in (stateless, git_feedback):
        for expression in run.accepted_expressions:
            expressions[expression] = f"{run.condition}_accepted"
        for expression in run.repair_expressions:
            expressions.setdefault(expression, f"{run.condition}_repair")
    if not expressions:
        return None
    return run_generalization(expressions, ns=stress_ns, root=root / "generalization")


def _record_scale_falsifications(root: Path, report) -> None:
    store = CognitiveStore(root)
    store.init()
    branch = "experiment/research-foundry"
    if not store.branch_exists(branch):
        store.create_branch(branch, from_branch="main")
    for result in report.results:
        if result.parseable and result.survived:
            continue
        store.commit(
            branch,
            f"scale falsification: {result.expression}",
            [
                Event(
                    event_type="scale_falsification",
                    actor="verifier",
                    message=(
                        f"accepted at small n but falsified at n={report.ns}: "
                        f"{result.expression}"
                    ),
                    output_refs=(f"expression:{result.expression}",),
                    tests=(
                        TestResult(
                            "generalization_check",
                            "fail",
                            json.dumps(
                                {
                                    "expression": result.expression,
                                    "source": result.source,
                                    "counterexamples": list(result.counterexamples),
                                },
                                sort_keys=True,
                            ),
                        ),
                    ),
                )
            ],
        )


def _run_condition(
    root: Path,
    condition: str,
    max_n: int,
    rounds: int,
    max_rules: int,
    mode: str,
    use_feedback: bool,
    client: LocalModelClient | None,
    stress_observations=None,
    judge=None,
    learner=None,
) -> FoundryRun:
    root.mkdir(parents=True, exist_ok=True)
    accepted_expressions: set[str] = set()
    repair_expressions: set[str] = set()
    round_rows: list[FoundryRound] = []
    previous_feedback = ""

    for round_index in range(rounds):
        round_root = root / f"round_{round_index:02d}"
        feedback = previous_feedback if use_feedback else ""
        _log_gpu(root, f"{condition}_round_{round_index:02d}")
        saved = _load_saved_evaluation(round_root / "proposal_evaluation.json")
        resumed = saved is not None
        if saved is not None:
            evaluation = saved
        elif mode in ("model", "fastcontext"):
            round_client = learner.client() if learner is not None else client
            if round_client is None:
                raise RuntimeError(f"{mode} mode requires a local model client")
            try:
                evaluation = run_model_conjecture_loop(
                    max_n=max_n,
                    root=round_root,
                    client=round_client,
                    max_rules=max_rules,
                    feedback=feedback,
                )
            except (LocalModelError, ValueError, KeyError) as exc:
                round_root.mkdir(parents=True, exist_ok=True)
                (round_root / "proposal_failure.txt").write_text(str(exc), encoding="utf-8")
                evaluation = evaluate_proposals(
                    (),
                    max_n=max_n,
                    backend=getattr(round_client, "backend", mode),
                    model=getattr(round_client, "model", mode),
                    root=round_root,
                )
        else:
            proposals = _scripted_proposals(condition, round_index, use_feedback, previous_feedback)
            evaluation = evaluate_proposals(
                proposals[:max_rules],
                max_n=max_n,
                backend="scripted",
                model=f"{condition}_script",
                root=round_root,
            )
        _absorb(evaluation, accepted_expressions, repair_expressions)
        previous_feedback = _load_feedback(round_root / "proposal_evaluation.json")
        scale_falsified = 0
        novel_proposed = 0
        novel_accepted = 0
        if stress_observations is not None:
            scale_falsified, stress_lines, novel_proposed, novel_accepted = _frontier_round_metrics(
                evaluation, stress_observations, judge
            )
            if stress_lines:
                previous_feedback = previous_feedback + "\n" + "\n".join(stress_lines)
        learner_new_examples = -1
        learner_probe_accepted = -1
        learner_promoted = False
        if learner is not None:
            record = learner.observe_and_update(evaluation, round_index)
            learner_new_examples = record.new_examples
            learner_probe_accepted = record.probe_accepted
            learner_promoted = record.promoted
        round_rows.append(
            FoundryRound(
                condition=condition,
                round_index=round_index,
                proposed=len(evaluation.proposed_rules),
                accepted=len(evaluation.accepted_rules),
                rejected=len(evaluation.rejected_rules),
                invalid=len(evaluation.invalid_rules),
                repairs=len(evaluation.repair_suggestions),
                unique_accepted_total=len(accepted_expressions),
                unique_repair_total=len(repair_expressions),
                evaluation_path=str(round_root / "proposal_evaluation.json"),
                scale_falsified=scale_falsified,
                novel_proposed=novel_proposed,
                novel_accepted=novel_accepted,
                resumed=resumed,
                learner_new_examples=learner_new_examples,
                learner_probe_accepted=learner_probe_accepted,
                learner_promoted=learner_promoted,
            )
        )

    sft_path = _merge_sft(
        root / "condition_sft.jsonl",
        tuple(str(root / f"round_{index:02d}" / "repair_sft.jsonl") for index in range(rounds)),
    )
    return FoundryRun(
        condition=condition,
        rounds=tuple(round_rows),
        accepted_expressions=tuple(sorted(accepted_expressions)),
        repair_expressions=tuple(sorted(repair_expressions)),
        sft_path=str(sft_path),
        sft_examples=_count_jsonl(sft_path),
    )


def _load_saved_evaluation(path: Path) -> ProposalEvaluation | None:
    """Reconstruct a completed round from disk so a crash costs one round, not the run."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None

    def rule(entry: dict) -> RuleResult:
        return RuleResult(
            name=entry["name"],
            description=entry["description"],
            support=entry["support"],
            precision=entry["precision"],
            false_positives=entry["false_positives"],
            counterexamples=tuple(entry.get("counterexamples", ())),
            promoted=entry["promoted"],
        )

    try:
        return ProposalEvaluation(
            max_n=raw["max_n"],
            graphs_checked=raw["graphs_checked"],
            backend=raw["backend"],
            model=raw["model"],
            proposed_rules=tuple(
                ProposedRule(name=p["name"], description=p["description"], expression=p["expression"])
                for p in raw["proposed_rules"]
            ),
            accepted_rules=tuple(rule(r) for r in raw["accepted_rules"]),
            rejected_rules=tuple(rule(r) for r in raw["rejected_rules"]),
            invalid_rules=tuple(dict(r) for r in raw["invalid_rules"]),
            repair_suggestions=tuple(dict(r) for r in raw["repair_suggestions"]),
        )
    except (KeyError, TypeError):
        return None


def _log_gpu(root: Path, tag: str) -> None:
    import datetime
    import subprocess

    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,power.draw",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / "gpu_log.csv").open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.datetime.now().isoformat()},{tag},{output}\n")


def _frontier_round_metrics(evaluation, stress_observations, judge):
    """Per-round frontier accounting: horizon-inherited failures and semantic novelty.

    scale_falsified counts expressions the n<=6 verifier blessed this round that die
    on the stress pool — the horizon-inheritance rate the feedback loop should drive
    down. Novelty is judged against the miner's <=2-atom conjunctive hull; only a
    parseable proposal outside that hull counts as the model expanding the frontier.
    """
    from bcv.graph_agent import compile_feature_expression

    proposed_by_name = {proposal.name: proposal.expression for proposal in evaluation.proposed_rules}
    verified: list[str] = []
    for rule in evaluation.accepted_rules:
        expression = proposed_by_name.get(rule.name) or _predicate_from_description(rule.description)
        if expression:
            verified.append(expression)
    for repair in evaluation.repair_suggestions:
        expression = str(repair.get("repair_expression", ""))
        if expression:
            verified.append(expression)

    falsified = 0
    lines: list[str] = []
    for expression in dict.fromkeys(verified):
        try:
            predicate = compile_feature_expression(expression)
        except (SyntaxError, ValueError, TypeError, KeyError):
            continue
        counterexamples = [
            obs for obs in stress_observations if predicate(obs) and not obs.greedy_is_optimal
        ]
        if counterexamples:
            falsified += 1
            lines.append(
                "SCALE_FALSIFIED "
                f"predicate={expression} "
                f"counterexample={counterexamples[0].graph.graph_id()} "
                "(passed the small-n verifier but fails on larger adversarial graphs; "
                "do not repeat this constraint pattern)"
            )

    novel_proposed = 0
    novel_accepted = 0
    if judge is not None:
        accepted_names = {rule.name for rule in evaluation.accepted_rules}
        for proposal in evaluation.proposed_rules:
            verdict = judge.judge(proposal.expression)
            if verdict.parseable and verdict.semantically_novel:
                novel_proposed += 1
                if proposal.name in accepted_names:
                    novel_accepted += 1
    return falsified, lines, novel_proposed, novel_accepted


def _scripted_proposals(
    condition: str,
    round_index: int,
    use_feedback: bool,
    previous_feedback: str,
) -> tuple[ProposedRule, ...]:
    broad = (
        ProposedRule("trees", "Trees should be exact.", "is_tree"),
        ProposedRule("forests", "Forests should be exact.", "is_forest"),
        ProposedRule("bipartite", "Bipartite graphs should be exact.", "is_bipartite"),
        ProposedRule("triangle_free", "Triangle-free graphs should be exact.", "is_triangle_free"),
    )
    if not use_feedback or round_index == 0:
        return broad
    repaired = _proposals_from_feedback(previous_feedback)
    if repaired:
        return repaired
    return (
        ProposedRule("complete_graphs", "Complete graphs should be exact.", "is_complete"),
        ProposedRule("universal_vertex", "Universal-vertex graphs should be exact.", "has_universal_vertex"),
        ProposedRule(
            "sparse_bipartite_isolated",
            "Sparse bipartite graphs with isolated vertices should be exact.",
            "is_bipartite and max_degree <= 2 and has_isolated_vertex",
        ),
        ProposedRule(
            "dense_bipartite",
            "Bipartite graphs with maximum degree at least three should be exact.",
            "is_bipartite and max_degree >= 3",
        ),
    )


def _proposals_from_feedback(feedback: str) -> tuple[ProposedRule, ...]:
    proposals: list[ProposedRule] = []
    for line in feedback.splitlines():
        if "REPAIR_SUGGESTION" not in line or "repair=" not in line:
            continue
        expression = line.split("repair=", 1)[1].split(" support=", 1)[0].strip()
        proposals.append(
            ProposedRule(
                f"repair_{len(proposals)}",
                "Verifier-mined repaired conjecture.",
                expression,
            )
        )
    return tuple(proposals)


def _absorb(
    evaluation: ProposalEvaluation,
    accepted_expressions: set[str],
    repair_expressions: set[str],
) -> None:
    proposed_by_name = {proposal.name: proposal.expression for proposal in evaluation.proposed_rules}
    for rule in evaluation.accepted_rules:
        expression = proposed_by_name.get(rule.name) or _predicate_from_description(rule.description)
        if expression:
            accepted_expressions.add(expression)
    for repair in evaluation.repair_suggestions:
        expression = str(repair.get("repair_expression", ""))
        if expression:
            repair_expressions.add(expression)


def _predicate_from_description(description: str) -> str:
    marker = "Predicate:"
    if marker not in description:
        return ""
    return description.split(marker, 1)[1].strip()


def _merge_sft(output_path: Path, input_paths: tuple[str, ...]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    lines: list[str] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line in seen:
                continue
            seen.add(line)
            lines.append(line)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def _count_jsonl(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _record_foundry_ledger(root: Path, result: FoundryComparison) -> None:
    store = CognitiveStore(root)
    store.init()
    branch = "experiment/research-foundry"
    if not store.branch_exists(branch):
        store.create_branch(branch, from_branch="main")
    store.commit(
        branch,
        "record research foundry comparison",
        [
            Event(
                event_type="foundry_comparison",
                actor="controller",
                message=(
                    f"mode={result.mode}; "
                    f"stateless accepted={len(result.stateless.accepted_expressions)}; "
                    f"git_feedback accepted={len(result.git_feedback.accepted_expressions)}"
                ),
                output_refs=("foundry:comparison",),
                tests=(
                    TestResult(
                        "foundry_comparison_recorded",
                        "pass",
                        json.dumps(
                            {
                                "stateless_sft_examples": result.stateless.sft_examples,
                                "git_feedback_sft_examples": result.git_feedback.sft_examples,
                            },
                            sort_keys=True,
                        ),
                    ),
                ),
            )
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the verifier-backed research foundry.")
    parser.add_argument("--root", default=".bcv_runs/research_foundry")
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-rules", type=int, default=4)
    parser.add_argument("--mode", choices=("scripted", "model", "fastcontext"), default="scripted")
    parser.add_argument("--train-adapter", action="store_true")
    parser.add_argument(
        "--stress-ns",
        type=int,
        nargs="*",
        default=[],
        help="Stress-check accepted rules and repairs at these larger n after the run.",
    )
    parser.add_argument(
        "--stress-feedback-ns",
        type=int,
        nargs="*",
        default=[],
        help="Stress-check every round and feed scale falsifications back into the next proposal prompt.",
    )
    parser.add_argument(
        "--library",
        default=None,
        help="adversary_library.jsonl to include in the stress-feedback pool",
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        help="Frozen-vs-learning A/B: the learning arm retrains a gated LoRA on verifier-accepted experience each round.",
    )
    parser.add_argument(
        "--proposer-model",
        default=None,
        help="HF model id for the fastcontext-mode proposer (e.g. Qwen/Qwen3-1.7B); default FastContext 4B.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    result = run_research_foundry(
        root=args.root,
        max_n=args.max_n,
        rounds=args.rounds,
        max_rules=args.max_rules,
        mode=args.mode,
        train_adapter=args.train_adapter,
        stress_ns=tuple(args.stress_ns),
        stress_feedback_ns=tuple(args.stress_feedback_ns),
        library_path=args.library,
        learn=args.learn,
        proposer_model=args.proposer_model,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
