"""A console-emulator control surface over a model's own inference.

The model reasons in a TRANSCRIPT it can checkpoint and rewind like an emulator
save-state, with one deliberate deviation from emulator semantics: rewinding is
rewind-WITH-NOTES. A LOAD truncates the transcript back to a checkpoint (the dead
branch vanishes from context — garbage-collected reasoning) but the model's note
about why the branch died survives in a NOTEPAD that is immune to rewinds. Pure
save-state semantics would restore the model's ignorance along with its state and
loop it into the same wall forever; the note is the information carried back from
the abandoned timeline.

Controls the model can emit, each on its own line:

  SAVE <name>                  checkpoint the current transcript
  LOAD <name> :: <note>        rewind to a checkpoint; the note survives
  CHECK <expression>           run the exact verifier on a candidate repair
  SKETCH                       fast-forward: a cheap high-temperature rollout is
                               taken, distilled to a note, and NOT committed
  ANSWER <expression>          commit the final answer and end the episode

CHECK is what grounds the buttons: rewind decisions follow verifier verdicts, not
vibes. Every operation (including the text of dead branches) goes to an event log —
the flight recorder — even though the model's context no longer carries it.

World-touching side effects would be save-barriers; this testbed is pure reasoning
plus a read-only verifier, so the whole tape is legally rewindable.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


CONTROL_PATTERN = re.compile(r"^(SAVE|LOAD|CHECK|SKETCH|ANSWER)\b(.*)$", re.MULTILINE)


@dataclass
class EmulatorEvent:
    step: int
    kind: str
    detail: str


@dataclass
class EpisodeResult:
    task: str
    answer: str | None
    answer_parseable: bool
    answer_verified: bool
    answer_refines: bool
    steps: int
    generated_chars: int
    final_context_chars: int
    controls_used: dict[str, int]
    notes: list[str]
    events: list[dict]


class ReasoningEmulator:
    def __init__(
        self,
        client,
        task: str,
        original_expression: str | None = None,
        max_n: int = 6,
        max_steps: int = 10,
        gen_tokens: int = 220,
        sketch_tokens: int = 160,
        temperature: float = 0.3,
        memory=None,
        memory_budget: int = 80,
        require_checked_answer: bool = True,
    ) -> None:
        self.memory = memory
        self.memory_budget = memory_budget
        self.require_checked_answer = require_checked_answer
        self.client = client
        self.task = task
        self.original_expression = original_expression
        self.max_n = max_n
        self.max_steps = max_steps
        self.gen_tokens = gen_tokens
        self.sketch_tokens = sketch_tokens
        self.temperature = temperature

        self.transcript: list[str] = []
        self.checkpoints: dict[str, int] = {}
        self.notepad: list[str] = []
        self.events: list[EmulatorEvent] = []
        self.controls_used: dict[str, int] = {}
        self.answer: str | None = None
        self.generated_chars = 0
        self.step_index = 0
        self.finished = False

    # ------------------------------------------------------------- rendering

    RENDER_WINDOW = 12
    """Only the most recent transcript segments are rendered into the prompt. The
    full transcript stays in state (checkpoints/rewinds operate on all of it), but
    unbounded prompt growth on an 8 GB card spills the KV cache into system memory
    and decode slows by an order of magnitude."""

    def render_prompt(self) -> str:
        notepad = "\n".join(f"- {note}" for note in self.notepad) or "- (empty)"
        visible = self.transcript[-self.RENDER_WINDOW :]
        elided = len(self.transcript) - len(visible)
        transcript = "\n".join(visible) or "(empty — begin reasoning)"
        if elided > 0:
            transcript = f"[... {elided} earlier segments elided ...]\n" + transcript
        checkpoints = ", ".join(sorted(self.checkpoints)) or "(none)"
        memory_block = ""
        if self.memory is not None:
            entities = tuple(
                {word for line in ([self.task] + self.transcript[-4:]) for word in re.findall(r"[A-Za-z_]{3,}", line)}
            )
            paged = self.memory.page_in(entities, self.step_index, self.memory_budget)
            if paged:
                memory_block = "\nMEMORY (paged in, uninvited — may be relevant):\n" + "\n".join(
                    f"- {item.content}" for item in paged
                ) + "\n"
        return f"""You are solving a problem with EMULATOR CONTROLS over your own reasoning.

Available controls — emit exactly one, alone on its own line, when you want to use it:
SAVE <name>            (checkpoint your reasoning here)
LOAD <name> :: <note>  (rewind to a checkpoint; your note about the failed branch is kept)
CHECK <expression>     (run the exact verifier on a candidate repair expression)
SKETCH                 (take a quick exploratory look ahead without committing to it)
ANSWER <expression>    (submit your final repair expression)

Strategy that works: reason briefly, SAVE before trying a candidate, CHECK it,
and if the verifier rejects it, LOAD back with a note and try a different one.
Submit with ANSWER only after a CHECK passes.

CLOCK: step {self.step_index} of {self.max_steps}. Time moves forward only; rewinds
restore your transcript, not the clock — spent steps stay spent.

TASK:
{self.task}

NOTEPAD (survives rewinds):
{notepad}
{memory_block}
CHECKPOINTS: {checkpoints}

TRANSCRIPT SO FAR:
{transcript}

Continue from the end of the transcript. Reason in at most a few sentences, then use a control."""

    # ------------------------------------------------------------- controls

    def run_episode(self) -> EpisodeResult:
        while not self.finished and self.step_index < self.max_steps:
            self.step()
        return self.result()

    def step(self) -> None:
        if self.finished:
            return
        self.step_index += 1
        # Loop-breaker: perseveration (re-checking an identical expression) is a
        # low-temperature attractor; repeated REPEAT verdicts heat the sampler.
        temperature = self.temperature
        if getattr(self, "_repeat_streak", 0) >= 2:
            temperature = 0.9
            self._log("loop_breaker", f"temperature raised to {temperature}")
        output = self.client.generate_text(self.render_prompt(), temperature=temperature)
        self.generated_chars += len(output)
        segment, control, argument = _split_at_first_control(output)
        if segment.strip():
            self.transcript.append(segment.strip())
        if control is None:
            self._log("drift", "no control emitted; nudging")
            self.transcript.append("(Reminder: use SAVE / CHECK / LOAD / ANSWER controls.)")
            return
        self._log("control", f"{control} {argument}".strip())
        self.controls_used[control] = self.controls_used.get(control, 0) + 1
        handler = {
            "SAVE": self._do_save,
            "LOAD": self._do_load,
            "CHECK": self._do_check,
            "SKETCH": self._do_sketch,
            "ANSWER": self._do_answer,
        }[control]
        handler(argument.strip())

    def _do_save(self, argument: str) -> None:
        name = _slug(argument) or f"cp{len(self.checkpoints)}"
        self.checkpoints[name] = len(self.transcript)
        self.transcript.append(f"[SAVED checkpoint '{name}']")

    def _do_load(self, argument: str) -> None:
        name_part, _, note = argument.partition("::")
        name = _slug(name_part)
        if name not in self.checkpoints:
            self.transcript.append(f"[LOAD failed: no checkpoint '{name}']")
            return
        position = self.checkpoints[name]
        dead_branch = "\n".join(self.transcript[position:])
        self._log("dead_branch", dead_branch[:2000])
        self.transcript = self.transcript[:position]
        note = note.strip() or "branch abandoned without a note"
        self.notepad.append(f"[t={self.step_index}] {note}")
        self.transcript.append(f"[REWOUND to '{name}'; note kept]")

    def _do_check(self, argument: str) -> None:
        from bcv.graph_lora import _analyze_expression

        expression = _clean_expression(argument)
        if expression in getattr(self, "_checked", {}):
            previous = self._checked[expression]
            self._repeat_streak = getattr(self, "_repeat_streak", 0) + 1
            self.transcript.append(
                f"VERIFIER[{expression}]: REPEAT — already checked, result unchanged: {previous} "
                "Try a DIFFERENT expression."
            )
            return
        self._repeat_streak = 0
        analysis = _analyze_expression(expression, self.original_expression, self.max_n)
        if not analysis.parseable:
            verdict = "INVALID — not a legal DSL expression (plain features/operators only, balanced parens)."
        elif not analysis.verified:
            verdict = "REJECTED — matching graphs where the claim fails."
        elif self.original_expression and not analysis.refines_original:
            verdict = "REJECTED — verified but not a refinement of the original."
        else:
            verdict = f"ACCEPTED — verified, support {analysis.support}. Submit it with ANSWER now."
        if not hasattr(self, "_checked"):
            self._checked: dict[str, str] = {}
        self._checked[expression] = verdict
        self.transcript.append(f"VERIFIER[{expression}]: {verdict}")
        self._log("verifier", f"{expression} -> {verdict}")

    def _do_sketch(self, argument: str) -> None:
        prompt = self.render_prompt() + "\n(SKETCH MODE: think fast and loose; this will not be kept.)"
        sketch = self.client.generate_text(prompt, temperature=0.9)
        self.generated_chars += len(sketch)
        summary = sketch.strip().replace("\n", " ")[:300]
        self.notepad.append(f"SKETCH: {summary}")
        self.transcript.append("[SKETCH taken; distilled to notepad]")
        self._log("sketch", summary)

    def _do_answer(self, argument: str) -> None:
        expression = _clean_expression(argument)
        if self.require_checked_answer:
            verdict = getattr(self, "_checked", {}).get(expression, "")
            if not verdict.startswith("ACCEPTED"):
                # The gate found loose in the 4B run: unverified answers convert
                # into a CHECK instead of ending the episode on a guess.
                self.transcript.append(
                    f"[ANSWER blocked — '{expression}' has no ACCEPTED check. Verifying it instead.]"
                )
                self._log("gate", f"unverified ANSWER converted to CHECK: {expression}")
                self._do_check(expression)
                verdict = self._checked.get(expression, "")
                if not verdict.startswith("ACCEPTED"):
                    return
        self.answer = expression
        self.finished = True

    # ------------------------------------------------------------- reporting

    def result(self) -> EpisodeResult:
        from bcv.graph_lora import _analyze_expression

        parseable = verified = refines = False
        if self.answer:
            analysis = _analyze_expression(self.answer, self.original_expression, self.max_n)
            parseable = analysis.parseable
            verified = analysis.verified
            refines = analysis.refines_original if self.original_expression else analysis.verified
        return EpisodeResult(
            task=self.task[:200],
            answer=self.answer,
            answer_parseable=parseable,
            answer_verified=verified,
            answer_refines=refines,
            steps=self.step_index,
            generated_chars=self.generated_chars,
            final_context_chars=len(self.render_prompt()),
            controls_used=dict(self.controls_used),
            notes=list(self.notepad),
            events=[asdict(event) for event in self.events],
        )

    def _log(self, kind: str, detail: str) -> None:
        self.events.append(EmulatorEvent(step=self.step_index, kind=kind, detail=detail))


def _split_at_first_control(output: str) -> tuple[str, str | None, str]:
    # Structured packet grammar first: a JSON line {"control": "CHECK", "arg": ...}
    # is an exact channel — prose grammars demand read-copy-emit fidelity that
    # small receivers demonstrably lack (the 1.7B transduction null).
    for line_match in re.finditer(r"^\{.*\}\s*$", output, re.MULTILINE):
        try:
            packet = json.loads(line_match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(packet, dict) and str(packet.get("control", "")).upper() in (
            "SAVE",
            "LOAD",
            "CHECK",
            "SKETCH",
            "ANSWER",
        ):
            control = str(packet["control"]).upper()
            argument = str(packet.get("arg", ""))
            if control == "LOAD" and "note" in packet:
                argument = f"{argument} :: {packet['note']}"
            return output[: line_match.start()], control, argument
    match = CONTROL_PATTERN.search(output)
    if match is None:
        return output, None, ""
    return output[: match.start()], match.group(1), match.group(2)


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")


def _clean_expression(argument: str) -> str:
    """Strip decoration (backticks, angle brackets, quotes) and balance parentheses."""
    expression = argument.strip().strip("`").strip()
    expression = expression.strip("<>").strip("\"'").strip()
    open_count = expression.count("(")
    close_count = expression.count(")")
    if open_count > close_count:
        expression = expression + ")" * (open_count - close_count)
    elif close_count > open_count:
        surplus = close_count - open_count
        while surplus and expression.endswith(")"):
            expression = expression[:-1].rstrip()
            surplus -= 1
    return expression


# ----------------------------------------------------------------- testbed


def repair_task_text(example: dict) -> tuple[str, str]:
    """Render a hard-dataset repair example as an emulator task."""
    payload = json.loads(example["messages"][1]["content"])
    original = payload["original_expression"]
    task = (
        "Repair a rejected graph conjecture. The predicate "
        f"`{original}` matched graphs where degree-descending greedy coloring was NOT optimal.\n"
        f"Counterexample evidence: {json.dumps(payload.get('counterexamples', []), sort_keys=True)}\n"
        f"Examples that must stay covered: {json.dumps(payload.get('kept_examples', []), sort_keys=True)}\n"
        "Find a strictly narrower DSL predicate (features: n, m, density, max_degree, min_degree, "
        "is_connected, is_complete, is_forest, is_tree, is_bipartite, is_triangle_free, "
        "max_degree_le_2, has_universal_vertex, has_isolated_vertex, is_regular, num_components, "
        "clique_number, girth; operators: and, or, not, ==, !=, <, <=, >, >=) that keeps positives "
        "and excludes every counterexample. It must start from the original predicate, e.g. "
        f"`({original}) and (<constraint>)`."
    )
    return task, original


def run_benchmark(
    dataset_path: str | Path,
    model_name: str = "Qwen/Qwen3-1.7B",
    limit: int = 6,
    max_steps: int = 10,
    root: str | Path = ".bcv_runs/emulator_benchmark",
    linear_only: bool = False,
    emulator_only: bool = False,
) -> dict:
    """Emulator-with-controls vs linear CoT on verifier-scored repair problems."""
    from bcv.graph_lora import _analyze_expression, _load_examples
    from bcv.transformers_client import TransformersLocalClient

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    examples = _load_examples(Path(dataset_path))[:limit]
    client = TransformersLocalClient(model_name=model_name, max_new_tokens=260)

    rows = []
    for index, example in enumerate(examples):
        task, original = repair_task_text(example)
        row: dict = {"index": index, "original": original}
        if not linear_only:
            emulator = ReasoningEmulator(client, task, original_expression=original, max_steps=max_steps)
            result = emulator.run_episode()
            row["emulator"] = asdict(result)
        if not emulator_only:
            linear_prompt = (
                task
                + "\nThink step by step, then give your final repair expression on the last "
                "line in the form ANSWER <expression>."
            )
            output = client.generate_text(linear_prompt, temperature=0.3)
            answer = _last_answer(output)
            analysis = _analyze_expression(answer, original, 6) if answer else None
            row["linear"] = {
                "answer": answer,
                "answer_refines": bool(analysis and analysis.verified and analysis.refines_original),
                "generated_chars": len(output),
            }
        rows.append(row)
        (root / "benchmark.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "problems": len(rows),
        "model": model_name,
        "emulator_solved": sum(1 for row in rows if row.get("emulator", {}).get("answer_refines")),
        "linear_solved": sum(1 for row in rows if row.get("linear", {}).get("answer_refines")),
        "emulator_chars": sum(row.get("emulator", {}).get("generated_chars", 0) for row in rows),
        "linear_chars": sum(row.get("linear", {}).get("generated_chars", 0) for row in rows),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    client.unload()
    return summary


def _last_answer(output: str) -> str | None:
    answers = re.findall(r"^ANSWER\s+(.+)$", output, re.MULTILINE)
    if answers:
        return answers[-1].strip().strip("`")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reasoning-emulator benchmark.")
    parser.add_argument("--dataset-path", default=".bcv_runs/graph_repair_hard_rich/hard_heldout.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--root", default=".bcv_runs/emulator_benchmark")
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument("--emulator-only", action="store_true")
    args = parser.parse_args()
    summary = run_benchmark(
        dataset_path=args.dataset_path,
        model_name=args.model,
        limit=args.limit,
        max_steps=args.max_steps,
        root=args.root,
        linear_only=args.linear_only,
        emulator_only=args.emulator_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
