"""Pro Action Replay for reasoning: an MCP server over live emulator sessions.

A console PAR pokes values straight into game RAM while the game runs. This server
does that to a model's inference: any MCP client (Claude, another agent, a script)
can start a reasoning session on a local model, single-step it, dump the full tape,
and — the PAR part — write directly into the running state: inject notes the model
will see after its next rewind, append text into its transcript as if it had thought
it, force save-states and rewinds it didn't ask for. Every poke is recorded in the
event log, so the flight recorder stays honest about which thoughts were the
model's own.

Run: $env:PYTHONPATH='src'; python -m bcv.emulator_mcp
(registered in the repo's .mcp.json as `reasoning-emulator`)
"""

from __future__ import annotations

import json

from mcp.server import MCPServer

from bcv._version import __version__

mcp = MCPServer("reasoning-emulator", version=__version__)

_SESSIONS: dict[str, object] = {}
_CLIENT = None
_MODEL_NAME = "Qwen/Qwen3-1.7B"
_MEMORY = None
_MEMORY_PATH = ".bcv_runs/emulator_memory.sqlite3"


def _memory():
    global _MEMORY
    if _MEMORY is None:
        from bcv.memstore import MemoryStore

        _MEMORY = MemoryStore(_MEMORY_PATH)
    return _MEMORY


def _client():
    global _CLIENT
    if _CLIENT is None:
        from bcv.transformers_client import TransformersLocalClient

        _CLIENT = TransformersLocalClient(model_name=_MODEL_NAME, max_new_tokens=260)
    return _CLIENT


def _session(session_id: str):
    if session_id not in _SESSIONS:
        raise ValueError(f"no session '{session_id}'; active: {sorted(_SESSIONS)}")
    return _SESSIONS[session_id]


@mcp.tool()
def emu_start(session_id: str, problem_index: int = 0, max_steps: int = 12) -> str:
    """Start a reasoning session on a heldout repair problem (or reuse the id to restart)."""
    from pathlib import Path

    from bcv.emulator import ReasoningEmulator, repair_task_text
    from bcv.graph_lora import _load_examples

    examples = _load_examples(Path(".bcv_runs/graph_repair_hard_rich/hard_heldout.jsonl"))
    example = examples[problem_index % len(examples)]
    task, original = repair_task_text(example)
    _SESSIONS[session_id] = ReasoningEmulator(
        _client(), task, original_expression=original, max_steps=max_steps, memory=_memory()
    )
    return json.dumps({"session_id": session_id, "original_expression": original, "task": task[:400]})


@mcp.tool()
def emu_step(session_id: str, steps: int = 1) -> str:
    """Advance the model's reasoning by up to `steps` generation steps."""
    emulator = _session(session_id)
    for _ in range(steps):
        if emulator.finished:
            break
        emulator.step()
    return emu_status(session_id)


@mcp.tool()
def emu_status(session_id: str) -> str:
    """Compact state: step count, checkpoints, notepad, transcript tail, answer."""
    emulator = _session(session_id)
    return json.dumps(
        {
            "step": emulator.step_index,
            "finished": emulator.finished,
            "answer": emulator.answer,
            "checkpoints": {name: pos for name, pos in emulator.checkpoints.items()},
            "notepad": emulator.notepad,
            "transcript_tail": emulator.transcript[-6:],
            "controls_used": emulator.controls_used,
        },
        sort_keys=True,
    )


@mcp.tool()
def emu_dump(session_id: str) -> str:
    """Full memory dump: entire transcript, notepad, checkpoints, and event log."""
    emulator = _session(session_id)
    return json.dumps(
        {
            "transcript": emulator.transcript,
            "notepad": emulator.notepad,
            "checkpoints": emulator.checkpoints,
            "events": [
                {"step": event.step, "kind": event.kind, "detail": event.detail}
                for event in emulator.events
            ],
        },
        sort_keys=True,
    )


@mcp.tool()
def emu_poke_note(session_id: str, note: str) -> str:
    """PAR: inject a note into the rewind-proof notepad. The model sees it next step."""
    emulator = _session(session_id)
    emulator.notepad.append(f"[EXTERNAL] {note}")
    emulator._log("poke_note", note)
    return emu_status(session_id)


@mcp.tool()
def emu_poke_transcript(session_id: str, text: str) -> str:
    """PAR: write directly into the model's transcript, as if it had thought it."""
    emulator = _session(session_id)
    emulator.transcript.append(text)
    emulator._log("poke_transcript", text)
    return emu_status(session_id)


@mcp.tool()
def emu_force_save(session_id: str, name: str) -> str:
    """PAR: drop a save-state the model didn't ask for."""
    emulator = _session(session_id)
    emulator._do_save(name)
    emulator._log("poke_save", name)
    return emu_status(session_id)


@mcp.tool()
def emu_force_load(session_id: str, name: str, note: str = "externally rewound") -> str:
    """PAR: force a rewind to a checkpoint, with a note the model keeps."""
    emulator = _session(session_id)
    emulator._do_load(f"{name} :: {note}")
    emulator._log("poke_load", f"{name} :: {note}")
    return emu_status(session_id)


@mcp.tool()
def emu_check(session_id: str, expression: str) -> str:
    """Run the exact verifier on any expression without touching the model's transcript."""
    from bcv.graph_lora import _analyze_expression

    emulator = _session(session_id)
    analysis = _analyze_expression(expression, emulator.original_expression, emulator.max_n)
    return json.dumps(
        {
            "expression": expression,
            "parseable": analysis.parseable,
            "verified": analysis.verified,
            "refines_original": analysis.refines_original,
            "support": analysis.support,
        },
        sort_keys=True,
    )


@mcp.tool()
def emu_memory_remember(content: str, entities: str, kind: str = "episodic", confidence: float = 0.8) -> str:
    """PAR: write a memory into the persistent store (entities comma-separated).
    It will page into ANY session whose context cues it — cross-session influence."""
    store = _memory()
    memory_id = store.remember(
        content,
        tuple(e.strip() for e in entities.split(",") if e.strip()),
        step=0,
        kind=kind,
        source="external",
        confidence=confidence,
    )
    return json.dumps({"memory_id": memory_id, "counts": store.counts()})


@mcp.tool()
def emu_memory_page(context_entities: str, token_budget: int = 80) -> str:
    """Preview what the pager would push for a given context (comma-separated entities)."""
    store = _memory()
    paged = store.page_in(
        tuple(e.strip() for e in context_entities.split(",") if e.strip()),
        step=0,
        token_budget=token_budget,
        reinforce=False,
    )
    return json.dumps([m.content for m in paged])


@mcp.tool()
def emu_result(session_id: str) -> str:
    """Final scored result for a finished (or aborted) session."""
    from dataclasses import asdict

    emulator = _session(session_id)
    return json.dumps(asdict(emulator.result()), sort_keys=True)


if __name__ == "__main__":
    mcp.run()
