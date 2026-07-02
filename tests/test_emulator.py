from __future__ import annotations

import json

from bcv.emulator import ReasoningEmulator, _split_at_first_control, repair_task_text


class ScriptedClient:
    """Plays a fixed tape of model outputs: save, fail a check, rewind, succeed."""

    def __init__(self):
        self.outputs = [
            "The counterexamples all have max_degree 2.\nSAVE base",
            "Try keeping only degree-2-capped graphs.\nCHECK (is_tree) and (max_degree_le_2)",
            "That failed; the counterexamples ARE degree-capped. Exclude them instead.\n"
            "LOAD base :: max_degree_le_2 keeps the counterexamples; need max_degree >= 3",
            "Excluding low-degree graphs should work.\nCHECK (is_tree) and (max_degree >= 3)",
            "Verifier accepted it.\nANSWER (is_tree) and (max_degree >= 3)",
        ]
        self.calls = 0

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return output


def test_split_at_first_control():
    segment, control, argument = _split_at_first_control("thinking...\nSAVE alpha\nmore text")
    assert segment == "thinking...\n"
    assert control == "SAVE"
    assert argument.strip() == "alpha"
    segment, control, argument = _split_at_first_control("no controls here")
    assert control is None


def test_episode_rewind_with_notes_and_verified_answer():
    emulator = ReasoningEmulator(
        ScriptedClient(),
        task="repair is_tree",
        original_expression="is_tree",
        max_steps=10,
    )
    result = emulator.run_episode()

    assert result.answer == "(is_tree) and (max_degree >= 3)"
    assert result.answer_verified
    assert result.answer_refines
    # The dead branch was garbage-collected from context...
    rendered = emulator.render_prompt()
    assert "max_degree_le_2" not in "\n".join(emulator.transcript).split("REWOUND")[0]
    # ...but the note survived the rewind and the flight recorder kept the branch.
    assert any("max_degree >= 3" in note for note in result.notes)
    assert any(event["kind"] == "dead_branch" for event in result.events)
    assert result.controls_used["LOAD"] == 1
    assert result.controls_used["CHECK"] == 2


def test_checkpoint_semantics_and_par_pokes():
    emulator = ReasoningEmulator(
        ScriptedClient(),
        task="repair is_tree",
        original_expression="is_tree",
        max_steps=3,
    )
    emulator.step()  # SAVE base
    assert "base" in emulator.checkpoints
    emulator.step()  # failing CHECK enters transcript
    assert any("VERIFIER" in line for line in emulator.transcript)

    # PAR-style pokes: external writes land in state and are logged.
    emulator.notepad.append("[EXTERNAL] try girth constraints")
    emulator.transcript.append("(externally injected thought)")
    emulator._do_load("base :: externally rewound")
    assert "(externally injected thought)" not in emulator.transcript
    assert any("externally rewound" in note for note in emulator.notepad)
    assert any("[EXTERNAL]" in note for note in emulator.notepad)


def test_repair_task_text_renders_original():
    example = {
        "messages": [
            {"role": "system", "content": "x"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_expression": "is_tree",
                        "counterexamples": [{"n": 6, "max_degree": 2}],
                        "kept_examples": [{"n": 4, "max_degree": 3}],
                        "false_positive_count": 9,
                    }
                ),
            },
            {"role": "assistant", "content": json.dumps({"repair_expression": "x"})},
        ]
    }
    task, original = repair_task_text(example)
    assert original == "is_tree"
    assert "is_tree" in task and "counterexample" in task.lower()
