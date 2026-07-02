"""Verifier-grounded continual learning for a small proposer model.

The 4B proposer bugchecked the GPU driver under sustained decode; a ~2B model is
cheap enough to run all night — but starts weaker. This module makes the small model
*earn* capability across rounds instead of assuming it zero-shot:

- every round's verifier-ACCEPTED rules and mined repairs become proposal-format SFT
  examples in a persistent replay buffer (replay across rounds is what prevented
  forgetting in the continuity experiments; the verifier is the grounding that
  prevented drift in the seed-agent experiments — no self-generated example enters
  the buffer without passing the exact checker),
- after each round a LoRA retrains on the whole buffer,
- the new adapter is PROMOTED only if it beats the incumbent on a fixed probe
  (deterministic proposal generation scored by verifier-accepted count) — the
  blueprint's "no model delta without replay tests" gate, and every state change is
  checkpointed so a machine crash costs one round, not the run.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoundLearningRecord:
    round_index: int
    new_examples: int
    buffer_examples: int
    trained: bool
    probe_accepted: int
    best_probe: int
    promoted: bool
    adapter_path: str | None


class ProposerLearner:
    def __init__(
        self,
        model_name: str,
        root: str | Path,
        max_n: int = 6,
        max_rules: int = 6,
        max_new_tokens: int = 384,
        lora_r: int = 4,
        lora_alpha: int = 8,
        epochs: int = 1,
        max_buffer_examples: int = 64,
        probe_fn=None,
        trainer_fn=None,
    ) -> None:
        self.model_name = model_name
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_n = max_n
        self.max_rules = max_rules
        self.max_new_tokens = max_new_tokens
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.epochs = epochs
        self.max_buffer_examples = max_buffer_examples
        self.buffer_path = self.root / "experience_buffer.jsonl"
        self.state_path = self.root / "learner_state.json"
        self.adapter_path: str | None = None
        self.best_probe: int | None = None
        self.records: list[RoundLearningRecord] = []
        self._client = None
        self._probe_fn = probe_fn
        self._trainer_fn = trainer_fn
        self._load_state()

    # ------------------------------------------------------------------ client

    def client(self):
        if self._client is None:
            from bcv.transformers_client import TransformersLocalClient

            self._client = TransformersLocalClient(
                adapter_path=self.adapter_path,
                max_new_tokens=self.max_new_tokens,
                model_name=self.model_name,
            )
        return self._client

    def _drop_client(self) -> None:
        if self._client is not None:
            self._client.unload()
            self._client = None

    # ------------------------------------------------------------------ probing

    def ensure_baseline(self) -> int:
        if self.best_probe is None:
            self.best_probe = self._probe(adapter_path=self.adapter_path)
            self._save_state()
        return self.best_probe

    def _probe(self, adapter_path: str | None) -> int:
        if self._probe_fn is not None:
            return self._probe_fn(adapter_path)
        from bcv.graph_agent import evaluate_proposals, propose_with_model
        from bcv.local_model import LocalModelError
        from bcv.transformers_client import TransformersLocalClient

        self._drop_client()
        client = TransformersLocalClient(
            adapter_path=adapter_path,
            max_new_tokens=self.max_new_tokens,
            model_name=self.model_name,
        )
        try:
            proposals = propose_with_model(client, max_rules=self.max_rules, temperature=0.0)
        except (LocalModelError, ValueError, KeyError):
            return 0
        finally:
            client.unload()
        if not proposals:
            return 0
        evaluation = evaluate_proposals(
            proposals,
            max_n=self.max_n,
            backend="probe",
            model=self.model_name,
            root=self.root / "probe_tmp",
        )
        return len(evaluation.accepted_rules)

    # ------------------------------------------------------------------ learning

    def observe_and_update(self, evaluation, round_index: int) -> RoundLearningRecord:
        # Resume: if this round was already learned from, replay the stored record.
        for record in self.records:
            if record.round_index == round_index:
                return record
        self.ensure_baseline()
        new_examples = self._round_examples(evaluation)
        if new_examples:
            with self.buffer_path.open("a", encoding="utf-8") as handle:
                for example in new_examples:
                    handle.write(json.dumps(example, sort_keys=True) + "\n")
        buffer_size = self._buffer_size()

        trained = False
        probe_accepted = self.best_probe or 0
        promoted = False
        if new_examples and buffer_size:
            adapter_dir = self.root / f"adapter_round_{round_index:02d}"
            candidate_adapter = self._train(adapter_dir)
            trained = candidate_adapter is not None
            if trained:
                probe_accepted = self._probe(adapter_path=candidate_adapter)
                # Gate: at least one verifier-accepted rule AND no regression.
                if probe_accepted >= max(self.best_probe or 0, 1):
                    self.adapter_path = candidate_adapter
                    self.best_probe = probe_accepted
                    promoted = True
                    self._drop_client()
        record = RoundLearningRecord(
            round_index=round_index,
            new_examples=len(new_examples),
            buffer_examples=buffer_size,
            trained=trained,
            probe_accepted=probe_accepted,
            best_probe=self.best_probe or 0,
            promoted=promoted,
            adapter_path=self.adapter_path,
        )
        self.records.append(record)
        self._save_state()
        return record

    def _train(self, adapter_dir: Path) -> str | None:
        if self._trainer_fn is not None:
            return self._trainer_fn(str(self.buffer_path), str(adapter_dir))
        from bcv.graph_lora import train_graph_adapter

        self._drop_client()
        result = train_graph_adapter(
            dataset_path=self.buffer_path,
            output_dir=adapter_dir,
            max_train_examples=self.max_buffer_examples,
            heldout_examples=0,
            epochs=self.epochs,
            max_length=1024,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            mask_prompt_loss=True,
            model_name=self.model_name,
        )
        from bcv.model_zoo import release_cuda

        release_cuda()
        return result.adapter_path if result.accepted else None

    def _round_examples(self, evaluation) -> list[dict[str, object]]:
        """Verifier-approved experience only: accepted rules and mined repairs."""
        from bcv.graph_agent import _proposal_prompt

        proposed_by_name = {proposal.name: proposal.expression for proposal in evaluation.proposed_rules}
        expressions: list[str] = []
        for rule in evaluation.accepted_rules:
            expression = proposed_by_name.get(rule.name) or _predicate_from_description(rule.description)
            if expression:
                expressions.append(expression)
        for repair in evaluation.repair_suggestions:
            expression = str(repair.get("repair_expression", ""))
            if expression:
                expressions.append(expression)
        expressions = list(dict.fromkeys(expressions))[: self.max_rules]
        if not expressions:
            return []
        prompt = _proposal_prompt(self.max_rules)
        target = {
            "rules": [
                {
                    "name": _slug(expression),
                    "description": "Verifier-accepted: degree-descending greedy is optimal on this class.",
                    "expression": expression,
                }
                for expression in expressions
            ]
        }
        examples = [_proposal_example(prompt, target)]
        for expression in expressions[:3]:
            examples.append(
                _proposal_example(
                    _proposal_prompt(1),
                    {
                        "rules": [
                            {
                                "name": _slug(expression),
                                "description": "Verifier-accepted conjecture.",
                                "expression": expression,
                            }
                        ]
                    },
                )
            )
        return examples

    # ------------------------------------------------------------------ state

    def _buffer_size(self) -> int:
        if not self.buffer_path.exists():
            return 0
        return sum(1 for line in self.buffer_path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _save_state(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "adapter_path": self.adapter_path,
                    "best_probe": self.best_probe,
                    "records": [asdict(record) for record in self.records],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.adapter_path = raw.get("adapter_path")
        self.best_probe = raw.get("best_probe")
        self.records = [RoundLearningRecord(**record) for record in raw.get("records", [])]


def _proposal_example(prompt: str, target: dict[str, object]) -> dict[str, object]:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You propose finite graph conjectures for an exact verifier.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(target, sort_keys=True)},
        ]
    }


def _predicate_from_description(description: str) -> str:
    marker = "Predicate:"
    if marker not in description:
        return ""
    return description.split(marker, 1)[1].strip()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return slug[:40] or "rule"
