from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from bcv.graph_agent import _observations_for, compile_feature_expression
from bcv.model_zoo import FASTCONTEXT, load_causal_lm_4bit, load_tokenizer
from bcv.taste_lora import BASE_MODEL


@dataclass(frozen=True)
class GraphAdapterTrainResult:
    accepted: bool
    adapter_path: str
    dataset_path: str
    train_examples: int
    heldout_examples: int
    epochs: int
    final_loss: float | None
    device: str
    failure: str | None = None
    skipped_steps: int = 0


@dataclass(frozen=True)
class GraphAdapterEvalRow:
    prompt: str
    original_expression: str | None
    expected_expression: str
    base_output: str
    adapter_output: str
    base_expression: str | None
    adapter_expression: str | None
    base_parseable: bool
    adapter_parseable: bool
    base_verified: bool
    adapter_verified: bool
    base_refines_original: bool
    adapter_refines_original: bool
    adapter_support: int
    original_true_positives: int


@dataclass(frozen=True)
class GraphAdapterEvalResult:
    adapter_path: str
    eval_examples: int
    base_parseable: int
    adapter_parseable: int
    base_verified: int
    adapter_verified: int
    base_refined: int
    adapter_refined: int
    distinct_adapter_expressions: int
    mean_support_retention: float | None
    rows: tuple[GraphAdapterEvalRow, ...]


@dataclass(frozen=True)
class GraphAdapterRunResult:
    train: GraphAdapterTrainResult
    eval: GraphAdapterEvalResult | None


def train_graph_adapter(
    dataset_path: str | Path,
    output_dir: str | Path = ".bcv_runs/graph_lora",
    max_train_examples: int = 128,
    heldout_examples: int = 4,
    epochs: int = 2,
    max_length: int = 768,
    lora_r: int = 4,
    lora_alpha: int = 8,
    heldout_path: str | Path | None = None,
    mask_prompt_loss: bool = True,
    model_name: str = FASTCONTEXT,
) -> GraphAdapterTrainResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_path = output_dir / "fastcontext_graph_repair_lora"
    try:
        examples = _load_examples(dataset_path)
        if heldout_path is not None or heldout_examples <= 0:
            train_examples = examples
        else:
            train_examples = examples[:-heldout_examples] if len(examples) > heldout_examples else examples
        train_examples = train_examples[:max_train_examples]
        if not train_examples:
            raise RuntimeError(f"no training examples found in {dataset_path}")

        tokenizer = load_tokenizer(model_name)
        model = load_causal_lm_4bit(model_name)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
        final_loss = None
        skipped_examples = 0
        for _ in range(epochs):
            for example in train_examples:
                text = _messages_to_text(example["messages"])
                batch = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
                batch = {key: value.to(model.device) for key, value in batch.items()}
                labels = batch["input_ids"].clone()
                if mask_prompt_loss:
                    # Supervise only the assistant completion. Without this, long
                    # evidence prompts dominate the loss and the adapter learns to
                    # regurgitate prompts instead of producing repairs.
                    prompt_length = tokenizer(
                        _prompt_from_example(example),
                        max_length=max_length,
                        truncation=True,
                    )["input_ids"]
                    labels[0, : min(len(prompt_length), labels.shape[1])] = -100
                if not (labels != -100).any():
                    # Truncation swallowed the completion; training on this example
                    # would yield a NaN loss and teach nothing.
                    skipped_examples += 1
                    continue
                batch["labels"] = labels
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    skipped_examples += 1
                    continue
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu())

        if final_loss is None:
            raise RuntimeError(
                f"every training step was skipped ({skipped_examples} skips); "
                f"prompts likely exceed max_length={max_length}"
            )
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        result = GraphAdapterTrainResult(
            accepted=True,
            adapter_path=str(adapter_path),
            dataset_path=str(dataset_path),
            train_examples=len(train_examples),
            heldout_examples=(
                _count_examples(heldout_path)
                if heldout_path is not None
                else max(0, min(heldout_examples, len(examples) - len(train_examples)))
            ),
            epochs=epochs,
            final_loss=final_loss,
            device=device,
            skipped_steps=skipped_examples,
        )
    except Exception as exc:
        result = GraphAdapterTrainResult(
            accepted=False,
            adapter_path=str(adapter_path),
            dataset_path=str(dataset_path),
            train_examples=0,
            heldout_examples=0,
            epochs=epochs,
            final_loss=None,
            device=device,
            failure=f"{type(exc).__name__}: {exc}",
        )
    (output_dir / "train_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def evaluate_graph_adapter(
    adapter_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path = ".bcv_runs/graph_lora_eval",
    heldout_examples: int = 4,
    max_new_tokens: int = 96,
    max_n: int = 6,
    heldout_path: str | Path | None = None,
    model_name: str = FASTCONTEXT,
) -> GraphAdapterEvalResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = Path(adapter_path)
    if heldout_path is not None:
        eval_examples = _load_examples(Path(heldout_path))
    else:
        examples = _load_examples(Path(dataset_path))
        eval_examples = examples[-heldout_examples:] if heldout_examples else examples
    tokenizer = load_tokenizer(model_name)

    base_model = load_causal_lm_4bit(model_name)
    base_outputs = [_generate(base_model, tokenizer, _prompt_from_example(example), max_new_tokens) for example in eval_examples]
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter_base = load_causal_lm_4bit(model_name)
    adapter_model = PeftModel.from_pretrained(adapter_base, adapter_path)
    adapter_outputs = [
        _generate(adapter_model, tokenizer, _prompt_from_example(example), max_new_tokens) for example in eval_examples
    ]
    del adapter_model
    del adapter_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows: list[GraphAdapterEvalRow] = []
    for example, base_output, adapter_output in zip(eval_examples, base_outputs, adapter_outputs):
        expected = _expected_expression(example)
        original = _original_expression(example)
        base_expression = _extract_expression(base_output)
        adapter_expression = _extract_expression(adapter_output)
        base_analysis = _analyze_expression(base_expression, original, max_n)
        adapter_analysis = _analyze_expression(adapter_expression, original, max_n)
        rows.append(
            GraphAdapterEvalRow(
                prompt=_prompt_from_example(example),
                original_expression=original,
                expected_expression=expected,
                base_output=base_output,
                adapter_output=adapter_output,
                base_expression=base_expression,
                adapter_expression=adapter_expression,
                base_parseable=base_analysis.parseable,
                adapter_parseable=adapter_analysis.parseable,
                base_verified=base_analysis.verified,
                adapter_verified=adapter_analysis.verified,
                base_refines_original=base_analysis.refines_original,
                adapter_refines_original=adapter_analysis.refines_original,
                adapter_support=adapter_analysis.support,
                original_true_positives=adapter_analysis.original_true_positives,
            )
        )
    retentions = [
        row.adapter_support / row.original_true_positives
        for row in rows
        if row.adapter_refines_original and row.original_true_positives
    ]
    result = GraphAdapterEvalResult(
        adapter_path=str(adapter_path),
        eval_examples=len(rows),
        base_parseable=sum(1 for row in rows if row.base_parseable),
        adapter_parseable=sum(1 for row in rows if row.adapter_parseable),
        base_verified=sum(1 for row in rows if row.base_verified),
        adapter_verified=sum(1 for row in rows if row.adapter_verified),
        base_refined=sum(1 for row in rows if row.base_refines_original),
        adapter_refined=sum(1 for row in rows if row.adapter_refines_original),
        distinct_adapter_expressions=len({row.adapter_expression for row in rows if row.adapter_expression}),
        mean_support_retention=(sum(retentions) / len(retentions)) if retentions else None,
        rows=tuple(rows),
    )
    (output_dir / "eval_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def run_graph_adapter_train_eval(
    dataset_path: str | Path,
    output_dir: str | Path = ".bcv_runs/graph_lora",
    max_train_examples: int = 128,
    heldout_examples: int = 4,
    epochs: int = 2,
    max_n: int = 6,
    max_length: int = 512,
    lora_r: int = 4,
    lora_alpha: int = 8,
    heldout_path: str | Path | None = None,
    mask_prompt_loss: bool = True,
) -> GraphAdapterRunResult:
    output_dir = Path(output_dir)
    train = train_graph_adapter(
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_train_examples=max_train_examples,
        heldout_examples=heldout_examples,
        epochs=epochs,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        heldout_path=heldout_path,
        mask_prompt_loss=mask_prompt_loss,
    )
    evaluation = (
        evaluate_graph_adapter(
            adapter_path=train.adapter_path,
            dataset_path=dataset_path,
            output_dir=output_dir / "eval",
            heldout_examples=heldout_examples,
            max_n=max_n,
            heldout_path=heldout_path,
        )
        if train.accepted
        else None
    )
    result = GraphAdapterRunResult(train=train, eval=evaluation)
    (output_dir / "run_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _load_examples(dataset_path: Path) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        messages = item.get("messages")
        if isinstance(messages, list) and len(messages) >= 3:
            examples.append(item)
    return examples


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    return "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    )


def _prompt_from_example(example: dict[str, object]) -> str:
    messages = example["messages"]
    assert isinstance(messages, list)
    prompt_messages = [message for message in messages if isinstance(message, dict)][:-1]
    return _messages_to_text(prompt_messages) + "<|im_start|>assistant\n"


def _expected_expression(example: dict[str, object]) -> str:
    messages = example["messages"]
    assert isinstance(messages, list)
    assistant = messages[-1]
    if isinstance(assistant, dict):
        return _extract_expression(str(assistant.get("content", ""))) or ""
    return ""


def _extract_expression(output: str) -> str | None:
    for candidate in _json_candidates(output):
        if isinstance(candidate, dict):
            expression = candidate.get("repair_expression")
            if isinstance(expression, str) and expression.strip():
                return expression.strip()
    backticked = re.findall(r"`([^`]+)`", output)
    for candidate in reversed(backticked):
        if _looks_like_expression(candidate):
            return candidate.strip()
    match = re.search(r"repair_expression['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", output)
    if match:
        return match.group(1).strip()
    return None


def _json_candidates(output: str) -> list[object]:
    candidates: list[object] = []
    try:
        candidates.append(json.loads(output))
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{[^{}]*\}", output):
        try:
            candidates.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return candidates


def _looks_like_expression(candidate: str) -> bool:
    return any(token in candidate for token in ("is_", "max_degree", "m ", "n ", "degree"))


@dataclass(frozen=True)
class ExpressionAnalysis:
    parseable: bool
    verified: bool
    refines_original: bool
    support: int
    original_true_positives: int


def _analyze_expression(expression: str | None, original_expression: str | None, max_n: int) -> ExpressionAnalysis:
    failed = ExpressionAnalysis(False, False, False, 0, 0)
    if not expression:
        return failed
    observations = _observations_for(max_n)
    try:
        predicate = compile_feature_expression(expression)
        matches = [obs for obs in observations if predicate(obs)]
    except Exception:
        return failed
    false_positives = [obs for obs in matches if not obs.greedy_is_optimal]
    verified = bool(matches) and not false_positives
    refines = False
    original_true_positives = 0
    if original_expression:
        try:
            original_predicate = compile_feature_expression(original_expression)
            original_true_positives = sum(
                1 for obs in observations if original_predicate(obs) and obs.greedy_is_optimal
            )
            refines = verified and all(original_predicate(obs) for obs in matches)
        except Exception:
            refines = False
    return ExpressionAnalysis(
        parseable=True,
        verified=verified,
        refines_original=refines,
        support=len(matches),
        original_true_positives=original_true_positives,
    )


def _verify_expression(expression: str | None, max_n: int) -> tuple[bool, bool]:
    analysis = _analyze_expression(expression, None, max_n)
    return analysis.parseable, analysis.verified


def _original_expression(example: dict[str, object]) -> str | None:
    messages = example.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            try:
                payload = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                original = payload.get("original_expression")
                if isinstance(original, str) and original.strip():
                    return original.strip()
    return None


def _count_examples(path: str | Path) -> int:
    return len(_load_examples(Path(path)))


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and verify-evaluate a graph-repair QLoRA adapter.")
    parser.add_argument("--dataset-path", default=".bcv_runs/research_foundry_model/foundry_sft.jsonl")
    parser.add_argument("--output-dir", default=".bcv_runs/graph_lora")
    parser.add_argument("--max-train-examples", type=int, default=128)
    parser.add_argument("--heldout-examples", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--heldout-path", default=None)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--no-mask-prompt-loss",
        action="store_true",
        help="Train on the full sequence including the prompt (legacy behavior).",
    )
    args = parser.parse_args()
    if args.eval_only:
        adapter_path = args.adapter_path or str(Path(args.output_dir) / "fastcontext_graph_repair_lora")
        payload = {
            "eval": asdict(
                evaluate_graph_adapter(
                    adapter_path=adapter_path,
                    dataset_path=args.dataset_path,
                    output_dir=Path(args.output_dir) / "eval",
                    heldout_examples=args.heldout_examples,
                    max_n=args.max_n,
                    heldout_path=args.heldout_path,
                )
            )
        }
    elif args.train_only:
        payload = {
            "train": asdict(
                train_graph_adapter(
                    dataset_path=args.dataset_path,
                    output_dir=args.output_dir,
                    max_train_examples=args.max_train_examples,
                    heldout_examples=args.heldout_examples,
                    epochs=args.epochs,
                    max_length=args.max_length,
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    heldout_path=args.heldout_path,
                    mask_prompt_loss=not args.no_mask_prompt_loss,
                )
            )
        }
    else:
        payload = asdict(
            run_graph_adapter_train_eval(
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                max_train_examples=args.max_train_examples,
                heldout_examples=args.heldout_examples,
                epochs=args.epochs,
                max_n=args.max_n,
                max_length=args.max_length,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                heldout_path=args.heldout_path,
                mask_prompt_loss=not args.no_mask_prompt_loss,
            )
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
