from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen2TokenizerFast

from bcv.lora_smoke import find_fastcontext_snapshot
from bcv.taste import TasteContext, score_taste_multi_anchor


BASE_MODEL = "microsoft/FastContext-1.0-4B-RL"


@dataclass(frozen=True)
class TasteTrainResult:
    accepted: bool
    adapter_path: str
    dataset_path: str
    examples_seen: int
    epochs: int
    final_loss: float | None
    device: str
    failure: str | None = None


@dataclass(frozen=True)
class TasteEvalRow:
    prompt: str
    base_output: str
    adapter_output: str
    slop_reference: str
    base_score: float
    adapter_score: float
    delta: float
    base_novelty: float
    adapter_novelty: float


@dataclass(frozen=True)
class TasteEvalResult:
    adapter_path: str
    prompts: int
    avg_base_score: float
    avg_adapter_score: float
    avg_delta: float
    rows: tuple[TasteEvalRow, ...]


def train_taste_adapter(
    dataset_path: str | Path = ".bcv_runs/taste_scaled/taste_sft.jsonl",
    output_dir: str | Path = ".bcv_runs/taste_adapter",
    max_examples: int = 32,
    epochs: int = 1,
    max_length: int = 512,
) -> TasteTrainResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        texts = _load_sft_texts(dataset_path)[:max_examples]
        if not texts:
            raise RuntimeError(f"no SFT examples found in {dataset_path}")
        tokenizer = load_fastcontext_tokenizer()
        model = load_base_model_4bit()
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
        final_loss = None
        for _ in range(epochs):
            for text in texts:
                batch = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
                batch = {key: value.to(model.device) for key, value in batch.items()}
                batch["labels"] = batch["input_ids"].clone()
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu())
        adapter_path = output_dir / "fastcontext_taste_lora"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        result = TasteTrainResult(
            accepted=True,
            adapter_path=str(adapter_path),
            dataset_path=str(dataset_path),
            examples_seen=len(texts) * epochs,
            epochs=epochs,
            final_loss=final_loss,
            device=device,
        )
    except Exception as exc:
        result = TasteTrainResult(
            accepted=False,
            adapter_path=str(output_dir / "fastcontext_taste_lora"),
            dataset_path=str(dataset_path),
            examples_seen=0,
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


def evaluate_taste_adapter(
    adapter_path: str | Path,
    output_dir: str | Path = ".bcv_runs/taste_adapter_eval",
    max_new_tokens: int = 96,
    max_prompts: int | None = None,
) -> TasteEvalResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = Path(adapter_path)
    tokenizer = load_fastcontext_tokenizer()

    prompts = heldout_taste_prompts()[:max_prompts] if max_prompts is not None else heldout_taste_prompts()
    base_model = load_base_model_4bit()
    base_outputs = [
        _generate(base_model, tokenizer, _heldout_prompt(row["prompt"]), max_new_tokens=max_new_tokens)
        for row in prompts
    ]
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter_base = load_base_model_4bit()
    adapter_model = PeftModel.from_pretrained(adapter_base, adapter_path)
    adapter_outputs = [
        _generate(adapter_model, tokenizer, _heldout_prompt(row["prompt"]), max_new_tokens=max_new_tokens)
        for row in prompts
    ]

    rows: list[TasteEvalRow] = []
    for spec, base_output, adapter_output in zip(prompts, base_outputs, adapter_outputs):
        context = TasteContext(
            prompt=spec["prompt"],
            domain=spec["domain"],  # type: ignore[arg-type]
            audience=spec["audience"],
            mode=spec["mode"],
            target_novelty=0.48,
        )
        anchors = tuple(spec["slop_references"])
        base_scores = score_taste_multi_anchor(context, base_output, anchors)
        adapter_scores = score_taste_multi_anchor(context, adapter_output, anchors)
        rows.append(
            TasteEvalRow(
                prompt=spec["prompt"],
                base_output=base_output,
                adapter_output=adapter_output,
                slop_reference=anchors[0],
                base_score=base_scores.final_score,
                adapter_score=adapter_scores.final_score,
                delta=adapter_scores.final_score - base_scores.final_score,
                base_novelty=base_scores.novelty,
                adapter_novelty=adapter_scores.novelty,
            )
        )
    result = TasteEvalResult(
        adapter_path=str(adapter_path),
        prompts=len(rows),
        avg_base_score=sum(row.base_score for row in rows) / len(rows),
        avg_adapter_score=sum(row.adapter_score for row in rows) / len(rows),
        avg_delta=sum(row.delta for row in rows) / len(rows),
        rows=tuple(rows),
    )
    (output_dir / "eval_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def load_fastcontext_tokenizer() -> Qwen2TokenizerFast:
    snapshot = find_fastcontext_snapshot()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_file=str(snapshot / "tokenizer.json"),
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model_4bit():
    snapshot = find_fastcontext_snapshot()
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=qconfig,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )


def heldout_taste_prompts() -> list[dict[str, str]]:
    return [
        {
            "prompt": "Explain why people like messy garage bands even when the recordings sound rough.",
            "domain": "explanation",
            "audience": "music critic",
            "mode": "critique",
            "slop_references": (
                "People like garage bands because they sound raw, energetic, and authentic. The rough recording can make the music feel real.",
                "Garage bands are appealing because they have passion, energy, and an authentic sound.",
            ),
        },
        {
            "prompt": "Explain why a simple interface can feel more premium than a feature-heavy one.",
            "domain": "design",
            "audience": "product designer",
            "mode": "design",
            "slop_references": (
                "A simple interface feels premium because it is clean, easy to use, and not cluttered with too many features.",
                "Simple interfaces feel elegant because they are minimal, uncluttered, and user friendly.",
            ),
        },
        {
            "prompt": "Explain why a plot twist fails when it only exists to shock the audience.",
            "domain": "fiction",
            "audience": "screenwriter",
            "mode": "critique",
            "slop_references": (
                "A plot twist fails when it is shocking but does not make sense. It needs to fit the story and characters.",
                "A twist should be surprising but also logical, otherwise it feels random and cheap.",
            ),
        },
        {
            "prompt": "Explain why some jokes get funnier the second time.",
            "domain": "comedy",
            "audience": "writer",
            "mode": "critique",
            "slop_references": (
                "Some jokes get funnier the second time because you understand them better and notice more details.",
                "A joke can improve on repeat because the setup and punchline become clearer.",
            ),
        },
        {
            "prompt": "Explain why a good horror scene can make an ordinary hallway scary.",
            "domain": "explanation",
            "audience": "film writer",
            "mode": "critique",
            "slop_references": (
                "A hallway can be scary because of lighting, music, suspense, and the fear of what might happen.",
                "Horror makes ordinary places scary by building tension and atmosphere.",
            ),
        },
        {
            "prompt": "Explain why a brand slogan can feel fake even when it says the right thing.",
            "domain": "persuasion",
            "audience": "brand strategist",
            "mode": "critique",
            "slop_references": (
                "A brand slogan can feel fake if it sounds generic or does not match what the company actually does.",
                "Slogans need to be authentic, specific, and aligned with the brand.",
            ),
        },
    ]


def _load_sft_texts(dataset_path: Path) -> list[str]:
    texts = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        texts.append(_messages_to_qwen_text(item["messages"]))
    return texts


def _messages_to_qwen_text(messages: list[dict[str, str]]) -> str:
    return "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    )


def _heldout_prompt(prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        "Answer with one compact, concrete paragraph. Avoid generic AI prose. Use a memorable frame, but keep it legible."
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=".bcv_runs/taste_scaled/taste_sft.jsonl")
    parser.add_argument("--output-dir", default=".bcv_runs/taste_adapter")
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-eval-prompts", type=int, default=None)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    train_result = train_taste_adapter(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        epochs=args.epochs,
    )
    payload: dict[str, object] = {"train": asdict(train_result)}
    if args.eval and train_result.accepted:
        payload["eval"] = asdict(
            evaluate_taste_adapter(
                train_result.adapter_path,
                Path(args.output_dir) / "eval",
                max_prompts=args.max_eval_prompts,
            )
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
