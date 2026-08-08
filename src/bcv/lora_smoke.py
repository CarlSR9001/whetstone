from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen2TokenizerFast


@dataclass(frozen=True)
class LoraSmokeResult:
    accepted: bool
    base_model: str
    adapter_path: str
    loss: float | None
    trainable_parameters: int | None
    total_parameters: int | None
    device: str
    failure: str | None = None


def find_fastcontext_snapshot() -> Path:
    override = os.environ.get("WHETSTONE_FASTCONTEXT_SNAPSHOT", "").strip()
    if override:
        snapshot = Path(override).expanduser()
        if not snapshot.is_dir():
            raise FileNotFoundError(f"FastContext snapshot override is not a directory: {snapshot}")
        return snapshot
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    root = hf_home / "hub" / "models--microsoft--FastContext-1.0-4B-RL" / "snapshots"
    if not root.is_dir():
        raise FileNotFoundError(
            f"no FastContext snapshot root at {root}; set WHETSTONE_FASTCONTEXT_SNAPSHOT"
        )
    snapshots = sorted(path for path in root.iterdir() if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"no FastContext snapshots under {root}")
    return snapshots[-1]


def run_lora_smoke(output_dir: str | Path = ".bcv_runs/lora_smoke") -> LoraSmokeResult:
    return run_lora_smoke_from_dataset(output_dir=output_dir)


def run_lora_smoke_from_dataset(
    output_dir: str | Path = ".bcv_runs/lora_smoke",
    dataset_path: str | Path | None = None,
) -> LoraSmokeResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = "microsoft/FastContext-1.0-4B-RL"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        snapshot = find_fastcontext_snapshot()
        tokenizer = Qwen2TokenizerFast(
            tokenizer_file=str(snapshot / "tokenizer.json"),
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",
        )
        tokenizer.padding_side = "right"
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=True,
            quantization_config=qconfig,
            device_map={"": 0} if device == "cuda" else None,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable_parameters, total_parameters = _parameter_counts(model)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        loss = None
        for text in _training_texts(dataset_path)[:4]:
            batch = tokenizer(text, return_tensors="pt", max_length=768, truncation=True)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            batch["labels"] = batch["input_ids"].clone()
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
        if loss is None:
            raise RuntimeError("no LoRA smoke training examples available")

        adapter_path = output_dir / "fastcontext_document_patch_lora"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        result = LoraSmokeResult(
            accepted=True,
            base_model=base_model,
            adapter_path=str(adapter_path),
            loss=float(loss.detach().cpu()),
            trainable_parameters=trainable_parameters,
            total_parameters=total_parameters,
            device=device,
        )
    except Exception as exc:
        result = LoraSmokeResult(
            accepted=False,
            base_model=base_model,
            adapter_path=str(output_dir / "fastcontext_document_patch_lora"),
            loss=None,
            trainable_parameters=None,
            total_parameters=None,
            device=device,
            failure=f"{type(exc).__name__}: {exc}",
        )

    (output_dir / "result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _parameter_counts(model) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def _training_texts(dataset_path: str | Path | None = None) -> list[str]:
    dataset_path = Path(dataset_path) if dataset_path is not None else Path(".bcv_runs") / "all" / "datasets" / "controller_sft.jsonl"
    if dataset_path.exists():
        texts: list[str] = []
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(_messages_to_qwen_text(item["messages"]))
        if texts:
            return texts

    return [
        (
            "<|im_start|>user\n"
            "Return a JSON patch that edits only the Scope section and preserves payment terms, dates, and citations.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{\"mode\":\"patch\",\"operations\":[{\"target_heading\":\"Scope\",\"find\":\"Northstar Labs will deliver the analytics dashboard described in Exhibit A.\","
            "\"replace\":\"Northstar Labs will deliver the analytics dashboard and a weekly deployment summary described in Exhibit A.\"}]}"
            "<|im_end|>"
        )
    ]


def _messages_to_qwen_text(messages: list[dict[str, str]]) -> str:
    return "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".bcv_runs/lora_smoke")
    parser.add_argument("--dataset-path", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            asdict(run_lora_smoke_from_dataset(args.output_dir, args.dataset_path)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
