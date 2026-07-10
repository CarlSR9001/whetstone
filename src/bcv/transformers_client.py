"""A LocalModelClient-compatible proposer backed by the cached FastContext 4B.

The foundry loop previously required a running Ollama/LM Studio endpoint. This
client runs the same 4-bit FastContext snapshot the LoRA experiments use, so the
whole propose -> verify -> stress -> ledger loop is self-contained on one machine:
GPU proposes, CPU verifier kills, ledger stores the blood.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

from bcv.local_model import LocalModelError


class TransformersLocalClient:
    backend = "transformers"

    def __init__(
        self,
        adapter_path: str | None = None,
        max_new_tokens: int = 512,
        model_name: str | None = None,
    ) -> None:
        from bcv.model_zoo import FASTCONTEXT

        self.model = model_name or FASTCONTEXT
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.provider = f"transformers/{self.model}"
        self.is_external = False
        self.trust_zone = "local_process"
        self.infrastructure = "local_gpu"
        self.quantization = (
            "checkpoint_config"
            if "bnb-4bit" in self.model.lower() or self.model.lower().endswith("-4bit")
            else "nf4_runtime"
        )
        self.model_revision: str | None = None
        self.adapter_sha256 = self._adapter_sha256(adapter_path)
        self._model = None
        self._tokenizer = None

    @staticmethod
    def _adapter_sha256(adapter_path: str | None) -> str | None:
        if not adapter_path:
            return None
        root = Path(adapter_path)
        model_file = root / "adapter_model.safetensors"
        config_file = root / "adapter_config.json"
        digest = hashlib.sha256()
        found = False
        for path in (config_file, model_file):
            if not path.exists():
                continue
            found = True
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest() if found else None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from bcv.model_zoo import load_causal_lm_4bit, load_tokenizer

        self._tokenizer = load_tokenizer(self.model)
        model = load_causal_lm_4bit(self.model)
        self.model_revision = getattr(model.config, "_commit_hash", None)
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model = model

    def unload(self) -> None:
        """Free GPU memory so a trainer can use the card between rounds."""
        from bcv.model_zoo import release_cuda

        self._model = None
        self._tokenizer = None
        release_cuda()

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        import torch

        self._ensure_loaded()
        messages = [
            {
                "role": "system",
                "content": "You are a careful combinatorics research assistant. Follow the output format exactly.",
            },
            {"role": "user", "content": prompt},
        ]
        if getattr(self._tokenizer, "chat_template", None):
            try:
                # Qwen3-style models: disable thinking or the token budget burns on it.
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            text = (
                f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(
            self._model.device
        )
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def score_nll(self, prefix: str, continuation: str) -> float:
        """Mean per-token NLL of `continuation` given `prefix` (one prefill, no decode)."""
        import torch

        self._ensure_loaded()
        prefix_ids = self._tokenizer(prefix, return_tensors="pt")["input_ids"]
        full_ids = self._tokenizer(prefix + continuation, return_tensors="pt")["input_ids"]
        full_ids = full_ids.to(self._model.device)
        prefix_length = prefix_ids.shape[1]
        continuation_length = full_ids.shape[1] - prefix_length
        if continuation_length <= 0:
            return 0.0
        with torch.no_grad():
            logits = self._model(full_ids).logits
        targets = full_ids[0, prefix_length:]
        predictions = logits[0, prefix_length - 1 : -1]
        loss = torch.nn.functional.cross_entropy(predictions.float(), targets, reduction="mean")
        return float(loss)

    def generate_json(self, prompt: str, temperature: float = 0.0) -> dict[str, Any] | list[Any]:
        last_text = ""
        for attempt in range(3):
            attempt_temperature = temperature if attempt == 0 else max(temperature, 0.3 * attempt)
            last_text = self.generate_text(prompt, temperature=attempt_temperature)
            parsed = extract_json(last_text)
            if parsed is not None:
                return parsed
        raise LocalModelError(f"model did not return valid JSON: {last_text[:500]}")


class RoutedAdapterCandidate(TransformersLocalClient):
    """Use a repair adapter only for graph-repair prompts, base weights elsewhere.

    This is a single loaded PEFT model with an explicit task router, not a
    post-hoc merge of scores. It prevents a specialized adapter from perturbing
    unrelated code behavior while preserving its graph-repair capability.
    """

    routing_policy = "repair_prompt_adapter_else_base"

    def __init__(self, adapter_path: str, max_new_tokens: int = 512, model_name: str | None = None) -> None:
        super().__init__(adapter_path=adapter_path, max_new_tokens=max_new_tokens, model_name=model_name)
        self.backend = "transformers_routed"

    @staticmethod
    def uses_adapter(prompt: str) -> bool:
        return prompt.lstrip().startswith("Repair a rejected conjecture.")

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        self._ensure_loaded()
        if self.uses_adapter(prompt):
            return super().generate_text(prompt, temperature=temperature)
        with self._model.disable_adapter():
            return super().generate_text(prompt, temperature=temperature)


def extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates.extend(block.strip() for block in fenced)
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None
