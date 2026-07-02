"""A LocalModelClient-compatible proposer backed by the cached FastContext 4B.

The foundry loop previously required a running Ollama/LM Studio endpoint. This
client runs the same 4-bit FastContext snapshot the LoRA experiments use, so the
whole propose -> verify -> stress -> ledger loop is self-contained on one machine:
GPU proposes, CPU verifier kills, ledger stores the blood.
"""

from __future__ import annotations

import json
import re
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
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from bcv.model_zoo import load_causal_lm_4bit, load_tokenizer

        self._tokenizer = load_tokenizer(self.model)
        model = load_causal_lm_4bit(self.model)
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
