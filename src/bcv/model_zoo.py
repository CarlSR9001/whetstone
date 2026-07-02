"""Model-agnostic 4-bit loaders.

Everything upstream hardcoded the FastContext 4B snapshot. After the 4B foundry run
bugchecked the GPU driver (sustained decode on a fresh Blackwell card), the loop
needs to run on smaller models too. FastContext keeps its bespoke tokenizer path;
any other model id goes through the standard Auto* loaders. All models are served
4-bit NF4 so the same 8 GB budget holds.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

FASTCONTEXT = "microsoft/FastContext-1.0-4B-RL"


def load_tokenizer(model_name: str = FASTCONTEXT):
    if model_name == FASTCONTEXT:
        from bcv.taste_lora import load_fastcontext_tokenizer

        return load_fastcontext_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm_4bit(model_name: str = FASTCONTEXT):
    if model_name == FASTCONTEXT:
        from bcv.taste_lora import load_base_model_4bit

        return load_base_model_4bit()
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=qconfig,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )


def release_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
