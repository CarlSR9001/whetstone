from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal


BackendName = Literal["ollama", "lmstudio"]


class LocalModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalModelResponse:
    backend: BackendName
    model: str
    text: str
    raw: dict[str, Any]


class LocalModelClient:
    def __init__(
        self,
        backend: BackendName,
        model: str,
        base_url: str | None = None,
        timeout_s: int = 120,
    ) -> None:
        self.backend = backend
        self.model = model
        self.base_url = base_url or (
            "http://localhost:11434" if backend == "ollama" else "http://localhost:1234/v1"
        )
        self.timeout_s = timeout_s

    def generate_json(self, prompt: str, temperature: float = 0.0) -> dict[str, Any]:
        response = self.generate(prompt, temperature=temperature, json_mode=True)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise LocalModelError(f"model did not return valid JSON: {response.text[:500]}") from exc

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LocalModelResponse:
        if self.backend == "ollama":
            return self._generate_ollama(prompt, temperature, json_mode)
        return self._generate_lmstudio(prompt, temperature, json_mode)

    def _generate_ollama(
        self,
        prompt: str,
        temperature: float,
        json_mode: bool,
    ) -> LocalModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        raw = _post_json(f"{self.base_url}/api/generate", payload, self.timeout_s)
        return LocalModelResponse(
            backend="ollama",
            model=self.model,
            text=raw.get("response", ""),
            raw=raw,
        )

    def _generate_lmstudio(
        self,
        prompt: str,
        temperature: float,
        json_mode: bool,
    ) -> LocalModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        raw = _post_json(f"{self.base_url}/chat/completions", payload, self.timeout_s)
        choices = raw.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        return LocalModelResponse(
            backend="lmstudio",
            model=self.model,
            text=text,
            raw=raw,
        )


def list_ollama_models(base_url: str = "http://localhost:11434", timeout_s: int = 10) -> list[str]:
    raw = _get_json(f"{base_url}/api/tags", timeout_s)
    return [item["name"] for item in raw.get("models", [])]


def list_lmstudio_models(base_url: str = "http://localhost:1234/v1", timeout_s: int = 10) -> list[str]:
    raw = _get_json(f"{base_url}/models", timeout_s)
    return [item["id"] for item in raw.get("data", [])]


def auto_local_client() -> LocalModelClient:
    ollama_models = list_ollama_models()
    if ollama_models:
        preferred = "qwen3:8b" if "qwen3:8b" in ollama_models else ollama_models[0]
        return LocalModelClient("ollama", preferred)

    lmstudio_models = list_lmstudio_models()
    if lmstudio_models:
        return LocalModelClient("lmstudio", lmstudio_models[0])

    raise LocalModelError("no local Ollama or LM Studio model endpoint is available")


def _post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LocalModelError(f"local model request failed: {url}: {exc}") from exc


def _get_json(url: str, timeout_s: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LocalModelError(f"local model probe failed: {url}: {exc}") from exc

