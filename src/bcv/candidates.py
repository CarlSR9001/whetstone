"""Systems under exam, as adapters. The gate should grade anything that answers.

Every candidate implements the same contract the research clients already use:
``generate_text(prompt, temperature=0.0) -> str``. Three adapters cover the
product surface:

- OpenAICompatibleCandidate: any /chat/completions endpoint — LM Studio,
  Ollama, vLLM, or a frontier API. Uses stdlib urllib; no SDK dependency.
- CommandCandidate: a subprocess that reads the prompt on stdin and writes the
  answer on stdout. This is the escape hatch: any agent framework that can be
  invoked from a shell can be graded.
- StoredAnswerCandidate: answers loaded from a JSONL file keyed by item id,
  for offline grading, replays, and demos. It cannot answer free prompts.

Exposure rule, load-bearing: grading through a NON-LOCAL endpoint sends private
exam items outside the trust boundary. ``is_external`` tells the caller, and
``grade_and_burn`` enforces the consequence — every item exposed externally is
permanently burned via the bank's burn accounting. Convenience never overrides
the leakage rule.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class CandidateError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleCandidate:
    """Chat-completions client over stdlib urllib.

    ``base_url`` is the API root, e.g. ``http://localhost:1234/v1`` or
    ``https://api.openai.com/v1``.
    """

    base_url: str
    model: str
    api_key: str | None = None
    max_tokens: int = 512
    timeout_seconds: float = 120.0
    system_prompt: str | None = None
    backend = "openai_compatible"

    @property
    def host(self) -> str:
        return urllib.parse.urlparse(self.base_url).hostname or ""

    @property
    def is_external(self) -> bool:
        return self.host not in LOCAL_HOSTS

    @property
    def provider(self) -> str:
        return f"{self.host}/{self.model}"

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise CandidateError(f"endpoint {self.base_url} unreachable: {error}") from error
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise CandidateError(f"malformed completion response: {body}") from error


@dataclass
class CommandCandidate:
    """Any agent invocable from a shell: prompt on stdin, answer on stdout."""

    argv: list[str]
    timeout_seconds: float = 120.0
    backend = "command"
    is_external = False  # runs inside the caller's own trust boundary

    @property
    def provider(self) -> str:
        return " ".join(self.argv)

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        try:
            completed = subprocess.run(
                self.argv,
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise CandidateError(f"command timed out after {self.timeout_seconds}s") from error
        if completed.returncode != 0:
            raise CandidateError(
                f"command exited {completed.returncode}: {completed.stderr.decode('utf-8', 'replace')[:400]}"
            )
        return completed.stdout.decode("utf-8", "replace")


@dataclass
class StoredAnswerCandidate:
    """Offline answers keyed by item id (JSONL rows: {"item_id": ..., "answer": ...}).

    Grading stored answers goes through ``answer_for``; this adapter cannot
    answer arbitrary prompts and says so loudly rather than guessing.
    """

    path: str | Path
    answers: dict[str, object] = field(default_factory=dict)
    backend = "stored"
    is_external = False

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.exists():
            raise CandidateError(f"answer file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self.answers[str(row["item_id"])] = row.get("answer")

    @property
    def provider(self) -> str:
        return f"stored:{Path(self.path).name}"

    def answer_for(self, item_id: str) -> object:
        return self.answers.get(item_id)

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        raise CandidateError(
            "StoredAnswerCandidate holds answers by item id and cannot answer free prompts; "
            "grade it with registry.grade_bank, which routes by item."
        )
