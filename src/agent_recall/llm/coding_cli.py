from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from agent_recall.llm.base import (
    LLMConfigError,
    LLMConnectionError,
    LLMProvider,
    LLMResponse,
    Message,
)

if TYPE_CHECKING:
    pass

VALID_CODING_CLIS = ("claude-code", "codex", "opencode")

_JSON_FENCE_RE = re.compile(r"```(?:json)?|```", flags=re.IGNORECASE)


class CodingCLIProvider(LLMProvider):
    """LLM provider that uses a coding CLI agent for synthesis."""

    def __init__(
        self,
        coding_cli: str,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        normalized = (coding_cli or "").strip().lower()
        if normalized not in VALID_CODING_CLIS:
            raise LLMConfigError(
                f"Unknown coding CLI: '{coding_cli}'. Valid options: {', '.join(VALID_CODING_CLIS)}"
            )
        if shutil.which(normalized) is None:
            raise LLMConfigError(f"Coding CLI '{normalized}' not found on PATH")

        self._coding_cli = normalized
        self._model = model
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return f"coding-cli:{self._coding_cli}"

    @property
    def model_name(self) -> str:
        return self._model or f"{self._coding_cli}/default"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        prompt = messages[-1].content if messages else ""

        try:
            result = await asyncio.to_thread(
                self._run_cli,
                prompt,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMConnectionError(f"Coding CLI timed out after {self._timeout}s") from exc
        except FileNotFoundError as exc:
            raise LLMConnectionError(f"Coding CLI '{self._coding_cli}' not found") from exc
        except OSError as exc:
            raise LLMConnectionError(f"Coding CLI error: {exc}") from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise LLMConnectionError(stderr or f"Coding CLI exited with code {result.returncode}")

        content = (result.stdout or "").strip()
        return LLMResponse(content=content, model=self.model_name)

    def _run_cli(
        self,
        prompt: str,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if self._coding_cli == "opencode":
            cmd = ["opencode", "--print"]
            if self._model:
                cmd.extend(["--model", self._model])
            cmd.append(prompt)
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        if self._coding_cli == "claude-code":
            cmd = ["claude"]
            if self._model:
                cmd.extend(["--model", self._model])
            cmd.extend(["--print", prompt])
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        if self._coding_cli == "codex":
            cmd = ["codex"]
            if self._model:
                cmd.extend(["--model", self._model])
            cmd.append(prompt)
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        raise LLMConfigError(f"Unsupported coding CLI: {self._coding_cli}")

    def validate(self) -> tuple[bool, str]:
        if shutil.which(self._coding_cli) is None:
            return False, f"Coding CLI '{self._coding_cli}' not found on PATH"
        return True, f"Coding CLI '{self._coding_cli}' available"


def create_coding_cli_provider(
    coding_cli: str,
    model: str | None = None,
    timeout: float = 120.0,
) -> CodingCLIProvider:
    """Factory function to create a CodingCLIProvider with validation."""
    return CodingCLIProvider(coding_cli=coding_cli, model=model, timeout=timeout)


def strip_json_fences(content: str) -> str:
    """Strip markdown JSON fences from LLM output."""
    return _JSON_FENCE_RE.sub("", content).strip()


def validate_compaction_output(content: str, expected_keys: set[str]) -> tuple[bool, str]:
    """Validate that compaction output is valid JSON with expected structure.

    Returns (is_valid, parsed_json_or_error_message).
    """
    cleaned = strip_json_fences(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON: {exc}"

    if not isinstance(parsed, dict):
        return False, "Output must be a JSON object"

    missing = expected_keys - set(parsed.keys())
    if missing:
        return False, f"Missing required keys: {missing}"

    return True, json.dumps(parsed)
