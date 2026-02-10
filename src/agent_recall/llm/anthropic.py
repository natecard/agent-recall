from __future__ import annotations

import importlib
import os
from collections.abc import AsyncIterator
from typing import Any

from agent_recall.llm.base import (
    LLMConfigError,
    LLMConnectionError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    Message,
)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    SUPPORTED_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env)
        self._api_key_env = api_key_env
        self._anthropic_module: Any = None
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "Anthropic"

    @property
    def model_name(self) -> str:
        return self.model

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMConfigError(
                    "Anthropic API key not found. "
                    f"Set {self._api_key_env} environment variable or provide api_key."
                )
            try:
                anthropic_module = importlib.import_module("anthropic")
            except ImportError as exc:
                raise LLMConfigError(
                    "anthropic package not installed. Install with: pip install anthropic"
                ) from exc
            self._anthropic_module = anthropic_module
            self._client = anthropic_module.AsyncAnthropic(api_key=self.api_key)

        return self._client

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        client = self._get_client()

        system_content = ""
        conversation: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_content or None,
                messages=conversation,
            )

            content = ""
            if response.content:
                maybe_text = getattr(response.content[0], "text", "")
                if isinstance(maybe_text, str):
                    content = maybe_text

            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc).lower()
            if "rate" in error and "limit" in error:
                raise LLMRateLimitError(f"Anthropic rate limit exceeded: {exc}") from exc
            if "auth" in error or "api_key" in error:
                raise LLMConfigError(f"Anthropic authentication error: {exc}") from exc
            raise LLMConnectionError(f"Anthropic API error: {exc}") from exc

    async def generate_stream(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        client = self._get_client()

        system_content = ""
        conversation: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        try:
            async with client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_content or None,
                messages=conversation,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:  # noqa: BLE001
            raise LLMConnectionError(f"Anthropic streaming error: {exc}") from exc

    def validate(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, f"API key not set. Set {self._api_key_env} environment variable."

        try:
            anthropic_module = self._anthropic_module or importlib.import_module("anthropic")
            client = anthropic_module.Anthropic(api_key=self.api_key)
            client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True, f"Connected to Anthropic ({self.model})"
        except Exception as exc:  # noqa: BLE001
            return False, f"Anthropic validation failed: {exc}"
