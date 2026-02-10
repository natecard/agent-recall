from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse

from agent_recall.llm.base import (
    LLMConfigError,
    LLMConnectionError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    Message,
)


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI and OpenAI-compatible APIs."""

    KNOWN_PROVIDERS = {
        "openai": {
            "base_url": None,
            "api_key_env": "OPENAI_API_KEY",
            "api_key_required": True,
            "default_model": "gpt-4o",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key_env": None,
            "api_key_required": False,
            "default_model": "llama3.1",
        },
        "vllm": {
            "base_url": "http://localhost:8000/v1",
            "api_key_env": "VLLM_API_KEY",
            "api_key_required": False,
            "default_model": "default",
        },
        "lmstudio": {
            "base_url": "http://localhost:1234/v1",
            "api_key_env": None,
            "api_key_required": False,
            "default_model": "local-model",
        },
        "textgen": {
            "base_url": "http://localhost:5000/v1",
            "api_key_env": None,
            "api_key_required": False,
            "default_model": "default",
        },
        "localai": {
            "base_url": "http://localhost:8080/v1",
            "api_key_env": "LOCALAI_API_KEY",
            "api_key_required": False,
            "default_model": "gpt-3.5-turbo",
        },
    }

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        provider_type: str | None = None,
        timeout: float = 120.0,
    ):
        if provider_type and provider_type in self.KNOWN_PROVIDERS:
            defaults = self.KNOWN_PROVIDERS[provider_type]
            if base_url is None:
                default_base = defaults.get("base_url")
                if isinstance(default_base, str) or default_base is None:
                    base_url = default_base
            if api_key_env is None:
                default_key_env = defaults.get("api_key_env")
                if isinstance(default_key_env, str) or default_key_env is None:
                    api_key_env = default_key_env
            if model == "default":
                default_model = defaults.get("default_model")
                if isinstance(default_model, str):
                    model = default_model

        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._provider_type = provider_type or self._infer_provider_type(base_url)

        if api_key:
            self.api_key = api_key
        elif api_key_env:
            self.api_key = os.environ.get(api_key_env)
        else:
            self.api_key = None

        if not self.api_key and self._is_local_provider():
            self.api_key = "not-needed"

        self._client = None

    @property
    def provider_name(self) -> str:
        type_names = {
            "openai": "OpenAI",
            "ollama": "Ollama (Local)",
            "vllm": "vLLM (Local)",
            "lmstudio": "LMStudio (Local)",
            "textgen": "text-generation-webui (Local)",
            "localai": "LocalAI (Local)",
            "custom": "OpenAI-Compatible (Custom)",
        }
        return type_names.get(self._provider_type, "OpenAI-Compatible")

    @property
    def model_name(self) -> str:
        return self.model

    @staticmethod
    def _infer_provider_type(base_url: str | None) -> str:
        if base_url is None:
            return "openai"

        lowered = base_url.lower()
        if "11434" in lowered:
            return "ollama"
        if "8000" in lowered:
            return "vllm"
        if "1234" in lowered:
            return "lmstudio"
        if "5000" in lowered:
            return "textgen"
        if "8080" in lowered:
            return "localai"
        return "custom"

    def _is_local_provider(self) -> bool:
        return self._provider_type in {
            "ollama",
            "vllm",
            "lmstudio",
            "textgen",
            "localai",
            "custom",
        }

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise LLMConfigError(
                    "openai package not installed. Install with: pip install openai"
                ) from exc

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "not-needed",
                timeout=self.timeout,
            )

        return self._client

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        client = self._get_client()

        payload = [{"role": msg.role, "content": msg.content} for msg in messages]

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]

            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc).lower()
            if "does not support chat" in error:
                return await self._generate_completion_fallback(
                    client=client,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            if "rate" in error and "limit" in error:
                raise LLMRateLimitError(f"Rate limit exceeded: {exc}") from exc
            if "auth" in error or "api_key" in error or "401" in error:
                raise LLMConfigError(f"Authentication error: {exc}") from exc
            if "connect" in error or "timeout" in error or "refused" in error:
                endpoint = self.base_url or "OpenAI API"
                raise LLMConnectionError(
                    f"Connection failed to {endpoint}. Is the server running? Error: {exc}"
                ) from exc
            raise LLMConnectionError(f"API error: {exc}") from exc

    async def generate_stream(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        client = self._get_client()

        payload = [{"role": msg.role, "content": msg.content} for msg in messages]

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:  # noqa: BLE001
            raise LLMConnectionError(f"Streaming error: {exc}") from exc

    def validate(self) -> tuple[bool, str]:
        if self._is_local_provider():
            try:
                parsed = urlparse(self.base_url or "")
                host = parsed.hostname or "localhost"
                port = parsed.port or 80
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                if result != 0:
                    return False, f"Cannot connect to {self.base_url}. Is the server running?"
                return True, f"Connected to {self.provider_name} ({self.model})"
            except Exception as exc:  # noqa: BLE001
                return False, f"Connection check failed: {exc}"

        if not self.api_key or self.api_key == "not-needed":
            return False, "API key not set. Set OPENAI_API_KEY environment variable."

        try:
            from openai import OpenAI

            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            client.chat.completions.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True, f"Connected to {self.provider_name} ({self.model})"
        except Exception as exc:  # noqa: BLE001
            return False, f"Validation failed: {exc}"

    @staticmethod
    def _messages_to_prompt(messages: list[Message]) -> str:
        lines: list[str] = []
        for message in messages:
            role = message.role.upper()
            lines.append(f"{role}: {message.content}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    async def _generate_completion_fallback(
        self,
        client,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        try:
            response = await client.completions.create(
                model=self.model,
                prompt=self._messages_to_prompt(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }

            return LLMResponse(
                content=choice.text or "",
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMConnectionError(f"Completion fallback failed: {exc}") from exc
