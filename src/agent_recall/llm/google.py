from __future__ import annotations

import importlib
import os
from typing import Any, cast

from agent_recall.llm.base import (
    LLMConfigError,
    LLMConnectionError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    Message,
)


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    SUPPORTED_MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemma-3-27b-it",
    ]

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        api_key_env: str = "GOOGLE_API_KEY",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env)
        self._api_key_env = api_key_env
        self._genai_module: Any = None
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "Google"

    @property
    def model_name(self) -> str:
        return self.model

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMConfigError(
                    "Google API key not found. "
                    f"Set {self._api_key_env} environment variable or provide api_key."
                )
            try:
                genai = importlib.import_module("google.genai")
            except ImportError as exc:
                raise LLMConfigError(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                ) from exc

            self._genai_module = genai
            self._client = genai.Client(api_key=self.api_key)

        return self._client

    @staticmethod
    def _collect_contents(messages: list[Message]) -> tuple[list[dict[str, object]], str | None]:
        contents: list[dict[str, object]] = []
        system_messages: list[str] = []

        for msg in messages:
            if msg.role == "system":
                text = msg.content.strip()
                if text:
                    system_messages.append(text)
                continue

            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        system_instruction = "\n\n".join(system_messages) if system_messages else None
        return contents, system_instruction

    async def _generate_content(
        self,
        client: Any,
        *,
        contents: list[dict[str, object]],
        config: dict[str, Any],
    ) -> Any:
        return await client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", None)
        if not isinstance(candidates, list):
            return ""

        chunks: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not isinstance(parts, list):
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    chunks.append(part_text)

        return "\n".join(chunks).strip()

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        client = self._get_client()
        contents, system_instruction = self._collect_contents(messages)

        try:
            generation_config: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system_instruction:
                generation_config["system_instruction"] = system_instruction

            if system_instruction:
                try:
                    response = await self._generate_content(
                        client,
                        contents=contents,
                        config=generation_config,
                    )
                except Exception as exc:  # noqa: BLE001
                    if self._supports_developer_instruction_error(exc):
                        fallback_contents = self._merge_system_instruction_into_contents(
                            contents,
                            system_instruction,
                        )
                        fallback_config = {
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                        }
                        response = await self._generate_content(
                            client,
                            contents=fallback_contents,
                            config=fallback_config,
                        )
                    else:
                        raise
            else:
                response = await self._generate_content(
                    client,
                    contents=contents,
                    config=generation_config,
                )

            return LLMResponse(
                content=self._response_text(response),
                model=self.model,
                usage=None,
                finish_reason=None,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc).lower()
            if any(
                token in error
                for token in ["quota", "rate", "429", "resource exhausted", "too many requests"]
            ):
                raise LLMRateLimitError(f"Google rate limit exceeded: {exc}") from exc
            if any(
                token in error
                for token in ["api key", "api_key", "auth", "permission denied", "unauthorized"]
            ):
                raise LLMConfigError(f"Google authentication error: {exc}") from exc
            raise LLMConnectionError(f"Google API error: {exc}") from exc

    @staticmethod
    def _supports_developer_instruction_error(exc: Exception) -> bool:
        lowered = str(exc).lower()
        return (
            "developer instruction is not enabled" in lowered
            or "system instruction is not enabled" in lowered
            or "system_instruction is not enabled" in lowered
        )

    @staticmethod
    def _merge_system_instruction_into_contents(
        contents: list[dict[str, object]],
        system_instruction: str,
    ) -> list[dict[str, object]]:
        merged: list[dict[str, object]] = []
        for item in contents:
            cloned = dict(item)
            parts = item.get("parts")
            if isinstance(parts, list):
                cloned["parts"] = list(parts)
            merged.append(cloned)
        prefix = f"System guidance:\n{system_instruction.strip()}\n\n"

        for item in merged:
            if item.get("role") != "user":
                continue

            parts = item.get("parts")
            if isinstance(parts, list) and parts:
                first_part = parts[0]
                if isinstance(first_part, dict):
                    first_part_dict = cast(dict[str, Any], first_part)
                    first_text = first_part_dict.get("text")
                else:
                    first_text = None
                if isinstance(first_text, str):
                    patched_first = dict(first_part_dict)
                    patched_first["text"] = f"{prefix}{first_text}"
                    item["parts"] = [patched_first, *parts[1:]]
                else:
                    item["parts"] = [{"text": f"{prefix}{first_part}"}, *parts[1:]]
            else:
                item["parts"] = [{"text": prefix.strip()}]
            return merged

        merged.insert(
            0,
            {
                "role": "user",
                "parts": [{"text": prefix.strip()}],
            },
        )
        return merged

    def validate(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, f"API key not set. Set {self._api_key_env} environment variable."

        try:
            client = self._get_client()
            client.models.generate_content(
                model=self.model,
                contents="Hi",
                config={"temperature": 0.0, "max_output_tokens": 8},
            )
            return True, f"Connected to Google ({self.model})"
        except Exception as exc:  # noqa: BLE001
            return False, f"Google validation failed: {exc}"
