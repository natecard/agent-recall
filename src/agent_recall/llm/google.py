from __future__ import annotations

import importlib
import os
from typing import Any

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
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro",
    ]

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
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
                genai = importlib.import_module("google.generativeai")
            except ImportError as exc:
                raise LLMConfigError(
                    "google-generativeai package not installed. "
                    "Install with: pip install google-generativeai"
                ) from exc

            genai.configure(api_key=self.api_key)
            self._genai_module = genai
            self._client = genai.GenerativeModel(self.model)

        return self._client

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        client = self._get_client()

        contents: list[dict[str, object]] = []
        system_instruction: str | None = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [msg.content]})

        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            if system_instruction:
                genai = self._genai_module or importlib.import_module("google.generativeai")
                model = genai.GenerativeModel(
                    self.model,
                    system_instruction=system_instruction,
                )
            else:
                model = client

            response = await model.generate_content_async(
                contents,
                generation_config=generation_config,
            )

            return LLMResponse(
                content=response.text,
                model=self.model,
                usage=None,
                finish_reason=None,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc).lower()
            if "quota" in error or "rate" in error:
                raise LLMRateLimitError(f"Google rate limit exceeded: {exc}") from exc
            if "api_key" in error or "auth" in error:
                raise LLMConfigError(f"Google authentication error: {exc}") from exc
            raise LLMConnectionError(f"Google API error: {exc}") from exc

    def validate(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, f"API key not set. Set {self._api_key_env} environment variable."

        try:
            genai = importlib.import_module("google.generativeai")
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            model.generate_content("Hi")
            return True, f"Connected to Google ({self.model})"
        except Exception as exc:  # noqa: BLE001
            return False, f"Google validation failed: {exc}"
