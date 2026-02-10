from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel


class Message(BaseModel):
    """Universal message format for LLM conversations."""

    role: str
    content: str


class LLMResponse(BaseModel):
    """Universal response format from LLM providers."""

    content: str
    model: str
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model being used."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate one completion."""

    async def generate_stream(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Default streaming implementation buffers one completion."""
        response = await self.generate(messages, temperature=temperature, max_tokens=max_tokens)
        yield response.content

    @abstractmethod
    def validate(self) -> tuple[bool, str]:
        """Validate provider configuration and connectivity."""


class LLMError(Exception):
    """Base exception for LLM related errors."""


class LLMConfigError(LLMError):
    """Configuration error (missing API key, invalid model, etc.)."""


class LLMConnectionError(LLMError):
    """Connection error (server unreachable, timeout, etc.)."""


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""
