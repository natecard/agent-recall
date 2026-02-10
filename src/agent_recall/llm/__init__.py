from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agent_recall.llm.base import (
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    Message,
)

if TYPE_CHECKING:
    from agent_recall.storage.models import LLMConfig

__all__ = [
    "LLMProvider",
    "Message",
    "LLMResponse",
    "LLMError",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "create_llm_provider",
    "get_available_providers",
    "validate_provider_config",
]


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider from config."""
    provider = config.provider.lower()

    if provider == "anthropic":
        from agent_recall.llm.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=config.model,
            api_key_env=config.api_key_env or "ANTHROPIC_API_KEY",
        )

    if provider == "google":
        from agent_recall.llm.google import GoogleProvider

        return GoogleProvider(
            model=config.model,
            api_key_env=config.api_key_env or "GOOGLE_API_KEY",
        )

    if provider == "openai":
        from agent_recall.llm.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url,
            api_key_env=config.api_key_env or "OPENAI_API_KEY",
            provider_type="openai",
            timeout=config.timeout,
        )

    if provider == "ollama":
        from agent_recall.llm.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url or "http://localhost:11434/v1",
            provider_type="ollama",
            timeout=config.timeout,
        )

    if provider == "vllm":
        from agent_recall.llm.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url or "http://localhost:8000/v1",
            api_key_env=config.api_key_env,
            provider_type="vllm",
            timeout=config.timeout,
        )

    if provider == "lmstudio":
        from agent_recall.llm.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url or "http://localhost:1234/v1",
            provider_type="lmstudio",
            timeout=config.timeout,
        )

    if provider in {"openai-compatible", "openai_compatible", "custom"}:
        from agent_recall.llm.openai_compat import OpenAICompatibleProvider

        if not config.base_url:
            raise LLMConfigError(
                "base_url is required for openai-compatible provider. "
                "Specify the URL of your OpenAI-compatible server."
            )
        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            provider_type="custom",
            timeout=config.timeout,
        )

    available = ", ".join(get_available_providers())
    raise LLMConfigError(
        f"Unknown LLM provider: '{provider}'. Available providers: {available}"
    )


def get_available_providers() -> list[str]:
    """Return available provider names."""
    return [
        "anthropic",
        "openai",
        "google",
        "ollama",
        "vllm",
        "lmstudio",
        "openai-compatible",
    ]


def validate_provider_config(config: LLMConfig) -> tuple[bool, str]:
    """Validate LLM config structure without calling provider APIs."""
    provider = config.provider.lower()

    if provider not in get_available_providers():
        return False, f"Unknown provider: {provider}"

    cloud_providers = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    if provider in cloud_providers:
        env_var = config.api_key_env or cloud_providers[provider]
        if not os.environ.get(env_var):
            return False, f"API key not set. Set {env_var} environment variable."

    if provider in {"openai-compatible", "custom"} and not config.base_url:
        return False, "base_url is required for openai-compatible provider."

    return True, f"Configuration valid for {provider}"
