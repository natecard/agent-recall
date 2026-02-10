from __future__ import annotations

import os

from agent_recall.llm.base import LLMProvider
from agent_recall.storage.models import LLMConfig


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider from config."""
    match config.provider:
        case "anthropic":
            from agent_recall.llm.anthropic import AnthropicProvider

            return AnthropicProvider(
                model=config.model,
                api_key_env=config.api_key_env or "ANTHROPIC_API_KEY",
            )
        case "google":
            from agent_recall.llm.google import GoogleProvider

            return GoogleProvider(
                model=config.model,
                api_key_env=config.api_key_env or "GOOGLE_API_KEY",
            )
        case "openai":
            from agent_recall.llm.openai_compat import OpenAICompatibleProvider

            return OpenAICompatibleProvider(
                model=config.model,
                base_url=config.base_url,
                api_key_env=config.api_key_env or "OPENAI_API_KEY",
            )
        case "ollama":
            from agent_recall.llm.openai_compat import OpenAICompatibleProvider

            return OpenAICompatibleProvider(
                model=config.model,
                base_url=config.base_url or "http://localhost:11434/v1",
                api_key="not-needed",
            )
        case "openai-compatible":
            from agent_recall.llm.openai_compat import OpenAICompatibleProvider

            if not config.base_url:
                raise ValueError("base_url required for openai-compatible provider")
            return OpenAICompatibleProvider(
                model=config.model,
                base_url=config.base_url,
                api_key=(
                    os.environ.get(config.api_key_env)
                    if config.api_key_env
                    else os.environ.get("OPENAI_API_KEY", "not-needed")
                ),
            )
        case _:
            raise ValueError(f"Unknown LLM provider: {config.provider}")


__all__ = ["create_llm_provider", "LLMProvider"]
