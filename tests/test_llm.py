from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_recall.llm import (
    create_llm_provider,
    ensure_provider_dependency,
    get_available_providers,
)
from agent_recall.llm.base import LLMConfigError, Message
from agent_recall.llm.openai_compat import OpenAICompatibleProvider
from agent_recall.storage.models import LLMConfig


class TestOpenAICompatibleProvider:
    def test_provider_type_inference(self) -> None:
        provider = OpenAICompatibleProvider(
            model="llama3",
            base_url="http://localhost:11434/v1",
        )
        assert provider._provider_type == "ollama"

        provider = OpenAICompatibleProvider(
            model="mistral",
            base_url="http://localhost:8000/v1",
        )
        assert provider._provider_type == "vllm"

        provider = OpenAICompatibleProvider(
            model="local",
            base_url="http://localhost:1234/v1",
        )
        assert provider._provider_type == "lmstudio"

        provider = OpenAICompatibleProvider(
            model="gpt-4o",
            base_url=None,
        )
        assert provider._provider_type == "openai"

    def test_local_provider_no_api_key_needed(self) -> None:
        provider = OpenAICompatibleProvider(
            model="llama3",
            base_url="http://localhost:11434/v1",
            provider_type="ollama",
        )
        assert provider.api_key == "not-needed"

    def test_provider_name(self) -> None:
        provider = OpenAICompatibleProvider(
            model="llama3",
            provider_type="ollama",
        )
        assert "Ollama" in provider.provider_name
        assert "Local" in provider.provider_name


class TestProviderFactory:
    def test_get_available_providers(self) -> None:
        providers = get_available_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "ollama" in providers
        assert "google" in providers

    def test_create_ollama_provider(self) -> None:
        config = LLMConfig(
            provider="ollama",
            model="llama3.1",
        )
        provider = create_llm_provider(config)
        assert provider.provider_name == "Ollama (Local)"
        assert provider.model_name == "llama3.1"

    def test_create_openai_compatible_requires_base_url(self) -> None:
        config = LLMConfig(
            provider="openai-compatible",
            model="my-model",
            base_url=None,
        )

        with pytest.raises(LLMConfigError):
            create_llm_provider(config)

    def test_unknown_provider_raises_error(self) -> None:
        config = LLMConfig(
            provider="unknown-provider",
            model="model",
        )

        with pytest.raises(LLMConfigError) as exc_info:
            create_llm_provider(config)

        assert "unknown-provider" in str(exc_info.value).lower()

    def test_ensure_provider_dependency_reports_missing_without_install(self, monkeypatch) -> None:
        monkeypatch.setattr("agent_recall.llm.importlib.util.find_spec", lambda _name: None)

        ok, message = ensure_provider_dependency("google", auto_install=False)

        assert ok is False
        assert message is not None
        assert "google-generativeai" in message

    def test_ensure_provider_dependency_installs_when_missing(self, monkeypatch) -> None:
        state = {"calls": 0}

        def fake_find_spec(_name: str):
            state["calls"] += 1
            return None if state["calls"] == 1 else object()

        class Completed:
            returncode = 0
            stdout = "ok"
            stderr = ""

        monkeypatch.setattr("agent_recall.llm.importlib.util.find_spec", fake_find_spec)
        monkeypatch.setattr(
            "agent_recall.llm.subprocess.run",
            lambda *_args, **_kwargs: Completed(),
        )

        ok, message = ensure_provider_dependency("google", auto_install=True)

        assert ok is True
        assert message is not None
        assert "Installed provider dependency" in message


class TestLLMGeneration:
    @pytest.mark.asyncio
    async def test_generate_mock(self) -> None:
        provider = OpenAICompatibleProvider(
            model="test-model",
            base_url="http://localhost:8000/v1",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "test-model"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        response = await provider.generate([Message(role="user", content="Hello")])

        assert response.content == "Test response"
        assert response.model == "test-model"
