from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_recall.llm.base import LLMConnectionError, Message
from agent_recall.llm.google import GoogleProvider


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeAsyncModels:
    def __init__(self, fail_with_system_instruction: bool = False):
        self.fail_with_system_instruction = fail_with_system_instruction
        self.calls: list[dict[str, Any]] = []

    async def generate_content(
        self,
        *,
        model: str,
        contents: list[dict[str, object]],
        config: dict[str, Any] | None = None,
    ) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config or {}})
        if self.fail_with_system_instruction and "system_instruction" in (config or {}):
            raise RuntimeError("400 Developer instruction is not enabled for models/gemma-3-27b-it")
        if self.fail_with_system_instruction:
            return _FakeResponse("ok-fallback")
        raise RuntimeError("backend unavailable")


class _FakeSyncModels:
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: dict[str, Any] | None = None,
    ) -> _FakeResponse:
        _ = (model, contents, config)
        return _FakeResponse("ok")


class _FakeClient:
    def __init__(self, async_models: _FakeAsyncModels):
        self.aio = SimpleNamespace(models=async_models)
        self.models = _FakeSyncModels()


def _patch_genai_module(monkeypatch, async_models: _FakeAsyncModels) -> None:
    fake_module = SimpleNamespace(Client=lambda api_key: _FakeClient(async_models))
    monkeypatch.setattr(
        "agent_recall.llm.google.importlib.import_module",
        lambda _name: fake_module,
    )


@pytest.mark.asyncio
async def test_google_provider_falls_back_when_system_instruction_not_supported(
    monkeypatch,
) -> None:
    async_models = _FakeAsyncModels(fail_with_system_instruction=True)
    _patch_genai_module(monkeypatch, async_models)

    provider = GoogleProvider(model="gemma-3-27b-it", api_key="test-key")
    result = await provider.generate(
        [
            Message(role="system", content="Only return JSON arrays."),
            Message(role="user", content="Analyze this transcript."),
        ]
    )

    assert result.content == "ok-fallback"
    assert len(async_models.calls) == 2
    assert "system_instruction" in async_models.calls[0]["config"]
    assert "system_instruction" not in async_models.calls[1]["config"]

    retried_contents = async_models.calls[1]["contents"]
    assert retried_contents[0]["role"] == "user"
    first_part = retried_contents[0]["parts"][0]
    assert isinstance(first_part, dict)
    assert str(first_part["text"]).startswith(
        "System guidance:\nOnly return JSON arrays.\n\nAnalyze this transcript."
    )


@pytest.mark.asyncio
async def test_google_provider_raises_connection_error_for_non_fallback_failures(
    monkeypatch,
) -> None:
    async_models = _FakeAsyncModels(fail_with_system_instruction=False)
    _patch_genai_module(monkeypatch, async_models)

    provider = GoogleProvider(model="gemma-3-27b-it", api_key="test-key")
    with pytest.raises(LLMConnectionError):
        await provider.generate(
            [
                Message(role="system", content="Only return JSON arrays."),
                Message(role="user", content="Analyze this transcript."),
            ]
        )
