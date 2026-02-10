from __future__ import annotations

import os
from typing import Any, cast

from agent_recall.llm.base import LLMProvider, LLMResponse, Message


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI and any OpenAI-compatible API provider."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env) or "not-needed"

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ValueError(
                "Install the 'openai' package to use OpenAI-compatible provider"
            ) from exc

        self.client = AsyncOpenAI(base_url=base_url, api_key=self.api_key)

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        message_payload = [{"role": m.role, "content": m.content} for m in messages]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(Any, message_payload),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        content = response.choices[0].message.content or ""
        return LLMResponse(content=content, model=response.model, usage=usage)
