from __future__ import annotations

import os
from typing import Any, cast

from agent_recall.llm.base import LLMProvider, LLMResponse, Message


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY"):
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key: set {api_key_env}")

        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ValueError("Install the 'anthropic' package to use Anthropic provider") from exc

        self.client = AsyncAnthropic(api_key=self.api_key)

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        system: str | None = None
        user_messages: list[dict[str, str]] = []

        for message in messages:
            if message.role == "system":
                system = message.content
            else:
                user_messages.append({"role": message.role, "content": message.content})

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=cast(Any, user_messages),
        )

        first_content = ""
        if response.content:
            maybe_text = getattr(response.content[0], "text", None)
            if isinstance(maybe_text, str):
                first_content = maybe_text
        return LLMResponse(
            content=first_content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        )
