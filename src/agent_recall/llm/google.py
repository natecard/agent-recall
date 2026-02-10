from __future__ import annotations

import importlib
import os
from typing import Any

from agent_recall.llm.base import LLMProvider, LLMResponse, Message


class GoogleProvider(LLMProvider):
    def __init__(self, model: str, api_key_env: str = "GOOGLE_API_KEY"):
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key: set {api_key_env}")

        try:
            genai = importlib.import_module("google.generativeai")
        except ImportError as exc:
            raise ValueError("Install google-generativeai to use Google provider") from exc

        genai.configure(api_key=self.api_key)
        self.client: Any = genai.GenerativeModel(model)

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        prompt_parts: list[str] = []
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"Instructions: {message.content}\n\n")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}\n")
            else:
                prompt_parts.append(f"Assistant: {message.content}\n")

        response = await self.client.generate_content_async(
            "".join(prompt_parts),
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        return LLMResponse(content=response.text or "", model=self.model, usage=None)
