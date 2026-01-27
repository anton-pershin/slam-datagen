from __future__ import annotations

from typing import Protocol

from pydantic_ai import Agent
from pydantic_ai.models import Model


class MessageGenerator(Protocol):
    def generate(self, user_prompt: str) -> str:
        raise NotImplementedError

    def generate_many(self, user_prompt: str, batch_size: int) -> list[str]:
        raise NotImplementedError


class MessageGeneratorViaLlm:
    """Generic helper for generating text via a pydantic-ai Agent."""

    def __init__(self, model: Model, system_prompt: str) -> None:
        self._agent = Agent(model, system_prompt=system_prompt)

    def generate(self, user_prompt: str) -> str:
        result = self._agent.run_sync(user_prompt)
        if hasattr(result, "output"):
            return result.output
        if hasattr(result, "text"):
            return str(result.text)
        return str(result)

    def generate_many(self, user_prompt: str, batch_size: int) -> list[str]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        instruction = (
            f"Produce {batch_size} distinct short chat messages as a JSON array of strings."
            " Avoid commentary."
        )
        prompt = f"{user_prompt}\n\n{instruction}"
        result = self._agent.run_sync(prompt, output_type=list[str])
        messages: list[str]
        if hasattr(result, "output"):
            messages = result.output
        else:
            messages = result  # type: ignore[assignment]
        cleaned: list[str] = []
        for message in messages:
            text = message.strip()
            if text:
                cleaned.append(text)
        if not cleaned:
            raise ValueError("LLM returned empty message list")
        return cleaned[:batch_size]
