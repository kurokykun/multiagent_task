from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.errors import LLMInvalidOutputError, LLMUnavailableError, TokenBudgetExceededError

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLM(Protocol):
    async def complete(self, messages: list[dict], *, purpose: str, max_tokens: int,
                       json_mode: bool = False) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class TokenBudget:
    limit: int
    used: int = 0
    calls: list[dict] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def reserve(self, prompt: str, max_tokens: int, purpose: str) -> int:
        est_prompt = estimate_tokens(prompt)
        available = self.remaining - est_prompt
        if available < 64:
            raise TokenBudgetExceededError(
                f"Presupuesto de tokens agotado antes de la llamada '{purpose}'",
                details={"limit": self.limit, "used": self.used, "estimated_prompt": est_prompt},
            )
        return min(max_tokens, available)

    def record(self, purpose: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.used += prompt_tokens + completion_tokens
        self.calls.append({"purpose": purpose, "prompt_tokens": prompt_tokens,
                           "completion_tokens": completion_tokens})
        if self.used > self.limit:
            raise TokenBudgetExceededError(
                f"La llamada '{purpose}' superó el presupuesto de tokens",
                details={"limit": self.limit, "used": self.used},
            )

    def summary(self) -> dict:
        return {"limit": self.limit, "used": self.used, "calls": self.calls}


class BudgetedLLM:
    def __init__(self, llm: LLM, budget: TokenBudget, max_tokens_per_call: int):
        self.llm = llm
        self.budget = budget
        self.max_tokens_per_call = max_tokens_per_call

    async def text(self, messages: list[dict], *, purpose: str, json_mode: bool = False) -> str:
        prompt = "\n".join(m["content"] for m in messages)
        max_tokens = self.budget.reserve(prompt, self.max_tokens_per_call, purpose)
        resp = await self.llm.complete(messages, purpose=purpose, max_tokens=max_tokens, json_mode=json_mode)
        self.budget.record(purpose, resp.prompt_tokens, resp.completion_tokens)
        if not resp.text or not resp.text.strip():
            raise LLMInvalidOutputError(f"El LLM devolvió una respuesta vacía en '{purpose}'")
        return resp.text

    async def structured(self, messages: list[dict], schema: type[T], *, purpose: str) -> T:
        raw = await self.text(messages, purpose=purpose, json_mode=True)
        return parse_structured(raw, schema, purpose)


def parse_structured(raw: str, schema: type[T], purpose: str) -> T:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[cleaned.find("{"):] if "{" in cleaned else cleaned
    try:
        return schema.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as e:
        raise LLMInvalidOutputError(
            f"Salida no válida del LLM en '{purpose}'",
            details={"error": str(e)[:300], "raw": raw[:300]},
        ) from e


class OpenAICompatibleLLM:
    def __init__(self, api_key: str, model: str, base_url: str | None, temperature: float, timeout_s: float):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url,
                                  timeout=timeout_s, max_retries=1)
        self.model = model
        self.temperature = temperature

    async def complete(self, messages: list[dict], *, purpose: str, max_tokens: int,
                       json_mode: bool = False) -> LLMResponse:
        import openai

        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens,
                temperature=self.temperature, **kwargs,
            )
        except openai.APIError as e:
            raise LLMUnavailableError(f"Fallo del proveedor LLM en '{purpose}': {type(e).__name__}",
                                      details={"error": str(e)[:300]}) from e
        choice = resp.choices[0] if resp.choices else None
        if choice is None or choice.message.content is None:
            raise LLMInvalidOutputError(f"El LLM no devolvió contenido en '{purpose}'")
        if choice.finish_reason == "length":
            raise LLMInvalidOutputError(f"Respuesta truncada por límite de tokens en '{purpose}'")
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content,
            prompt_tokens=usage.prompt_tokens if usage else estimate_tokens(str(messages)),
            completion_tokens=usage.completion_tokens if usage else estimate_tokens(choice.message.content),
        )
