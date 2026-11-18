from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from app.config import Settings
from app.data.clean import clean_orders
from app.data.pipeline import load_to_duckdb
from app.llm import LLMResponse

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    def __init__(self, handlers: dict[str, object], delay_s: float = 0.0, tokens_per_call: int = 50):
        self.handlers = handlers
        self.delay_s = delay_s
        self.tokens_per_call = tokens_per_call
        self.calls: list[dict] = []

    async def complete(self, messages, *, purpose, max_tokens, json_mode=False):
        self.calls.append({"purpose": purpose, "messages": messages, "json_mode": json_mode})
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        key = purpose.split("_attempt_")[0] if purpose.startswith("sql_attempt_") else purpose
        handler = self.handlers[key]
        if isinstance(handler, list):
            handler = handler.pop(0)
        if isinstance(handler, Exception):
            raise handler
        text = handler(messages) if callable(handler) else handler
        return LLMResponse(text=text, prompt_tokens=self.tokens_per_call, completion_tokens=self.tokens_per_call)


def plan(route: str) -> str:
    return json.dumps({"route": route, "reason": "test"})


def sql(q: str) -> str:
    return json.dumps({"sql": q})


@pytest.fixture
def fake_llm_factory() -> Callable[..., FakeLLM]:
    return FakeLLM


@pytest.fixture(scope="session")
def sample_db(tmp_path_factory) -> Path:
    from app.data.generate import generate_rows

    raw = pd.DataFrame(generate_rows(n_orders=600, seed=7)).astype(str)
    clean, _ = clean_orders(raw)
    db = tmp_path_factory.mktemp("db") / "test.duckdb"
    load_to_duckdb(clean, db)
    return db


@pytest.fixture
def settings(sample_db) -> Settings:
    return Settings(_env_file=None, db_path=sample_db, docs_dir=ROOT / "docs" / "policies",
                    embeddings_provider="local", request_timeout_s=5, request_token_budget=2000,
                    llm_max_tokens_per_call=300)
