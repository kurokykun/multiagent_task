from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens_per_call: int = 800
    llm_timeout_s: float = 20.0

    embeddings_provider: str = "openai"
    embeddings_model: str = "text-embedding-3-small"

    request_timeout_s: float = 45.0
    request_token_budget: int = 6000

    sql_max_rows: int = 200
    sql_max_retries: int = 2

    db_path: Path = Path("data/pedidos.duckdb")
    docs_dir: Path = Path("docs/policies")
    rag_top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
