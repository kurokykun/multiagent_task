from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agents.graph import Deps, build_graph
from app.config import Settings, get_settings
from app.errors import AppError, RequestTimeoutError
from app.llm import LLM, BudgetedLLM, OpenAICompatibleLLM, TokenBudget
from app.rag.store import Embedder, HashingEmbedder, OpenAIEmbedder, VectorStore
from app.sql.executor import ReadOnlyExecutor

log = logging.getLogger("app")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Citation(BaseModel):
    ref: int
    source: str
    section: str
    score: float
    excerpt: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    route: str
    sql: str | None = Field(description="SQL ejecutado (ya validado y con LIMIT)")
    sql_result: dict[str, Any] | None
    citations: list[Citation]
    trace: list[dict[str, Any]]
    usage: dict[str, Any]
    elapsed_ms: float


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    message: str
    details: dict[str, Any] = {}
    trace_partial: list[dict[str, Any]] = []


def _default_llm(s: Settings) -> LLM:
    return OpenAICompatibleLLM(api_key=s.llm_api_key, model=s.llm_model, base_url=s.llm_base_url,
                               temperature=s.llm_temperature, timeout_s=s.llm_timeout_s)


def _default_embedder(s: Settings) -> Embedder:
    if s.embeddings_provider == "local":
        return HashingEmbedder()
    return OpenAIEmbedder(api_key=s.llm_api_key, model=s.embeddings_model, base_url=s.llm_base_url,
                          timeout_s=s.llm_timeout_s)


def create_app(settings: Settings | None = None, *, llm: LLM | None = None,
               embedder: Embedder | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.db_path.exists():
            raise RuntimeError(f"No existe {settings.db_path}. Ejecuta antes: python -m app.data.pipeline")
        app.state.llm = llm or _default_llm(settings)
        app.state.store = VectorStore.from_dir(embedder or _default_embedder(settings), settings.docs_dir)
        app.state.executor = ReadOnlyExecutor(settings.db_path)
        app.state.graph = build_graph()
        yield

    app = FastAPI(title="Asistente multiagente de pedidos", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": settings.llm_model, "embeddings": settings.embeddings_provider}

    @app.post("/ask", response_model=AskResponse, responses={429: {"model": ErrorResponse},
              502: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
    async def ask(body: AskRequest, request: Request):
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        st = request.app.state
        budget = TokenBudget(limit=settings.request_token_budget)
        deps = Deps(llm=BudgetedLLM(st.llm, budget, settings.llm_max_tokens_per_call),
                    executor=st.executor, store=st.store, sql_max_rows=settings.sql_max_rows,
                    sql_max_retries=settings.sql_max_retries, rag_top_k=settings.rag_top_k)
        partial: list[dict] = []

        async def run() -> dict:
            state: dict = {"question": body.question, "trace": []}
            async for update in st.graph.astream(state, config={"configurable": {"deps": deps}},
                                                 stream_mode="updates"):
                for node_out in update.values():
                    if not node_out:
                        continue
                    partial.extend(node_out.get("trace", []))
                    state.update({k: v for k, v in node_out.items() if k != "trace"})
            return state

        try:
            final = await asyncio.wait_for(run(), timeout=settings.request_timeout_s)
        except TimeoutError:
            return _error(request_id, RequestTimeoutError(
                f"La petición superó el tiempo máximo de {settings.request_timeout_s}s"), partial, budget)
        except AppError as e:
            log.warning("request %s failed: %s %s", request_id, e.code, e.message)
            return _error(request_id, e, partial, budget)

        sql = final.get("sql")
        docs = final.get("docs")
        return AskResponse(
            request_id=request_id,
            answer=final["answer"],
            route=final["route"],
            sql=sql["query"] if sql else None,
            sql_result=({"columns": sql["columns"], "rows": sql["rows"], "row_count": sql["row_count"],
                         "attempts": sql["attempts"], "error": sql["error"]} if sql else None),
            citations=docs["citations"] if docs else [],
            trace=partial,
            usage=budget.summary(),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    return app


def _error(request_id: str, e: AppError, partial: list[dict], budget: TokenBudget) -> JSONResponse:
    body = ErrorResponse(request_id=request_id, error=e.code, message=e.message,
                         details={**e.details, "usage": budget.summary()}, trace_partial=partial)
    return JSONResponse(status_code=e.http_status, content=body.model_dump())


def app_factory() -> FastAPI:
    return create_app()
