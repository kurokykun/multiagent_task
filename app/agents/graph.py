from __future__ import annotations

import asyncio
import json
import operator
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import duckdb
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents import prompts
from app.errors import LLMInvalidOutputError, SQLValidationError
from app.llm import BudgetedLLM
from app.rag.store import VectorStore
from app.sql.executor import SCHEMA_DESCRIPTION, ReadOnlyExecutor
from app.sql.validator import validate_sql


@dataclass
class Deps:
    llm: BudgetedLLM
    executor: ReadOnlyExecutor
    store: VectorStore
    sql_max_rows: int = 200
    sql_max_retries: int = 2
    rag_top_k: int = 4
    writer_max_rows: int = 30


class AgentState(TypedDict, total=False):
    question: str
    route: str
    plan_reason: str
    sql: dict[str, Any] | None
    docs: dict[str, Any] | None
    answer: str
    trace: Annotated[list[dict], operator.add]


class Plan(BaseModel):
    route: Literal["sql", "docs", "both"]
    reason: str = ""


class SQLDraft(BaseModel):
    sql: str = Field(min_length=1)


class DocsAnswer(BaseModel):
    answer: str = Field(min_length=1)
    used_sources: list[int] = Field(default_factory=list)


def _deps(config: RunnableConfig) -> Deps:
    return config["configurable"]["deps"]


def _step(agent: str, action: str, started: float, **detail) -> dict:
    return {"agent": agent, "action": action, "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            **detail}


async def planner(state: AgentState, config: RunnableConfig) -> dict:
    t0 = time.perf_counter()
    deps = _deps(config)
    plan = await deps.llm.structured(
        [{"role": "system", "content": prompts.PLANNER_SYSTEM},
         {"role": "user", "content": state["question"]}],
        Plan, purpose="planner",
    )
    return {"route": plan.route, "plan_reason": plan.reason,
            "trace": [_step("planner", "route", t0, route=plan.route, reason=plan.reason)]}


def route_after_planner(state: AgentState) -> list[str]:
    return {"sql": ["sql_agent"], "docs": ["docs_agent"], "both": ["sql_agent", "docs_agent"]}[state["route"]]


async def sql_agent(state: AgentState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    messages = [
        {"role": "system", "content": prompts.SQL_SYSTEM.format(schema=SCHEMA_DESCRIPTION, max_rows=deps.sql_max_rows)},
        {"role": "user", "content": state["question"]},
    ]
    trace: list[dict] = []
    attempts: list[dict] = []
    for attempt in range(1, deps.sql_max_retries + 2):
        t0 = time.perf_counter()
        draft_sql: str | None = None
        try:
            draft = await deps.llm.structured(messages, SQLDraft, purpose=f"sql_attempt_{attempt}")
            draft_sql = draft.sql
            validated = validate_sql(draft_sql, max_rows=deps.sql_max_rows)
            result = await asyncio.to_thread(deps.executor.run, validated.sql)
        except (SQLValidationError, LLMInvalidOutputError, duckdb.Error) as e:
            stage = ("validation" if isinstance(e, SQLValidationError)
                     else "llm_output" if isinstance(e, LLMInvalidOutputError) else "execution")
            msg = getattr(e, "message", None) or str(e).splitlines()[0][:300]
            attempts.append({"attempt": attempt, "sql": draft_sql, "stage": stage, "error": msg})
            trace.append(_step("sql_agent", "attempt_failed", t0, attempt=attempt, stage=stage, error=msg,
                               sql=draft_sql))
            if draft_sql:
                messages.append({"role": "assistant", "content": json.dumps({"sql": draft_sql})})
            messages.append({"role": "user", "content": prompts.SQL_RETRY_USER.format(error=f"[{stage}] {msg}")})
            continue

        attempts.append({"attempt": attempt, "sql": validated.sql, "stage": "ok"})
        trace.append(_step("sql_agent", "executed", t0, attempt=attempt, sql=validated.sql,
                           tables=validated.tables, rows=result.row_count))
        return {"sql": {"query": validated.sql, "columns": result.columns, "rows": result.rows,
                        "row_count": result.row_count, "attempts": attempts, "error": None},
                "trace": trace}

    return {"sql": {"query": None, "columns": [], "rows": [], "row_count": 0, "attempts": attempts,
                    "error": f"No se obtuvo una consulta válida tras {len(attempts)} intentos"},
            "trace": trace + [{"agent": "sql_agent", "action": "gave_up", "attempts": len(attempts)}]}


async def docs_agent(state: AgentState, config: RunnableConfig) -> dict:
    t0 = time.perf_counter()
    deps = _deps(config)
    hits = await asyncio.to_thread(deps.store.search, state["question"], deps.rag_top_k)
    embed_tokens = getattr(deps.store.embedder, "last_usage", 0)
    if embed_tokens:
        deps.llm.budget.record("docs_embedding", embed_tokens, 0)
    if not hits:
        return {"docs": {"answer": "No se encontró información relevante en las políticas.", "citations": []},
                "trace": [_step("docs_agent", "retrieve", t0, hits=0)]}

    context = "\n\n".join(f"[{i}] ({h.chunk.source} § {h.chunk.section})\n{h.chunk.text}"
                          for i, h in enumerate(hits, start=1))
    out = await deps.llm.structured(
        [{"role": "system", "content": prompts.DOCS_SYSTEM},
         {"role": "user", "content": f"Fragmentos:\n{context}\n\nPregunta: {state['question']}"}],
        DocsAnswer, purpose="docs",
    )
    used = [i for i in out.used_sources if 1 <= i <= len(hits)] or list(range(1, len(hits) + 1))
    citations = [{"ref": i, "source": hits[i - 1].chunk.source, "section": hits[i - 1].chunk.section,
                  "score": round(hits[i - 1].score, 3), "excerpt": hits[i - 1].chunk.text[:240]}
                 for i in used]
    return {"docs": {"answer": out.answer, "citations": citations},
            "trace": [_step("docs_agent", "retrieve_and_answer", t0, hits=len(hits),
                            sources=[f"{c['source']} § {c['section']}" for c in citations])]}


async def writer(state: AgentState, config: RunnableConfig) -> dict:
    t0 = time.perf_counter()
    deps = _deps(config)
    parts = [f"Pregunta: {state['question']}"]
    sql = state.get("sql")
    if sql is not None:
        if sql["error"]:
            parts.append(f"Datos SQL: ERROR — {sql['error']}")
        else:
            preview = sql["rows"][: deps.writer_max_rows]
            parts.append(f"Datos SQL ({sql['row_count']} filas, se muestran {len(preview)}):\n"
                         f"columnas={sql['columns']}\nfilas={json.dumps(preview, ensure_ascii=False)}")
    docs = state.get("docs")
    if docs is not None:
        cites = "; ".join(f"{c['source']} § {c['section']}" for c in docs["citations"]) or "ninguna"
        parts.append(f"Información de políticas: {docs['answer']}\nFuentes: {cites}")

    answer = await deps.llm.text(
        [{"role": "system", "content": prompts.WRITER_SYSTEM},
         {"role": "user", "content": "\n\n".join(parts)}],
        purpose="writer",
    )
    return {"answer": answer.strip(), "trace": [_step("writer", "compose", t0)]}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("sql_agent", sql_agent)
    g.add_node("docs_agent", docs_agent)
    g.add_node("writer", writer)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_after_planner, ["sql_agent", "docs_agent"])
    g.add_edge("sql_agent", "writer")
    g.add_edge("docs_agent", "writer")
    g.add_edge("writer", END)
    return g.compile()
