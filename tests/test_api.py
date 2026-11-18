import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.errors import LLMUnavailableError
from app.rag.store import HashingEmbedder
from tests.conftest import plan, sql


def _client(settings, llm):
    return TestClient(create_app(settings, llm=llm, embedder=HashingEmbedder()))


def test_ask_both_returns_answer_sql_citations_and_trace(settings, fake_llm_factory):
    llm = fake_llm_factory({
        "planner": plan("both"),
        "sql": sql("SELECT COUNT(*) AS n FROM clientes_inactivos_90d"),
        "docs": '{"answer": "Cupón del 15 % [1]", "used_sources": [1]}',
        "writer": "Hay N clientes inactivos; se les ofrece un 15 % [descuentos.md § Recuperación de clientes inactivos].",
    })
    with _client(settings, llm) as c:
        r = c.post("/ask", json={"question": "¿Cuántos clientes inactivos hay y qué descuento les damos?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "both"
    assert body["answer"].startswith("Hay N clientes")
    assert "clientes_inactivos_90d" in body["sql"] and body["sql"].endswith("LIMIT 200")
    assert body["sql_result"]["row_count"] == 1
    assert body["citations"] and body["citations"][0]["source"].endswith(".md")
    assert [s["agent"] for s in body["trace"]][0] == "planner"
    assert body["trace"][-1]["agent"] == "writer"
    assert body["usage"]["used"] > 0


def test_sql_agent_retries_after_rejection_and_execution_error(settings, fake_llm_factory):
    llm = fake_llm_factory({
        "planner": plan("sql"),
        "sql": [sql("DELETE FROM pedidos"),
                sql("SELECT columna_inexistente FROM pedidos"),
                sql("SELECT pais, COUNT(*) n FROM pedidos GROUP BY pais")],
        "writer": "ok",
    })
    with _client(settings, llm) as c:
        body = c.post("/ask", json={"question": "pedidos por país"}).json()
    stages = [a["stage"] for a in body["sql_result"]["attempts"]]
    assert stages == ["validation", "execution", "ok"]
    retry_prompts = [m["content"] for call in llm.calls if call["purpose"] == "sql_attempt_3"
                     for m in call["messages"] if m["role"] == "user"]
    assert any("execution" in p for p in retry_prompts)


def test_sql_agent_gives_up_after_two_retries(settings, fake_llm_factory):
    llm = fake_llm_factory({"planner": plan("sql"), "sql": sql("DROP TABLE pedidos"), "writer": "no se pudo"})
    with _client(settings, llm) as c:
        body = c.post("/ask", json={"question": "borra todo"}).json()
    assert body["sql"] is None
    assert len(body["sql_result"]["attempts"]) == 3
    assert sum(1 for call in llm.calls if call["purpose"].startswith("sql_attempt_")) == 3
    assert body["sql_result"]["error"]


@pytest.mark.parametrize("handlers,status,code", [
    ({"planner": LLMUnavailableError("proveedor caído")}, 502, "llm_unavailable"),
    ({"planner": "esto no es JSON"}, 502, "llm_invalid_output"),
    ({"planner": plan("docs"), "docs": '{"answer": ""}'}, 502, "llm_invalid_output"),
])
def test_llm_failures_return_clear_errors(settings, fake_llm_factory, handlers, status, code):
    with _client(settings, fake_llm_factory(handlers)) as c:
        r = c.post("/ask", json={"question": "¿plazo de devolución?"})
    assert r.status_code == status
    assert r.json()["error"] == code
    assert r.json()["message"]


def test_request_timeout(settings, fake_llm_factory):
    settings.request_timeout_s = 0.2
    llm = fake_llm_factory({"planner": plan("docs")}, delay_s=1.0)
    with _client(settings, llm) as c:
        r = c.post("/ask", json={"question": "¿plazo de devolución?"})
    assert r.status_code == 504
    assert r.json()["error"] == "request_timeout"


def test_token_budget_per_request(settings, fake_llm_factory):
    settings.request_token_budget = 250
    llm = fake_llm_factory({"planner": plan("docs"), "docs": '{"answer": "x [1]", "used_sources": [1]}',
                            "writer": "x"}, tokens_per_call=100)
    with _client(settings, llm) as c:
        r = c.post("/ask", json={"question": "¿plazo de devolución?"})
    assert r.status_code == 429
    body = r.json()
    assert body["error"] == "token_budget_exceeded"
    assert body["trace_partial"][0]["agent"] == "planner"


def test_request_validation(settings, fake_llm_factory):
    with _client(settings, fake_llm_factory({})) as c:
        assert c.post("/ask", json={"question": ""}).status_code == 422
        assert c.get("/health").json()["status"] == "ok"
