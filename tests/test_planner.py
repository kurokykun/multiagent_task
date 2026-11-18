import pytest

from app.agents.graph import Deps, build_graph, planner, route_after_planner
from app.errors import LLMInvalidOutputError
from app.llm import BudgetedLLM, TokenBudget
from app.rag.store import HashingEmbedder, VectorStore
from app.sql.executor import ReadOnlyExecutor
from tests.conftest import ROOT, plan, sql


def _deps(llm, db):
    store = VectorStore.from_dir(HashingEmbedder(), ROOT / "docs" / "policies")
    return Deps(llm=BudgetedLLM(llm, TokenBudget(5000), 300), executor=ReadOnlyExecutor(db), store=store)


@pytest.mark.parametrize("route,expected", [
    ("sql", ["sql_agent"]), ("docs", ["docs_agent"]), ("both", ["sql_agent", "docs_agent"]),
])
def test_route_function(route, expected):
    assert route_after_planner({"route": route}) == expected


@pytest.mark.parametrize("route,agents", [
    ("sql", {"planner", "sql_agent", "writer"}),
    ("docs", {"planner", "docs_agent", "writer"}),
    ("both", {"planner", "sql_agent", "docs_agent", "writer"}),
])
async def test_graph_only_runs_the_agents_chosen_by_the_planner(fake_llm_factory, sample_db, route, agents):
    llm = fake_llm_factory({
        "planner": plan(route),
        "sql": sql("SELECT pais, SUM(importe) AS total FROM pedidos GROUP BY pais"),
        "docs": '{"answer": "30 días [1]", "used_sources": [1]}',
        "writer": "respuesta final",
    })
    out = await build_graph().ainvoke({"question": "q", "trace": []},
                                      config={"configurable": {"deps": _deps(llm, sample_db)}})
    assert out["route"] == route
    assert {s["agent"] for s in out["trace"]} == agents
    assert ("sql" in out) == ("sql_agent" in agents)
    assert ("docs" in out) == ("docs_agent" in agents)


@pytest.mark.parametrize("bad", [
    "no es json",
    '{"route": "web"}',
    '{"reason": "falta route"}',
])
async def test_planner_rejects_invalid_llm_output(fake_llm_factory, sample_db, bad):
    llm = fake_llm_factory({"planner": bad})
    with pytest.raises(LLMInvalidOutputError):
        await planner({"question": "q"}, {"configurable": {"deps": _deps(llm, sample_db)}})


def test_rag_retrieves_the_right_policy():
    store = VectorStore.from_dir(HashingEmbedder(), ROOT / "docs" / "policies")
    assert store.search("¿cuántos días tengo para devolver un producto?", k=1)[0].chunk.source == "devoluciones.md"
    assert store.search("¿cuánto cuesta el envío a Argentina y cuánto tarda?", k=1)[0].chunk.source == "envios.md"
    assert store.search("cupón para clientes inactivos", k=1)[0].chunk.source == "descuentos.md"
