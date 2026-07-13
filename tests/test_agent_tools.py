"""Offline tests for adapters around existing MiniRAG services."""

from collections.abc import Callable

import pytest

from app.agent.models import AgentExecutionContext
from app.agent.tools import AskTool, CollectionsTool, RoutingTool, SearchTool
from app.config import Settings
from app.services.answering import AnswerResult
from app.services.retrieval import RetrievalResponse
from app.services.routing import RoutingDecision
from app.services.runtime import RoutedAnswer, RoutedSearch


def _routing() -> RoutingDecision:
    return RoutingDecision("technical", "Matched technical terms.", 0.8)


def _answer() -> RoutedAnswer:
    return RoutedAnswer(
        AnswerResult("question", "answer", (), False, "technical"),
        _routing(),
    )


def _search() -> RoutedSearch:
    return RoutedSearch(
        RetrievalResponse("query", (), "technical"),
        _routing(),
    )


def test_ask_tool_uses_automatic_routed_answering() -> None:
    """AskTool should delegate once with automatic routing and configured top-k."""
    captured: list[tuple[object, ...]] = []

    def runner(*arguments: object) -> RoutedAnswer:
        captured.append(arguments)
        return _answer()

    settings = Settings(default_top_k=3)
    result = AskTool(settings, runner).run("How does authentication work?")

    assert result == _answer()
    assert captured == [
        ("How does authentication work?", 3, settings, "automatic", None)
    ]


def test_search_tool_uses_automatic_retrieval() -> None:
    """SearchTool should retrieve only and preserve routed metadata."""
    captured: list[tuple[object, ...]] = []

    def runner(*arguments: object) -> RoutedSearch:
        captured.append(arguments)
        return _search()

    settings = Settings(default_top_k=2)
    result = SearchTool(settings, runner).run("Find authentication")

    assert result == _search()
    assert captured == [("Find authentication", 2, settings, "automatic", None)]


def test_collections_tool_uses_existing_registry() -> None:
    """CollectionsTool should retain configured registry order."""
    settings = Settings(rag_collections=("general", "policies", "technical"))

    result = CollectionsTool(settings).run("list collections")

    assert result == ("general", "policies", "technical")


def test_routing_tool_reuses_automatic_router() -> None:
    """RoutingTool should delegate without retrieval or duplicated rules."""
    captured: list[tuple[object, ...]] = []

    def runner(*arguments: object) -> RoutingDecision:
        captured.append(arguments)
        return _routing()

    settings = Settings()
    result = RoutingTool(settings, runner).run("Which collection?")

    assert result == _routing()
    assert captured == [("Which collection?", settings, "automatic", None)]


def test_routing_tool_records_collection_in_execution_context() -> None:
    """Routing should make its one decision available to a following step."""
    context = AgentExecutionContext()
    settings = Settings()

    RoutingTool(settings, lambda *args: _routing()).run_with_context(
        "Which collection?", context
    )

    assert context.selected_collection == "technical"


@pytest.mark.parametrize(
    ("tool_type", "result_factory"),
    [(AskTool, _answer), (SearchTool, _search)],
)
def test_query_tools_reuse_context_collection(
    tool_type: type[AskTool] | type[SearchTool],
    result_factory: Callable[[], RoutedAnswer | RoutedSearch],
) -> None:
    """A second query step should bypass automatic routing consistently."""
    captured: list[tuple[object, ...]] = []

    def runner(*arguments: object) -> RoutedAnswer | RoutedSearch:
        captured.append(arguments)
        return result_factory()

    settings = Settings(default_top_k=2)
    context = AgentExecutionContext(selected_collection="technical")

    tool_type(settings, runner).run_with_context("question", context)

    assert captured == [("question", 2, settings, "manual", "technical")]
