"""Tests for fused answering, runtime strategy branching, and agent tools."""

from pathlib import Path

import pytest

from app import main as cli
from app import ui
from app.agent.models import AgentExecutionContext
from app.agent.tools import AskTool, SearchTool
from app.config import Settings
from app.services.answering import AnswerResult
from app.services.collection_selection import CollectionSelectionResult
from app.services.cross_collection import CrossCollectionRetrievalResponse
from app.services.multirag_answering import answer_cross_collection
from app.services.retrieval import RetrievalResult
from app.services.routing import RoutingDecision
from app.services.runtime import RoutedAnswer, RoutedSearch


class FakeProvider:
    """Record one grounded generation call without a network."""

    def __init__(self) -> None:
        self.calls = 0
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "Implementation and policy evidence agree [Source 1] [Source 2]."


def _selection() -> CollectionSelectionResult:
    return CollectionSelectionResult(
        collections=("technical", "policies"),
        strategy="deterministic",
        reason="Matched two routes.",
        confidence=0.9,
    )


def _result(source: str, collection: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        text=f"Evidence from {collection}.",
        source_file=source,
        file_type="txt",
        page_number=None,
        chunk_index=rank,
        document_hash=source,
        distance=0.2,
        collection=collection,
        chunk_id=f"{source}:{rank}",
        normalized_score=0.8,
        fusion_score=1 / (60 + rank),
        rank_within_collection=1,
        global_rank=rank,
        matched_collections=(collection,),
        raw_score=0.2,
    )


def _response(results: tuple[RetrievalResult, ...]) -> CrossCollectionRetrievalResponse:
    return CrossCollectionRetrievalResponse(
        query="Compare implementation and policy",
        results=results,
        selection=_selection(),
        selected_collections=("technical", "policies"),
        collections_searched=("technical", "policies"),
        results_per_collection={"technical": 1, "policies": 1},
        total_candidates=len(results),
        deduplicated_candidates=len(results),
        returned_results=len(results),
        collection_failures={},
        latency_ms=2.0,
    )


def test_fused_answer_calls_provider_once_with_collection_aware_context() -> None:
    provider = FakeProvider()
    response = _response(
        (_result("architecture.txt", "technical", 1), _result("policy.txt", "policies", 2))
    )

    result = answer_cross_collection(response, 4_000, lambda: provider)

    assert provider.calls == 1
    assert result.selected_collections == ("technical", "policies")
    assert [source.label for source in result.sources] == ["Source 1", "Source 2"]
    assert [source.collection for source in result.sources] == [
        "technical",
        "policies",
    ]
    assert "Collections: technical" in provider.user_prompt
    assert "incomplete or conflicting" in provider.system_prompt
    assert "unsupported comparisons" in provider.system_prompt
    assert "hidden reasoning" in provider.system_prompt


def test_empty_fused_evidence_uses_existing_no_evidence_behavior() -> None:
    provider = FakeProvider()

    result = answer_cross_collection(_response(()), 4_000, lambda: provider)

    assert result.has_relevant_context is False
    assert result.sources == ()
    assert provider.calls == 0


@pytest.mark.parametrize("tool_class", [AskTool, SearchTool])
def test_agent_retrieval_tools_use_automatic_cross_collection_strategy(
    tool_class: type[AskTool] | type[SearchTool],
) -> None:
    captured: list[tuple[object, ...]] = []
    routed_answer = RoutedAnswer(
        AnswerResult("q", "a", (), False), RoutingDecision("general", "test")
    )
    routed_search = RoutedSearch(
        response=_response(()),
        routing=RoutingDecision("general", "test"),
        selection=_selection(),
    )

    def runner(*arguments: object) -> RoutedAnswer | RoutedSearch:
        captured.append(arguments)
        return routed_answer if tool_class is AskTool else routed_search

    tool = tool_class(
        Settings(rag_retrieval_strategy="cross_collection"), runner=runner
    )
    context = AgentExecutionContext(selected_collection="technical")

    tool.run_with_context("query", context)

    assert captured[0][-2:] == ("automatic", None)


def test_single_collection_agent_tool_still_reuses_routed_context() -> None:
    captured: list[tuple[object, ...]] = []

    def runner(*arguments: object) -> RoutedAnswer:
        captured.append(arguments)
        return RoutedAnswer(
            AnswerResult("q", "a", (), False),
            RoutingDecision("technical", "test"),
        )

    tool = AskTool(Settings(), runner=runner)
    tool.run_with_context(
        "query", AgentExecutionContext(selected_collection="technical")
    )

    assert captured[0][-2:] == ("manual", "technical")


def test_cli_parses_bounded_manual_collection_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, ...] | None] = []
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(rag_retrieval_strategy="cross_collection"),
    )
    monkeypatch.setattr(
        cli,
        "_run_search",
        lambda *arguments: captured.append(arguments[-1]),
    )

    status = cli.main(
        ["search", "query", "--collections", "technical,policies"]
    )

    assert status == 0
    assert captured == [("technical", "policies")]


def test_unknown_manual_collection_fails_before_embedding() -> None:
    from app.services.runtime import search_with_settings

    with pytest.raises(ValueError, match="Unknown collection"):
        search_with_settings(
            "query",
            4,
            Settings(rag_retrieval_strategy="cross_collection"),
            query_mode="manual",
            collections=("secret",),
        )


def test_streamlit_orchestration_passes_manual_multi_collection_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = RoutedAnswer(
        AnswerResult("q", "a", (), False), RoutingDecision("technical", "test")
    )

    def answer(*args: object, **kwargs: object) -> RoutedAnswer:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(ui, "answer_with_routing", answer)
    settings = Settings(rag_retrieval_strategy="cross_collection")

    result = ui._run_question(
        "query",
        settings,
        "general",
        False,
        False,
        ("technical", "policies"),
    )

    assert result is expected
    assert captured["kwargs"] == {"collections": ("technical", "policies")}


def test_cli_cross_collection_output_shows_counts_and_ranked_sources(
    capsys: pytest.CaptureFixture[str],
) -> None:
    response = _response((_result("architecture.txt", "technical", 1),))
    routed = RoutedSearch(
        response=response,
        routing=RoutingDecision("technical", "test"),
        selection=_selection(),
    )

    cli._print_search_response(routed)

    output = capsys.readouterr().out
    assert "technical: 1" in output
    assert "policies: 1" in output
    assert "Duplicates removed: 0" in output
    assert "1. technical — architecture.txt" in output


def test_answer_sources_render_collection_labels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = FakeProvider()
    result = answer_cross_collection(
        _response((_result("architecture.txt", "technical", 1),)),
        4_000,
        lambda: provider,
    )

    cli._print_answer(result)

    assert "technical — architecture.txt" in capsys.readouterr().out


def test_benchmark_page_file_remains_present() -> None:
    """Cross-collection UI work must not replace the Sprint 4 page."""
    assert Path("app/pages/benchmark.py").is_file()
