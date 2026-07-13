"""Tests for command-line answer output and command compatibility."""

from pathlib import Path

import pytest

from app import main as cli
from app.agent.models import AgentResponse, Intent, ToolDecision
from app.config import ConfigurationError, Settings
from app.services.answering import AnswerResult
from app.services.context_builder import AnswerSource
from app.services.retrieval import RetrievalResponse
from app.services.routing import RoutingDecision
from app.services.runtime import RoutedSearch


def _answer(with_context: bool = True) -> AnswerResult:
    source = AnswerSource(
        label="Source 1",
        source_file="plan.pdf",
        file_type="pdf",
        page_number=4,
        chunk_index=7,
        document_hash="abc",
        distance=0.25,
        text_preview="The deadline is Friday.",
    )
    return AnswerResult(
        question="When?",
        answer=(
            "The deadline is Friday [Source 1]."
            if with_context
            else "I could not find relevant information in the indexed documents."
        ),
        sources=(source,) if with_context else (),
        has_relevant_context=with_context,
    )


def test_ask_command_prints_answer_and_sources(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ask should display provider-independent answers and citations."""
    monkeypatch.setattr(cli, "ask_with_settings", lambda *args: _answer())

    exit_code = cli.main(["ask", "When?"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "The deadline is Friday" in output
    assert "Sources:" in output
    assert "plan.pdf, page 4, chunk 7, distance 0.2500" in output


def test_ask_no_context_is_clear_and_successful(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No-context answers are valid results without a Sources section."""
    monkeypatch.setattr(cli, "ask_with_settings", lambda *args: _answer(False))

    exit_code = cli.main(["ask", "Unknown?"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "could not find relevant information" in output
    assert "Sources:" not in output


def test_configuration_error_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Configuration failures should produce a concise non-zero CLI result."""
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: (_ for _ in ()).throw(ConfigurationError("bad configuration")),
    )

    assert cli.main(["ask", "Question?"]) == 1
    assert "Error: bad configuration" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["ingest", "data"],
        ["index", "data"],
        ["search", "query"],
    ],
)
def test_existing_commands_still_dispatch(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    """Existing command syntax should remain accepted and dispatched."""
    calls: list[str] = []
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=Path("data")))
    monkeypatch.setattr(cli, "_run_ingest", lambda *args: calls.append("ingest"))
    monkeypatch.setattr(cli, "_run_index", lambda *args: calls.append("index"))
    monkeypatch.setattr(cli, "_run_search", lambda *args: calls.append("search"))

    assert cli.main(arguments) == 0
    assert calls == [arguments[0]]


def test_collection_argument_is_passed_to_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI collection selection should be explicit and optional."""
    captured: list[str | None] = []
    monkeypatch.setattr(cli, "get_settings", Settings)
    monkeypatch.setattr(
        cli,
        "_run_index",
        lambda path, settings, collection: captured.append(collection),
    )

    assert cli.main(["index", "data", "--collection", "technical"]) == 0
    assert captured == ["technical"]


@pytest.mark.parametrize(
    ("command", "runner"),
    [("search", "_run_search"), ("ask", "_run_ask")],
)
def test_collection_argument_is_passed_to_query_commands(
    monkeypatch: pytest.MonkeyPatch, command: str, runner: str
) -> None:
    """Search and grounded answers must receive the selected collection."""
    captured: list[str | None] = []
    monkeypatch.setattr(cli, "get_settings", Settings)
    monkeypatch.setattr(
        cli, runner, lambda *arguments: captured.append(arguments[-1])
    )

    assert cli.main([command, "question", "--collection", "project"]) == 0
    assert captured == ["project"]


def test_collections_command_is_deterministic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Configured collections should list in stable order with the default."""
    monkeypatch.setattr(cli, "get_settings", Settings)

    assert cli.main(["collections"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "general (default)",
        "project",
        "technical",
        "policies",
    ]


def test_invalid_explicit_collection_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unsafe explicit name must not fall back to the default."""
    monkeypatch.setattr(cli, "get_settings", Settings)

    exit_code = cli.main(
        ["search", "authentication", "--collection", "../technical"]
    )

    assert exit_code == 1
    assert "path traversal" in capsys.readouterr().err


def test_auto_route_conflicts_with_explicit_collection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Argparse should reject ambiguous manual and automatic selection."""
    with pytest.raises(SystemExit) as captured:
        cli.main(
            [
                "search",
                "question",
                "--collection",
                "project",
                "--auto-route",
            ]
        )

    assert captured.value.code != 0
    assert "not allowed with argument" in capsys.readouterr().err


def test_automatic_search_prints_routing_decision(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Automatic CLI queries should make their route observable."""
    decision = RoutingDecision(
        "technical",
        "Matched technical terms: authentication, implementation",
        0.75,
        "deterministic",
    )
    monkeypatch.setattr(cli, "get_settings", Settings)
    monkeypatch.setattr(
        cli,
        "search_with_settings",
        lambda *args: RoutedSearch(
            RetrievalResponse("question", (), "technical"), decision
        ),
    )

    assert cli.main(["search", "question", "--auto-route"]) == 0
    output = capsys.readouterr().out
    assert "Selected collection: technical" in output
    assert "Routing strategy: deterministic" in output
    assert "Reason: Matched technical terms" in output
    assert "No relevant results found" in output


def test_agent_command_prints_selection_reason_and_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Agent CLI should expose its one tool decision and structured output."""
    response = AgentResponse(
        "list collections",
        Intent.COLLECTIONS,
        ToolDecision("collections", "The request asks for collections."),
        ("general", "technical"),
    )

    class FakeAgent:
        def run(self, request: str) -> AgentResponse:
            assert request == "list collections"
            return response

    monkeypatch.setattr(cli, "get_settings", Settings)
    monkeypatch.setattr(cli, "build_agent", lambda settings: FakeAgent())

    assert cli.main(["agent", "list collections"]) == 0
    output = capsys.readouterr().out
    assert "Selected tool: collections" in output
    assert "Reason: The request asks for collections." in output
    assert "- general" in output
    assert "- technical" in output
