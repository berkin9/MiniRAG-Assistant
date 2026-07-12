"""Tests for command-line answer output and command compatibility."""

from pathlib import Path

import pytest

from app import main as cli
from app.config import ConfigurationError, Settings
from app.services.answering import AnswerResult
from app.services.context_builder import AnswerSource


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
