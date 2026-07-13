"""Tests for opt-in Streamlit agent orchestration."""

import pytest

from app import ui
from app.agent.models import AgentResponse, Intent, ToolDecision
from app.config import Settings
from app.services.answering import AnswerResult
from app.services.routing import RoutingDecision
from app.services.runtime import RoutedAnswer


class FakeAgent:
    """Return one controlled response and record requests."""

    def __init__(self, response: AgentResponse) -> None:
        self.response = response
        self.requests: list[str] = []

    def run(self, request: str) -> AgentResponse:
        self.requests.append(request)
        return self.response


def _routed_answer() -> RoutedAnswer:
    return RoutedAnswer(
        AnswerResult("question", "answer", (), False, "technical"),
        RoutingDecision("technical", "Matched technical terms."),
    )


def test_agent_checkbox_path_uses_agent_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled agent mode should bypass the normal Streamlit answer call."""
    response = AgentResponse(
        "question",
        Intent.ASK,
        ToolDecision("ask", "Default grounded question answering."),
        _routed_answer(),
    )
    agent = FakeAgent(response)
    monkeypatch.setattr(ui, "build_agent", lambda settings: agent)
    monkeypatch.setattr(
        ui,
        "answer_with_routing",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("Normal answering must be bypassed")
        ),
    )

    result = ui._run_question("question", Settings(), "project", False, True)

    assert result is response
    assert agent.requests == ["question"]


def test_unchecked_agent_preserves_existing_streamlit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled agent mode should call existing answering with unchanged inputs."""
    expected = _routed_answer()
    captured: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        ui,
        "build_agent",
        lambda settings: (_ for _ in ()).throw(
            AssertionError("Agent must not be built")
        ),
    )

    def answer(*arguments: object) -> RoutedAnswer:
        captured.append(arguments)
        return expected

    monkeypatch.setattr(ui, "answer_with_routing", answer)
    settings = Settings(default_top_k=3)

    result = ui._run_question("question", settings, "project", False, False)

    assert result is expected
    assert captured == [("question", 3, settings, "manual", "project")]
