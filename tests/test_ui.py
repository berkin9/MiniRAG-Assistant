"""Tests for opt-in Streamlit agent orchestration."""

from contextlib import contextmanager

import pytest

from app import ui
from app.agent.models import (
    AgentPlan,
    AgentResponse,
    AgentStep,
    AgentStepResult,
    Intent,
    ToolDecision,
)
from app.agent.planner_models import (
    AgentDecision,
    AgentPlanningResult,
    AgentPlanningStep,
)
from app.agent.planned_agent import PlannedAgentResult
from app.config import Settings
from app.services.answering import AnswerResult
from app.services.routing import RoutingDecision
from app.services.runtime import RoutedAnswer


class FakeAgent:
    """Return one controlled response and record requests."""

    def __init__(self, response: PlannedAgentResult) -> None:
        self.response = response
        self.requests: list[str] = []

    def run(self, request: str) -> PlannedAgentResult:
        self.requests.append(request)
        return self.response


class FakeStreamlit:
    """Provide session state and visible spinner events for UX tests."""

    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.events: list[str] = []

    @contextmanager
    def spinner(self, message: str):
        self.events.append(f"start:{message}")
        try:
            yield
        finally:
            self.events.append("stop")

    def error(self, message: str) -> None:
        self.events.append(f"error:{message}")


def _routed_answer() -> RoutedAnswer:
    return RoutedAnswer(
        AnswerResult("question", "answer", (), False, "technical"),
        RoutingDecision("technical", "Matched technical terms."),
    )


def _planned(
    response: AgentResponse,
    *,
    requested: str = "deterministic",
    used: str = "deterministic",
    fallback: bool = False,
) -> PlannedAgentResult:
    plan = response.plan
    assert plan is not None
    planning = AgentPlanningResult(
        decision=AgentDecision(
            intent=response.intent.value,
            selected_plan=plan.name,
            steps=tuple(
                AgentPlanningStep(tool=step.tool) for step in plan.steps
            ),
            reason=plan.reason,
            confidence=1.0,
        ),
        requested_strategy=requested,
        used_strategy=used,
        fallback_used=fallback,
        fallback_reason="Safe fallback." if fallback else None,
    )
    return PlannedAgentResult(planning, response)


def test_submission_state_disables_empty_and_active_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The submit button guard should reflect input and processing state."""
    fake_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake_st)

    ui._initialize_request_state()

    assert ui._submit_disabled("   ") is True
    assert ui._submit_disabled("question") is False
    fake_st.session_state["is_processing"] = True
    assert ui._submit_disabled("question") is True


def test_start_processing_prevents_duplicate_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An active request must ignore repeated start callbacks."""
    fake_st = FakeStreamlit()
    fake_st.session_state.update(
        {
            "is_processing": False,
            "request_outcome": "old",
            "request_error": "old error",
        }
    )
    monkeypatch.setattr(ui, "st", fake_st)

    ui._start_processing()
    fake_st.session_state["request_outcome"] = "active"
    ui._start_processing()

    assert fake_st.session_state["is_processing"] is True
    assert fake_st.session_state["request_outcome"] == "active"
    assert "request_error" not in fake_st.session_state


def test_processing_shows_spinner_and_reenables_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful requests should visibly process and always release the guard."""
    fake_st = FakeStreamlit()
    fake_st.session_state["is_processing"] = True
    expected = _routed_answer()
    monkeypatch.setattr(ui, "st", fake_st)
    monkeypatch.setattr(ui, "_run_question", lambda *args: expected)

    ui._process_submission("question", Settings(), "general", False, False)

    assert fake_st.events == ["start:Processing request...", "stop"]
    assert fake_st.session_state["is_processing"] is False
    assert fake_st.session_state["request_outcome"] is expected
    assert "request_error" not in fake_st.session_state


def test_processing_reenables_after_expected_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected request failures should release the guard and save the error."""
    fake_st = FakeStreamlit()
    fake_st.session_state["is_processing"] = True
    monkeypatch.setattr(ui, "st", fake_st)
    monkeypatch.setattr(
        ui,
        "_run_question",
        lambda *args: (_ for _ in ()).throw(ValueError("bad request")),
    )

    ui._process_submission("question", Settings(), "general", False, False)

    assert fake_st.events == ["start:Processing request...", "stop"]
    assert fake_st.session_state["is_processing"] is False
    assert fake_st.session_state["request_error"] == "bad request"
    assert "request_outcome" not in fake_st.session_state


def test_processing_guard_resets_when_unexpected_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finally guard should release even when Streamlit reports a traceback."""
    fake_st = FakeStreamlit()
    fake_st.session_state["is_processing"] = True
    monkeypatch.setattr(ui, "st", fake_st)
    monkeypatch.setattr(
        ui,
        "_run_question",
        lambda *args: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        ui._process_submission("question", Settings(), "general", False, False)

    assert fake_st.session_state["is_processing"] is False


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
    planned = _planned(response)
    agent = FakeAgent(planned)
    monkeypatch.setattr(
        ui,
        "build_planned_agent_service",
        lambda settings: agent,
    )
    monkeypatch.setattr(
        ui,
        "answer_with_routing",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("Normal answering must be bypassed")
        ),
    )

    result = ui._run_question("question", Settings(), "project", False, True)

    assert result is planned
    assert agent.requests == ["question"]


def test_unchecked_agent_preserves_existing_streamlit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled agent mode should call existing answering with unchanged inputs."""
    expected = _routed_answer()
    captured: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        ui,
        "build_planned_agent_service",
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


def test_streamlit_renders_two_step_agent_results_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two-step responses should show plan, routing, and final answer."""
    routing = RoutingDecision("technical", "Matched technical terms.")
    answer = _routed_answer()
    response = AgentResponse(
        "compound",
        Intent.ROUTING,
        ToolDecision("routing", "Routing requested."),
        answer,
        AgentPlan(
            "route_and_ask",
            (AgentStep("routing"), AgentStep("ask")),
            "Routing and answer requested.",
        ),
        (
            AgentStepResult("routing", routing),
            AgentStepResult("ask", answer),
        ),
    )
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(ui.st, "info", lambda value: events.append(("info", value)))
    monkeypatch.setattr(
        ui.st, "subheader", lambda value: events.append(("subheader", value))
    )
    monkeypatch.setattr(
        ui,
        "_render_routing",
        lambda value: events.append(("routing", value.collection)),
    )
    monkeypatch.setattr(
        ui,
        "_render_routed_answer",
        lambda value, show_routing: events.append(("answer", show_routing)),
    )

    ui._render_agent_response(response)

    assert events[0][0] == "info"
    assert "route_and_ask" in str(events[0][1])
    assert events[1:] == [
        ("subheader", "Step 1: routing"),
        ("routing", "technical"),
        ("subheader", "Step 2: ask"),
        ("answer", False),
    ]


def test_streamlit_renders_safe_planning_metadata_after_main_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning details should be expandable without obscuring tool output."""
    response = AgentResponse(
        "question",
        Intent.ASK,
        ToolDecision("ask", "Grounded answer."),
        _routed_answer(),
    )
    result = _planned(
        response,
        requested="llm",
        used="deterministic",
        fallback=True,
    )
    events: list[tuple[str, object]] = []

    @contextmanager
    def expander(label: str):
        events.append(("expander", label))
        yield

    monkeypatch.setattr(
        ui,
        "_render_agent_response",
        lambda value: events.append(("execution", value)),
    )
    monkeypatch.setattr(ui.st, "expander", expander)
    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda value: events.append(("markdown", value)),
    )

    ui._render_planned_agent_result(result)

    assert events[0] == ("execution", response)
    assert events[1] == ("expander", "Agent planning details")
    details = str(events[2][1])
    assert "Requested strategy: **llm**" in details
    assert "Used strategy: **deterministic**" in details
    assert "Fallback used: **yes**" in details
    assert "Fallback reason: Safe fallback." in details
    assert "Executed tools: **ask**" in details
