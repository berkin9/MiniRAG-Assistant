"""Tests for deterministic bounded plan selection and execution."""

import logging
from dataclasses import dataclass, field

import pytest

from app.agent.agent import Agent
from app.agent.intent import classify_intent
from app.agent.models import (
    AgentExecutionContext,
    AgentPlan,
    AgentStep,
    Intent,
)
from app.agent.plan_selector import PlanSelector, extract_question
from app.agent.tool_selector import ToolSelector
from app.agent.tools import AskTool, RoutingTool
from app.config import Settings
from app.services.answering import AnswerResult
from app.services.routing import RoutingDecision
from app.services.runtime import RoutedAnswer


def _select(request: str) -> AgentPlan:
    intent = classify_intent(request)
    return PlanSelector().select(request, ToolSelector().select(intent))


@pytest.mark.parametrize(
    ("user_input", "name", "tools"),
    [
        ("How is authentication implemented?", "ask", ("ask",)),
        ("Find authentication chunks", "search", ("search",)),
        ("List collections", "collections", ("collections",)),
        ("Which collection handles authentication?", "routing", ("routing",)),
        ("Show the routing", "routing", ("routing",)),
        ("Show the matching sources", "search", ("search",)),
    ],
)
def test_existing_requests_keep_single_step_plans(
    user_input: str, name: str, tools: tuple[str, ...]
) -> None:
    """Normal deterministic intent behavior must remain unchanged."""
    plan = _select(user_input)

    assert plan.name == name
    assert tuple(step.tool for step in plan.steps) == tools


@pytest.mark.parametrize(
    ("user_input", "name", "tools"),
    [
        (
            "Which collection handles authentication, and what do the documents say?",
            "route_and_ask",
            ("routing", "ask"),
        ),
        (
            "Route this question and retrieve the matching chunks: "
            "How are refresh tokens stored?",
            "route_and_search",
            ("routing", "search"),
        ),
    ],
)
def test_clear_compound_requests_select_predefined_two_step_plan(
    user_input: str, name: str, tools: tuple[str, ...]
) -> None:
    """Only supported routing compounds should produce two steps."""
    plan = _select(user_input)

    assert plan.name == name
    assert tuple(step.tool for step in plan.steps) == tools
    assert all(step.input_mode == "extracted_question" for step in plan.steps)


def test_ambiguous_routing_question_remains_single_step() -> None:
    """Mentioning routing alone must not create a compound plan."""
    plan = _select("What is routing in this project?")

    assert plan.name == "ask"
    assert len(plan.steps) == 1


def test_plan_rejects_zero_steps() -> None:
    """A bounded plan must execute something."""
    with pytest.raises(ValueError, match="at least one"):
        AgentPlan("ask", (), "Invalid")


def test_plan_rejects_more_than_two_steps() -> None:
    """No plan may exceed the hard two-step limit."""
    with pytest.raises(ValueError, match="more than two"):
        AgentPlan(
            "route_and_ask",
            (AgentStep("routing"), AgentStep("ask"), AgentStep("ask")),
            "Invalid",
        )


def test_unknown_tool_name_is_rejected() -> None:
    """Plans cannot invent tool names."""
    with pytest.raises(ValueError, match="Unsupported agent tool"):
        AgentStep("invented")


def test_unsupported_tool_combination_is_rejected() -> None:
    """Only the six declared plan shapes are valid."""
    with pytest.raises(ValueError, match="requires tools"):
        AgentPlan(
            "route_and_ask",
            (AgentStep("routing"), AgentStep("search")),
            "Invalid",
        )


@pytest.mark.parametrize(
    ("user_input", "expected"),
    [
        (
            "Explain the routing, then answer: how is authentication implemented?",
            "how is authentication implemented?",
        ),
        (
            "Route this question and show matching sources: "
            "where are refresh tokens stored?",
            "where are refresh tokens stored?",
        ),
        ("Which collection handles authentication?", "Which collection handles authentication?"),
    ],
)
def test_question_extraction_is_conservative(
    user_input: str, expected: str
) -> None:
    """Only obvious command separators should remove the instruction prefix."""
    assert extract_question(user_input) == expected


@dataclass
class FakeContextTool:
    """Record ordered calls and explicit per-run routing context."""

    name: str
    calls: list[tuple[str, str, str | None]]
    fail: bool = False
    collections: list[str | None] = field(default_factory=list)

    def run(self, request: str) -> tuple[str, ...]:
        return self.run_with_context(request, AgentExecutionContext())

    def run_with_context(
        self, request: str, context: AgentExecutionContext
    ) -> tuple[str, ...] | RoutingDecision:
        self.calls.append((self.name, request, context.selected_collection))
        self.collections.append(context.selected_collection)
        if self.fail:
            raise RuntimeError("first step failed")
        if self.name == "routing":
            context.selected_collection = "technical"
            return RoutingDecision("technical", "Fake route.")
        return (f"{self.name} result",)


def _agent(
    calls: list[tuple[str, str, str | None]], *, routing_fails: bool = False
) -> tuple[Agent, dict[str, FakeContextTool]]:
    tools = {
        name: FakeContextTool(name, calls, fail=routing_fails and name == "routing")
        for name in ("ask", "search", "collections", "routing")
    }
    return Agent(tools.values()), tools


def test_two_tools_execute_in_order_and_return_ordered_results() -> None:
    """A compound plan should execute exactly its two declared steps."""
    calls: list[tuple[str, str, str | None]] = []
    agent, _ = _agent(calls)

    response = agent.run(
        "Explain the routing, then answer: how is authentication implemented?"
    )

    assert [call[0] for call in calls] == ["routing", "ask"]
    assert calls[0][1] == "how is authentication implemented?"
    assert calls[1][1] == "how is authentication implemented?"
    assert tuple(step.tool for step in response.steps) == ("routing", "ask")
    assert response.final_result == ("ask result",)
    assert response.result == response.final_result


def test_failed_first_step_stops_execution_and_preserves_exception() -> None:
    """The second tool must not run when routing raises."""
    calls: list[tuple[str, str, str | None]] = []
    agent, _ = _agent(calls, routing_fails=True)

    with pytest.raises(RuntimeError, match="first step failed"):
        agent.run("Route this question and retrieve the matching chunks")

    assert [call[0] for call in calls] == ["routing"]


def test_routed_collection_is_passed_to_second_tool() -> None:
    """The second step should reuse routing rather than route independently."""
    calls: list[tuple[str, str, str | None]] = []
    agent, tools = _agent(calls)

    agent.run("Route this question and retrieve the matching chunks")

    assert tools["routing"].collections == [None]
    assert tools["search"].collections == ["technical"]


def test_execution_context_does_not_persist_between_requests() -> None:
    """A prior route must not influence a later independent question."""
    calls: list[tuple[str, str, str | None]] = []
    agent, tools = _agent(calls)

    agent.run("Route this question and answer the question")
    agent.run("How does deployment work?")

    assert tools["ask"].collections == ["technical", None]


def test_agent_rejects_a_plan_tool_missing_from_injected_registry() -> None:
    """Every selected step must exist in the concrete agent registry."""
    agent = Agent((FakeContextTool("ask", []),))

    with pytest.raises(ValueError, match="not configured: search"):
        agent.run("Find the matching chunks")


def test_builtin_tools_route_once_and_reuse_collection() -> None:
    """A compound answer must not independently invoke routing twice."""
    routing_calls = 0
    answer_arguments: list[tuple[object, ...]] = []
    settings = Settings()

    def route_runner(*arguments: object) -> RoutingDecision:
        nonlocal routing_calls
        routing_calls += 1
        return RoutingDecision("technical", "Fake route.")

    def answer_runner(*arguments: object) -> RoutedAnswer:
        answer_arguments.append(arguments)
        return RoutedAnswer(
            AnswerResult("question", "answer", (), False, "technical"),
            RoutingDecision("technical", "Explicit collection.", strategy="manual"),
        )

    agent = Agent(
        (
            RoutingTool(settings, route_runner),
            AskTool(settings, answer_runner),
        )
    )

    agent.run("Route this question and answer: how does authentication work?")

    assert routing_calls == 1
    assert answer_arguments == [
        ("how does authentication work?", 4, settings, "manual", "technical")
    ]


def test_agent_logs_plan_and_tools_without_request_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Execution metadata should be observable without logging user text."""
    agent, _ = _agent([])

    with caplog.at_level(logging.INFO, logger="app.agent.agent"):
        agent.run("Route this question and answer the question SECRET-TEXT")

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "plan=route_and_ask steps=2" in messages
    assert "tool=routing" in messages
    assert "tool=ask" in messages
    assert "success=true" in messages
    assert "SECRET-TEXT" not in messages
