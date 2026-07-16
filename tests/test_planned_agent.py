"""Offline tests for controlled execution of planned agent decisions."""

import logging
from dataclasses import dataclass, field

import pytest

from app.agent.agent import Agent
from app.agent.decision_adapter import (
    AgentDecisionAdapter,
    AgentExecutionPreparationError,
)
from app.agent.models import AgentExecutionContext
from app.agent.plan_selector import PlanSelector
from app.agent.planner_models import (
    AgentDecision,
    AgentPlanningResult,
    AgentPlanningStep,
)
from app.agent.planned_agent import (
    PlannedAgentExecutionError,
    PlannedAgentService,
    build_planned_agent_service,
)
from app.agent.planning_service import build_agent_planning_service
from app.config import Settings
from app.services.llm_providers import LLMRequestError
from app.services.routing import RoutingDecision


def _decision(
    plan: str,
    tools: tuple[str, ...],
    confidence: float = 0.9,
) -> AgentDecision:
    return AgentDecision(
        intent="free form metadata that does not control execution",
        selected_plan=plan,
        steps=tuple(AgentPlanningStep(tool=tool) for tool in tools),
        reason="Validated planning reason.",
        confidence=confidence,
    )


def _planning_result(
    plan: str,
    tools: tuple[str, ...],
    *,
    requested: str = "llm",
    used: str = "llm",
    fallback: bool = False,
) -> AgentPlanningResult:
    return AgentPlanningResult(
        decision=_decision(plan, tools, 1.0 if used == "deterministic" else 0.9),
        requested_strategy=requested,
        used_strategy=used,
        fallback_used=fallback,
        fallback_reason="Safe deterministic fallback." if fallback else None,
    )


class FakePlanningService:
    """Return one planning result and count planning attempts."""

    def __init__(self, result: AgentPlanningResult) -> None:
        self.result = result
        self.calls = 0

    def create_plan(self, query: str) -> AgentPlanningResult:
        self.calls += 1
        return self.result


@dataclass
class RecordingTool:
    """Record exact calls and optionally participate in routing context."""

    name: str
    calls: list[tuple[str, str, str | None]]
    fail: bool = False
    seen_collections: list[str | None] = field(default_factory=list)

    def run(self, request: str) -> tuple[str, ...]:
        return self.run_with_context(request, AgentExecutionContext())

    def run_with_context(
        self,
        request: str,
        context: AgentExecutionContext,
    ) -> tuple[str, ...] | RoutingDecision:
        self.calls.append((self.name, request, context.selected_collection))
        self.seen_collections.append(context.selected_collection)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        if self.name == "routing":
            context.selected_collection = "technical"
            return RoutingDecision("technical", "Fake route.")
        return (f"{self.name} result",)


def _service(
    planning: AgentPlanningResult,
    tools: tuple[RecordingTool, ...],
    *,
    plan_selector: PlanSelector | None = None,
) -> tuple[PlannedAgentService, FakePlanningService]:
    planning_service = FakePlanningService(planning)
    agent = Agent(tools, plan_selector=plan_selector)
    return (
        PlannedAgentService(
            planning_service,
            AgentDecisionAdapter(),
            agent,
        ),
        planning_service,
    )


@pytest.mark.parametrize(
    ("plan", "tools"),
    [
        ("ask", ("ask",)),
        ("search", ("search",)),
        ("collections", ("collections",)),
        ("routing", ("routing",)),
        ("route_and_ask", ("routing", "ask")),
        ("route_and_search", ("routing", "search")),
    ],
)
def test_accepted_decision_executes_exact_tools_once_in_order(
    plan: str,
    tools: tuple[str, ...],
) -> None:
    """Every registered LLM plan should use the shared bounded executor."""
    calls: list[tuple[str, str, str | None]] = []
    registered = tuple(RecordingTool(name, calls) for name in tools)
    service, planning_service = _service(
        _planning_result(plan, tools),
        registered,
    )

    result = service.run("Route this question: what is authentication?")

    assert planning_service.calls == 1
    assert [call[0] for call in calls] == list(tools)
    assert result.selected_plan == plan
    assert result.executed_tools == tools
    assert result.executed_steps == len(tools)


def test_prebuilt_execution_never_calls_deterministic_plan_selector() -> None:
    """The integrated flow must not select a second plan after planning."""
    class FailingPlanSelector(PlanSelector):
        def select(self, request: str, decision: object) -> None:
            raise AssertionError("PlanSelector must not be called")

    calls: list[tuple[str, str, str | None]] = []
    service, planning = _service(
        _planning_result("ask", ("ask",)),
        (RecordingTool("ask", calls),),
        plan_selector=FailingPlanSelector(),
    )

    result = service.run("question")

    assert result.executed_tools == ("ask",)
    assert planning.calls == 1


def test_free_form_planner_intent_cannot_control_execution_intent() -> None:
    """Execution metadata should derive from the registry, not LLM intent text."""
    calls: list[tuple[str, str, str | None]] = []
    service, _ = _service(
        _planning_result("route_and_ask", ("routing", "ask")),
        (RecordingTool("routing", calls), RecordingTool("ask", calls)),
    )

    result = service.run("route and answer")

    assert result.planning.decision.intent.startswith("free form")
    assert result.execution.intent.value == "routing"
    assert result.execution.decision.tool == "routing"


def test_combined_result_rejects_execution_that_differs_from_planning() -> None:
    """Planning and execution metadata must not report inconsistent plans."""
    calls: list[tuple[str, str, str | None]] = []
    ask_service, _ = _service(
        _planning_result("ask", ("ask",)),
        (RecordingTool("ask", calls),),
    )
    ask_result = ask_service.run("question")

    with pytest.raises(ValueError, match="Execution plan does not match"):
        type(ask_result)(
            _planning_result("search", ("search",)),
            ask_result.execution,
        )


@pytest.mark.parametrize("final_tool", ["ask", "search"])
def test_routing_context_is_reused_without_second_route(final_tool: str) -> None:
    """The second tool must consume the one collection selected by routing."""
    calls: list[tuple[str, str, str | None]] = []
    routing = RecordingTool("routing", calls)
    final = RecordingTool(final_tool, calls)
    service, _ = _service(
        _planning_result(f"route_and_{final_tool}", ("routing", final_tool)),
        (routing, final),
    )

    result = service.run("Route this question: how does authentication work?")

    assert [call[0] for call in calls] == ["routing", final_tool]
    assert routing.seen_collections == [None]
    assert final.seen_collections == ["technical"]
    assert result.execution.steps[0].result == RoutingDecision(
        "technical", "Fake route."
    )


def test_fallback_decision_is_the_only_plan_executed() -> None:
    """Planning fallback metadata and exact deterministic execution should survive."""
    calls: list[tuple[str, str, str | None]] = []
    planning = _planning_result(
        "search",
        ("search",),
        requested="llm",
        used="deterministic",
        fallback=True,
    )
    service, planner = _service(
        planning,
        (RecordingTool("search", calls),),
    )

    result = service.run("find chunks")

    assert planner.calls == 1
    assert [call[0] for call in calls] == ["search"]
    assert result.used_strategy == "deterministic"
    assert result.fallback_used is True
    assert result.planning.fallback_reason == "Safe deterministic fallback."


def test_missing_second_tool_is_detected_before_first_tool_executes() -> None:
    """Registry preparation should prevent partial execution."""
    calls: list[tuple[str, str, str | None]] = []
    service, planning = _service(
        _planning_result("route_and_ask", ("routing", "ask")),
        (RecordingTool("routing", calls),),
    )

    with pytest.raises(AgentExecutionPreparationError, match="not configured: ask"):
        service.run("route and answer")

    assert planning.calls == 1
    assert calls == []


def test_first_tool_failure_stops_execution_without_replanning() -> None:
    """Execution errors must short-circuit and remain distinct from fallback."""
    calls: list[tuple[str, str, str | None]] = []
    routing = RecordingTool("routing", calls, fail=True)
    ask = RecordingTool("ask", calls)
    service, planning = _service(
        _planning_result("route_and_ask", ("routing", "ask")),
        (routing, ask),
    )

    with pytest.raises(PlannedAgentExecutionError) as captured:
        service.run("route and answer")

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert planning.calls == 1
    assert [call[0] for call in calls] == ["routing"]


class FakeProvider:
    """Provide one planner response with explicit call counts."""

    def __init__(self, response: str = "", fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.fail:
            raise LLMRequestError("provider failure")
        return self.response


def test_deterministic_factory_executes_without_building_planning_provider() -> None:
    """Default integrated execution should remain credential-free."""
    provider_builds = 0

    def provider_factory() -> FakeProvider:
        nonlocal provider_builds
        provider_builds += 1
        return FakeProvider()

    service = build_planned_agent_service(Settings(), provider_factory)

    result = service.run("list collections")

    assert provider_builds == 0
    assert result.requested_strategy == "deterministic"
    assert result.executed_tools == ("collections",)


@pytest.mark.parametrize(
    ("user_request", "tools"),
    [
        ("How is authentication implemented?", ("ask",)),
        (
            "Explain the routing, then answer: how is authentication implemented?",
            ("routing", "ask"),
        ),
    ],
)
def test_integrated_deterministic_planning_preserves_existing_execution(
    user_request: str,
    tools: tuple[str, ...],
) -> None:
    """Default planning should feed existing fixed plans to the shared executor."""
    calls: list[tuple[str, str, str | None]] = []
    executor = Agent(tuple(RecordingTool(name, calls) for name in tools))
    service = PlannedAgentService(
        build_agent_planning_service(Settings()),
        AgentDecisionAdapter(),
        executor,
    )

    result = service.run(user_request)

    assert result.requested_strategy == "deterministic"
    assert result.used_strategy == "deterministic"
    assert result.executed_tools == tools
    assert [call[0] for call in calls] == list(tools)


@pytest.mark.parametrize(
    ("response", "fallback_expected"),
    [
        (
            '{"intent":"list","selected_plan":"collections","steps":'
            '[{"tool":"collections"}],"reason":"List collections.",'
            '"confidence":0.9}',
            False,
        ),
        ("not-json", True),
        (
            '{"intent":"list","selected_plan":"collections","steps":'
            '[{"tool":"collections"}],"reason":"Low confidence.",'
            '"confidence":0.2}',
            True,
        ),
    ],
)
def test_llm_factory_calls_planner_once_and_executes_accepted_or_fallback_plan(
    response: str,
    fallback_expected: bool,
) -> None:
    """Integrated factory should never retry LLM planning."""
    provider = FakeProvider(response)
    service = build_planned_agent_service(
        Settings(agent_planning_mode="llm", openai_api_key="unused"),
        lambda: provider,
    )

    result = service.run("list collections")

    assert provider.calls == 1
    assert result.executed_tools == ("collections",)
    assert result.fallback_used is fallback_expected


def test_provider_planning_failure_falls_back_once_and_executes_once() -> None:
    """A provider error should trigger deterministic planning, not an LLM retry."""
    provider = FakeProvider(fail=True)
    service = build_planned_agent_service(
        Settings(agent_planning_mode="llm", openai_api_key="unused"),
        lambda: provider,
    )

    result = service.run("list collections")

    assert provider.calls == 1
    assert result.used_strategy == "deterministic"
    assert result.fallback_used is True
    assert result.executed_tools == ("collections",)


def test_planned_execution_logs_metadata_without_request_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lifecycle logs should contain safe plan metadata, not user text."""
    calls: list[tuple[str, str, str | None]] = []
    service, _ = _service(
        _planning_result("ask", ("ask",)),
        (RecordingTool("ask", calls),),
    )

    with caplog.at_level(logging.INFO):
        service.run("SECRET USER REQUEST")

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "planned_agent_started" in messages
    assert "agent_plan_prepared" in messages
    assert "agent_tool_started tool=ask step_index=1" in messages
    assert "agent_tool_completed tool=ask step_index=1" in messages
    assert "planned_agent_execution_completed plan=ask steps=1" in messages
    assert "SECRET USER REQUEST" not in messages
