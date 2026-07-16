"""Offline tests for safe planner orchestration and deterministic fallback."""

from collections.abc import Callable

import pytest

from app.agent.planner import LLMAgentPlanner
from app.agent.planner_models import AgentDecision, AgentPlanningStep
from app.agent.planner_parser import AgentPlanningError
from app.agent.planning_policy import AgentDecisionPolicy
from app.agent.planning_service import (
    AgentPlanningService,
    AgentPlanningServiceError,
    build_agent_planning_service,
)
from app.config import Settings
from app.services.llm_providers import LLMRequestError


def _decision(
    plan: str = "ask",
    tools: tuple[str, ...] = ("ask",),
    confidence: float = 0.9,
) -> AgentDecision:
    return AgentDecision(
        intent="grounded_question",
        selected_plan=plan,
        steps=tuple(AgentPlanningStep(tool=tool) for tool in tools),
        reason="A short planning justification.",
        confidence=confidence,
    )


class FakePlanner:
    """Return one decision or raise one configured error with a call counter."""

    def __init__(self, outcome: AgentDecision | Exception) -> None:
        self.outcome = outcome
        self.calls = 0

    def create_plan(self, query: str) -> AgentDecision:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeProvider:
    """Return a raw response or a provider error without network access."""

    def __init__(self, response: str = "", fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.fail:
            raise LLMRequestError("provider body with secret-token")
        return self.response


def _fallback_service(
    primary: FakePlanner,
    fallback: FakePlanner,
    *,
    minimum_confidence: float = 0.6,
    maximum_steps: int = 2,
    fallback_enabled: bool = True,
) -> AgentPlanningService:
    return AgentPlanningService(
        primary,
        fallback,
        AgentDecisionPolicy(minimum_confidence, maximum_steps),
        requested_strategy="llm",
        fallback_enabled=fallback_enabled,
    )


def test_deterministic_mode_uses_only_deterministic_planner() -> None:
    """Deterministic orchestration should not treat its planner as fallback."""
    primary = FakePlanner(_decision())
    unused_fallback = FakePlanner(AssertionError("fallback called"))
    service = AgentPlanningService(
        primary,
        unused_fallback,
        AgentDecisionPolicy(0.6, 2),
        requested_strategy="deterministic",
    )

    result = service.create_plan("question")

    assert primary.calls == 1
    assert unused_fallback.calls == 0
    assert result.requested_strategy == "deterministic"
    assert result.used_strategy == "deterministic"
    assert result.fallback_used is False


def test_deterministic_factory_never_builds_llm_provider() -> None:
    """Default mode must remain lazy, offline, and credential-free."""
    provider_calls = 0

    def provider_factory() -> FakeProvider:
        nonlocal provider_calls
        provider_calls += 1
        return FakeProvider()

    service = build_agent_planning_service(Settings(), provider_factory)

    result = service.create_plan("How is authentication implemented?")

    assert provider_calls == 0
    assert result.used_strategy == "deterministic"


def test_high_confidence_llm_decision_is_accepted_without_fallback() -> None:
    """A trusted decision should be returned without deterministic planning."""
    primary = FakePlanner(_decision("route_and_ask", ("routing", "ask"), 0.91))
    fallback = FakePlanner(AssertionError("fallback called"))

    result = _fallback_service(primary, fallback).create_plan("question")

    assert primary.calls == 1
    assert fallback.calls == 0
    assert result.decision.selected_plan == "route_and_ask"
    assert result.requested_strategy == "llm"
    assert result.used_strategy == "llm"
    assert result.fallback_used is False


def test_confidence_equal_to_threshold_is_accepted() -> None:
    """The configured confidence threshold should be inclusive."""
    primary = FakePlanner(_decision(confidence=0.6))
    fallback = FakePlanner(AssertionError("fallback called"))

    result = _fallback_service(primary, fallback).create_plan("question")

    assert result.used_strategy == "llm"
    assert fallback.calls == 0


def test_low_confidence_decision_uses_safe_deterministic_fallback() -> None:
    """A valid but untrusted decision should fall back with safe metadata."""
    primary = FakePlanner(_decision(confidence=0.42))
    fallback_decision = _decision("search", ("search",), 1.0)
    fallback = FakePlanner(fallback_decision)

    result = _fallback_service(primary, fallback).create_plan("question")

    assert primary.calls == 1
    assert fallback.calls == 1
    assert result.decision == fallback_decision
    assert result.used_strategy == "deterministic"
    assert result.fallback_used is True
    assert "0.42" in (result.fallback_reason or "")
    assert "0.60" in (result.fallback_reason or "")
    assert result.policy_rejection_reason == result.fallback_reason
    assert result.primary_decision == primary.outcome


def test_agent_planning_error_uses_deterministic_fallback() -> None:
    """Expected primary failures should be converted into safe fallback results."""
    primary = FakePlanner(
        AgentPlanningError("unsafe internal details", code="invalid_decision")
    )
    fallback = FakePlanner(_decision())

    result = _fallback_service(primary, fallback).create_plan("question")

    assert result.fallback_used is True
    assert result.primary_error_type == "AgentPlanningError"
    assert result.fallback_reason == (
        "LLM planner returned an invalid structured decision."
    )
    assert "unsafe" not in (result.fallback_reason or "")


@pytest.mark.parametrize(
    ("provider_factory", "expected_reason"),
    [
        (lambda: FakeProvider("not JSON secret-token"), "invalid JSON"),
        (lambda: FakeProvider(""), "empty response"),
        (lambda: FakeProvider(fail=True), "provider request failed"),
    ],
)
def test_llm_provider_outputs_fall_back_once_without_leaking_details(
    provider_factory: Callable[[], FakeProvider],
    expected_reason: str,
) -> None:
    """Parser and provider failures should make one call and expose safe metadata."""
    provider = provider_factory()
    primary = LLMAgentPlanner(provider)
    fallback = FakePlanner(_decision())
    service = AgentPlanningService(
        primary,
        fallback,
        AgentDecisionPolicy(0.6, 2),
    )

    result = service.create_plan("question")

    assert provider.calls == 1
    assert fallback.calls == 1
    assert expected_reason in (result.fallback_reason or "")
    assert "secret-token" not in (result.fallback_reason or "")


def test_policy_step_limit_rejection_triggers_fallback() -> None:
    """Runtime policy may be stricter than the hard two-step domain limit."""
    primary = FakePlanner(_decision("route_and_search", ("routing", "search")))
    fallback = FakePlanner(_decision())

    result = _fallback_service(
        primary,
        fallback,
        maximum_steps=1,
    ).create_plan("question")

    assert result.fallback_used is True
    assert "configured maximum is 1" in (result.policy_rejection_reason or "")
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.parametrize(
    ("plan", "tools", "reason"),
    [
        ("invented", ("ask",), "unregistered plan"),
        ("ask", ("invented",), "unregistered tool"),
        ("route_and_ask", ("ask", "routing"), "registered plan sequence"),
    ],
)
def test_policy_defensively_rechecks_registered_plan_shape(
    plan: str,
    tools: tuple[str, ...],
    reason: str,
) -> None:
    """The trust boundary should reject models built without validation."""
    decision = AgentDecision.model_construct(
        intent="test",
        selected_plan=plan,
        steps=tuple(
            AgentPlanningStep.model_construct(tool=tool, purpose=None)
            for tool in tools
        ),
        reason="test",
        confidence=0.9,
    )

    result = AgentDecisionPolicy(0.6, 2).evaluate(decision)

    assert result.accepted is False
    assert reason in (result.reason or "")


@pytest.mark.parametrize(
    "primary_outcome",
    [
        AgentPlanningError("invalid", code="invalid_json"),
        _decision(confidence=0.4),
    ],
)
def test_disabled_fallback_raises_without_calling_deterministic_planner(
    primary_outcome: AgentDecision | Exception,
) -> None:
    """Disabled fallback should surface orchestration failure immediately."""
    primary = FakePlanner(primary_outcome)
    fallback = FakePlanner(_decision())
    service = _fallback_service(primary, fallback, fallback_enabled=False)

    with pytest.raises(AgentPlanningServiceError, match="fallback is disabled"):
        service.create_plan("question")

    assert primary.calls == 1
    assert fallback.calls == 0


def test_double_failure_raises_with_exception_chaining() -> None:
    """No partial result should be returned when both planners fail."""
    primary_error = AgentPlanningError("invalid", code="invalid_json")
    fallback_error = RuntimeError("deterministic failure")
    service = _fallback_service(
        FakePlanner(primary_error),
        FakePlanner(fallback_error),
    )

    with pytest.raises(AgentPlanningServiceError) as captured:
        service.create_plan("question")

    assert captured.value.__cause__ is fallback_error
    assert fallback_error.__context__ is primary_error


def test_llm_factory_is_lazy_and_calls_provider_once() -> None:
    """LLM construction should occur on first use and only once per request."""
    provider = FakeProvider(
        '{"intent":"ask","selected_plan":"ask","steps":[{"tool":"ask"}],'
        '"reason":"Use grounded answering.","confidence":0.9}'
    )
    provider_factory_calls = 0

    def provider_factory() -> FakeProvider:
        nonlocal provider_factory_calls
        provider_factory_calls += 1
        return provider

    service = build_agent_planning_service(
        Settings(agent_planning_mode="llm", openai_api_key="unused"),
        provider_factory,
    )
    assert provider_factory_calls == 0

    result = service.create_plan("question")

    assert result.used_strategy == "llm"
    assert provider_factory_calls == 1
    assert provider.calls == 1
