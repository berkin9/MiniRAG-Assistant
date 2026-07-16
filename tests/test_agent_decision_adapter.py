"""Tests for preparing validated decisions as executable agent plans."""

import pytest

from app.agent.decision_adapter import (
    AgentDecisionAdapter,
    AgentExecutionPreparationError,
)
from app.agent.planner_models import AgentDecision, AgentPlanningStep


def _decision(plan: str, tools: tuple[str, ...]) -> AgentDecision:
    return AgentDecision(
        intent="planner metadata only",
        selected_plan=plan,
        steps=tuple(
            AgentPlanningStep(tool=tool, purpose="ignored planner purpose")
            for tool in tools
        ),
        reason="Short planning reason.",
        confidence=0.9,
    )


@pytest.mark.parametrize(
    ("plan_name", "tools", "input_modes"),
    [
        ("ask", ("ask",), ("original_request",)),
        ("search", ("search",), ("original_request",)),
        ("collections", ("collections",), ("original_request",)),
        ("routing", ("routing",), ("original_request",)),
        (
            "route_and_ask",
            ("routing", "ask"),
            ("extracted_question", "extracted_question"),
        ),
        (
            "route_and_search",
            ("routing", "search"),
            ("extracted_question", "extracted_question"),
        ),
    ],
)
def test_registered_decision_converts_with_exact_application_owned_inputs(
    plan_name: str,
    tools: tuple[str, ...],
    input_modes: tuple[str, ...],
) -> None:
    """All fixed plans should preserve order and predefined input semantics."""
    plan = AgentDecisionAdapter().to_agent_plan(
        _decision(plan_name, tools),
        "Original user request",
    )

    assert plan.name == plan_name
    assert tuple(step.tool for step in plan.steps) == tools
    assert tuple(step.input_mode for step in plan.steps) == input_modes
    assert plan.reason == "Short planning reason."


@pytest.mark.parametrize(
    ("plan_name", "tools", "message"),
    [
        ("invented", ("ask",), "plan is not registered"),
        ("ask", ("invented",), "tool is not registered"),
        ("route_and_ask", ("ask", "routing"), "does not match"),
        ("route_and_ask", ("routing", "ask", "ask"), "step limit"),
    ],
)
def test_adapter_rejects_decisions_that_bypass_model_validation(
    plan_name: str,
    tools: tuple[str, ...],
    message: str,
) -> None:
    """The final preparation boundary must not silently repair invalid models."""
    decision = AgentDecision.model_construct(
        intent="untrusted",
        selected_plan=plan_name,
        steps=tuple(
            AgentPlanningStep.model_construct(tool=tool, purpose=None)
            for tool in tools
        ),
        reason="test",
        confidence=0.9,
    )

    with pytest.raises(AgentExecutionPreparationError, match=message):
        AgentDecisionAdapter().to_agent_plan(decision, "request")


def test_adapter_rejects_empty_request_before_execution() -> None:
    """A blank request should never reach a tool."""
    with pytest.raises(AgentExecutionPreparationError, match="must not be empty"):
        AgentDecisionAdapter().to_agent_plan(_decision("ask", ("ask",)), "  ")
