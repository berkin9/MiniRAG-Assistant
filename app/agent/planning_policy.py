"""Execution-readiness policy for structurally valid agent decisions."""

from dataclasses import dataclass

from app.agent.definitions import (
    HARD_AGENT_MAX_STEPS,
    SUPPORTED_AGENT_PLANS,
    SUPPORTED_AGENT_TOOLS,
)
from app.agent.planner_models import AgentDecision, AgentDecisionPolicyResult


@dataclass(frozen=True)
class AgentDecisionPolicy:
    """Apply runtime trust limits at the non-executing planning boundary."""

    minimum_confidence: float
    maximum_steps: int

    def __post_init__(self) -> None:
        """Reject policy limits outside the hard bounded-agent capabilities."""
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum confidence must be between 0 and 1")
        if not 1 <= self.maximum_steps <= HARD_AGENT_MAX_STEPS:
            raise ValueError(
                f"maximum steps must be between 1 and {HARD_AGENT_MAX_STEPS}"
            )

    def evaluate(self, decision: AgentDecision) -> AgentDecisionPolicyResult:
        """Return whether a validated decision is trusted for later execution."""
        if decision.confidence < self.minimum_confidence:
            return AgentDecisionPolicyResult(
                accepted=False,
                reason=(
                    f"LLM decision confidence {decision.confidence:.2f} is below "
                    f"the configured minimum {self.minimum_confidence:.2f}."
                ),
            )
        if len(decision.steps) > self.maximum_steps:
            return AgentDecisionPolicyResult(
                accepted=False,
                reason=(
                    f"LLM decision has {len(decision.steps)} steps; the configured "
                    f"maximum is {self.maximum_steps}."
                ),
            )
        expected_tools = SUPPORTED_AGENT_PLANS.get(decision.selected_plan)
        if expected_tools is None:
            return AgentDecisionPolicyResult(
                accepted=False,
                reason="LLM decision selected an unregistered plan.",
            )
        actual_tools = tuple(step.tool for step in decision.steps)
        if any(tool not in SUPPORTED_AGENT_TOOLS for tool in actual_tools):
            return AgentDecisionPolicyResult(
                accepted=False,
                reason="LLM decision selected an unregistered tool.",
            )
        if actual_tools != expected_tools:
            return AgentDecisionPolicyResult(
                accepted=False,
                reason="LLM decision does not match the registered plan sequence.",
            )
        return AgentDecisionPolicyResult(accepted=True)
