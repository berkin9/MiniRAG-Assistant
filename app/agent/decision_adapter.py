"""Convert validated planning decisions into registered executable plans."""

from app.agent.definitions import (
    AGENT_PLAN_INPUT_MODES,
    AGENT_PLAN_RESULT_INTENTS,
    HARD_AGENT_MAX_STEPS,
    SUPPORTED_AGENT_PLANS,
    SUPPORTED_AGENT_TOOLS,
)
from app.agent.models import AgentPlan, AgentStep
from app.agent.planner_models import AgentDecision


class AgentExecutionPreparationError(ValueError):
    """Raised when a planning decision cannot be prepared for safe execution."""


class AgentDecisionAdapter:
    """Apply application-owned execution mappings to one validated decision."""

    def to_agent_plan(
        self,
        decision: AgentDecision,
        original_request: str,
    ) -> AgentPlan:
        """Build one exact registered plan without selecting or executing tools."""
        if not original_request.strip():
            raise AgentExecutionPreparationError(
                "Agent request must not be empty"
            )
        if len(decision.steps) > HARD_AGENT_MAX_STEPS:
            raise AgentExecutionPreparationError(
                "Agent decision exceeds the hard execution step limit"
            )

        expected_tools = SUPPORTED_AGENT_PLANS.get(decision.selected_plan)
        if expected_tools is None:
            raise AgentExecutionPreparationError(
                f"Agent plan is not registered: {decision.selected_plan}"
            )
        actual_tools = tuple(step.tool for step in decision.steps)
        unknown_tools = tuple(
            tool for tool in actual_tools if tool not in SUPPORTED_AGENT_TOOLS
        )
        if unknown_tools:
            raise AgentExecutionPreparationError(
                f"Agent tool is not registered: {unknown_tools[0]}"
            )
        if actual_tools != expected_tools:
            raise AgentExecutionPreparationError(
                f"Agent decision does not match registered plan: "
                f"{decision.selected_plan}"
            )

        input_modes = AGENT_PLAN_INPUT_MODES.get(decision.selected_plan)
        if (
            input_modes is None
            or len(input_modes) != len(expected_tools)
            or decision.selected_plan not in AGENT_PLAN_RESULT_INTENTS
        ):
            raise AgentExecutionPreparationError(
                f"Agent plan has no executable input mapping: "
                f"{decision.selected_plan}"
            )
        return AgentPlan(
            decision.selected_plan,
            tuple(
                AgentStep(tool, input_mode)
                for tool, input_mode in zip(
                    expected_tools,
                    input_modes,
                    strict=True,
                )
            ),
            decision.reason,
        )
