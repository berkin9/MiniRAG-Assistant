"""Deterministic and LLM implementations of the agent planner contract."""

from collections.abc import Callable
from typing import Protocol

from app.agent.intent import classify_intent
from app.agent.planner_models import AgentDecision, AgentPlanningStep
from app.agent.planner_parser import AgentPlanningError, parse_agent_decision
from app.agent.planner_prompt import AgentPlanningPromptBuilder
from app.agent.plan_selector import PlanSelector
from app.agent.tool_selector import ToolSelector
from app.config import Settings
from app.services.llm_providers import (
    LLMProvider,
    LLMProviderError,
    build_agent_planning_provider,
)


class AgentPlanner(Protocol):
    """Provider-independent synchronous agent planning contract."""

    def create_plan(self, query: str) -> AgentDecision:
        """Create a validated decision without executing any tool."""


class DeterministicAgentPlanner:
    """Expose the existing deterministic selection as a planner decision."""

    def __init__(
        self,
        tool_selector: ToolSelector | None = None,
        plan_selector: PlanSelector | None = None,
    ) -> None:
        self._tool_selector = tool_selector or ToolSelector()
        self._plan_selector = plan_selector or PlanSelector()

    def create_plan(self, query: str) -> AgentDecision:
        """Map current deterministic intent and plan rules without execution."""
        intent = classify_intent(query)
        tool_decision = self._tool_selector.select(intent)
        plan = self._plan_selector.select(query, tool_decision)
        return AgentDecision(
            intent=intent.value,
            selected_plan=plan.name,
            steps=tuple(AgentPlanningStep(tool=step.tool) for step in plan.steps),
            reason=plan.reason,
            confidence=1.0,
        )


class LLMAgentPlanner:
    """Request and validate a plan through an injected LLM provider."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: AgentPlanningPromptBuilder | None = None,
    ) -> None:
        self._provider = provider
        self._prompt_builder = prompt_builder or AgentPlanningPromptBuilder()

    def create_plan(self, query: str) -> AgentDecision:
        """Call the provider once and parse its JSON without executing tools."""
        prompt = self._prompt_builder.build(query)
        try:
            response = self._provider.generate(
                prompt.system_prompt, prompt.user_prompt
            )
        except LLMProviderError as error:
            raise AgentPlanningError("LLM agent planning request failed") from error
        return parse_agent_decision(response)


def build_agent_planner(
    settings: Settings,
    provider_factory: Callable[[], LLMProvider] | None = None,
) -> AgentPlanner:
    """Build the configured planner without connecting it to execution."""
    if settings.agent_planning_mode == "deterministic":
        return DeterministicAgentPlanner()
    factory = provider_factory or (lambda: build_agent_planning_provider(settings))
    return LLMAgentPlanner(factory())
