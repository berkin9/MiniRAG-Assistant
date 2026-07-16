"""Controlled execution of validated agent-planning results."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from app.agent.agent import build_agent
from app.agent.decision_adapter import (
    AgentDecisionAdapter,
    AgentExecutionPreparationError,
)
from app.agent.models import AgentPlan, AgentResponse
from app.agent.planner_models import AgentPlanningResult
from app.agent.planning_service import (
    AgentPlanningService,
    build_agent_planning_service,
)
from app.config import Settings
from app.services.llm_providers import LLMProvider

logger = logging.getLogger(__name__)
ProviderFactory = Callable[[], LLMProvider]


class PlanningService(Protocol):
    """Planning dependency required by controlled execution."""

    def create_plan(self, query: str) -> AgentPlanningResult:
        """Return one validated planning result."""


class PlanExecutor(Protocol):
    """Existing bounded executor operation used by planned execution."""

    def execute_plan(self, request: str, plan: AgentPlan) -> AgentResponse:
        """Execute one prebuilt plan without selecting another plan."""


@dataclass(frozen=True)
class PlannedAgentResult:
    """Planning metadata paired with the existing execution response."""

    planning: AgentPlanningResult
    execution: AgentResponse

    def __post_init__(self) -> None:
        """Ensure execution exactly reflects the accepted planning decision."""
        plan = self.execution.plan
        if plan is None or plan.name != self.planning.decision.selected_plan:
            raise ValueError("Execution plan does not match the planning decision")
        planned_tools = tuple(
            step.tool for step in self.planning.decision.steps
        )
        executed_tools = tuple(step.tool for step in self.execution.steps)
        if executed_tools != planned_tools:
            raise ValueError("Executed tools do not match the planning decision")

    @property
    def selected_plan(self) -> str:
        """Return the single plan used for execution."""
        return self.planning.decision.selected_plan

    @property
    def requested_strategy(self) -> str:
        """Return the configured planning strategy."""
        return self.planning.requested_strategy

    @property
    def used_strategy(self) -> str:
        """Return the strategy that produced the executed decision."""
        return self.planning.used_strategy

    @property
    def fallback_used(self) -> bool:
        """Return whether deterministic planning replaced LLM planning."""
        return self.planning.fallback_used

    @property
    def executed_steps(self) -> int:
        """Return the number of completed tool steps."""
        return len(self.execution.steps)

    @property
    def executed_tools(self) -> tuple[str, ...]:
        """Return completed tool names in exact execution order."""
        return tuple(step.tool for step in self.execution.steps)


class PlannedAgentExecutionError(RuntimeError):
    """Raised when tool execution fails after planning has completed."""


class PlannedAgentService:
    """Plan once, prepare once, and execute once through the bounded agent."""

    def __init__(
        self,
        planning_service: PlanningService,
        decision_adapter: AgentDecisionAdapter,
        executor: PlanExecutor,
    ) -> None:
        self._planning_service = planning_service
        self._decision_adapter = decision_adapter
        self._executor = executor

    def run(self, request: str) -> PlannedAgentResult:
        """Execute exactly one validated plan without retries or replanning."""
        if not request.strip():
            raise AgentExecutionPreparationError(
                "Agent request must not be empty"
            )
        logger.info("planned_agent_started")
        planning = self._planning_service.create_plan(request)
        plan = self._decision_adapter.to_agent_plan(
            planning.decision,
            request,
        )
        logger.info(
            "agent_plan_prepared requested_strategy=%s used_strategy=%s "
            "plan=%s confidence=%.2f fallback_used=%s",
            planning.requested_strategy,
            planning.used_strategy,
            plan.name,
            planning.decision.confidence,
            str(planning.fallback_used).lower(),
        )
        try:
            execution = self._executor.execute_plan(request, plan)
        except AgentExecutionPreparationError:
            raise
        except Exception as error:
            logger.info(
                "planned_agent_execution_failed plan=%s error_type=%s",
                plan.name,
                type(error).__name__,
            )
            raise PlannedAgentExecutionError(
                f"Agent execution failed for plan: {plan.name}"
            ) from error
        logger.info(
            "planned_agent_execution_completed plan=%s steps=%s",
            plan.name,
            len(execution.steps),
        )
        return PlannedAgentResult(planning, execution)


def build_planned_agent_service(
    settings: Settings,
    planning_provider_factory: ProviderFactory | None = None,
) -> PlannedAgentService:
    """Build one planner and the existing shared bounded tool executor."""
    planning_service: AgentPlanningService = build_agent_planning_service(
        settings,
        planning_provider_factory,
    )
    return PlannedAgentService(
        planning_service,
        AgentDecisionAdapter(),
        build_agent(settings),
    )
