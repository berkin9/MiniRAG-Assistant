"""Bounded execution of one predefined one- or two-tool plan."""

import logging
from collections.abc import Callable, Iterable
from typing import Any

from app.agent.definitions import AGENT_PLAN_RESULT_INTENTS
from app.agent.decision_adapter import AgentExecutionPreparationError
from app.agent.intent import classify_intent
from app.agent.models import (
    AgentExecutionContext,
    AgentPlan,
    AgentResponse,
    AgentStep,
    AgentStepResult,
    AgentToolResult,
    Intent,
    ToolDecision,
)
from app.agent.plan_selector import PlanSelector, extract_question
from app.agent.tool_selector import ToolSelector
from app.agent.tools import (
    AgentTool,
    AskTool,
    CollectionsTool,
    RoutingTool,
    SearchTool,
)
from app.config import Settings

IntentClassifier = Callable[[str], Intent]
logger = logging.getLogger(__name__)


class Agent:
    """Classify a request and execute a strictly bounded predefined plan."""

    def __init__(
        self,
        tools: Iterable[AgentTool],
        classifier: IntentClassifier = classify_intent,
        selector: ToolSelector | None = None,
        plan_selector: PlanSelector | None = None,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._classifier = classifier
        self._selector = selector or ToolSelector()
        self._plan_selector = plan_selector or PlanSelector()

    def run(self, request: str) -> AgentResponse:
        """Execute one or two declared steps sequentially without retry."""
        intent = self._classifier(request)
        decision = self._selector.select(intent)
        plan = self._plan_selector.select(request, decision)
        return self.execute_plan(
            request,
            plan,
            intent=intent,
            decision=decision,
        )

    def execute_plan(
        self,
        request: str,
        plan: AgentPlan,
        *,
        intent: Intent | None = None,
        decision: ToolDecision | None = None,
    ) -> AgentResponse:
        """Execute one already prepared plan without selecting it again."""
        self._validate_registered_tools(plan)
        logger.info("Agent plan selected plan=%s steps=%s", plan.name, len(plan.steps))
        logger.info("agent_execution_started plan=%s", plan.name)

        context = AgentExecutionContext()
        try:
            first = self._execute_step(plan.steps[0], request, context, 1)
            step_results = (first,)
            if len(plan.steps) == 2:
                second = self._execute_step(plan.steps[1], request, context, 2)
                step_results = (first, second)
        except Exception as error:
            logger.info(
                "agent_execution_failed plan=%s error_type=%s",
                plan.name,
                type(error).__name__,
            )
            logger.info("Agent execution completed plan=%s success=false", plan.name)
            raise

        logger.info("agent_execution_completed plan=%s", plan.name)
        logger.info("Agent execution completed plan=%s success=true", plan.name)
        final_result = step_results[-1].result
        resolved_intent = intent or Intent(AGENT_PLAN_RESULT_INTENTS[plan.name])
        resolved_decision = decision or ToolDecision(
            resolved_intent.value,
            plan.reason,
        )
        return AgentResponse(
            request,
            resolved_intent,
            resolved_decision,
            final_result,
            plan,
            step_results,
        )

    def _validate_registered_tools(self, plan: AgentPlan) -> None:
        """Reject plans referencing tools absent from this agent instance."""
        missing = tuple(
            step.tool for step in plan.steps if step.tool not in self._tools
        )
        if missing:
            raise AgentExecutionPreparationError(
                f"Agent tool is not configured: {missing[0]}"
            )

    def _execute_step(
        self,
        step: AgentStep,
        request: str,
        context: AgentExecutionContext,
        step_index: int,
    ) -> AgentStepResult:
        """Execute one declared step with optional ephemeral context support."""
        logger.info(
            "agent_tool_started tool=%s step_index=%s",
            step.tool,
            step_index,
        )
        tool = self._tools[step.tool]
        tool_input = (
            extract_question(request)
            if step.input_mode == "extracted_question"
            else request
        )
        context_runner: Any = getattr(tool, "run_with_context", None)
        result: AgentToolResult
        if callable(context_runner):
            result = context_runner(tool_input, context)
        else:
            result = tool.run(tool_input)
        logger.info(
            "agent_tool_completed tool=%s step_index=%s",
            step.tool,
            step_index,
        )
        return AgentStepResult(step.tool, result)


def build_agent(settings: Settings) -> Agent:
    """Build the four supported tools over existing application services."""
    return Agent(
        (
            AskTool(settings),
            SearchTool(settings),
            CollectionsTool(settings),
            RoutingTool(settings),
        )
    )
