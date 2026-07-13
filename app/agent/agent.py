"""Bounded execution of one predefined one- or two-tool plan."""

import logging
from collections.abc import Callable, Iterable
from typing import Any

from app.agent.intent import classify_intent
from app.agent.models import (
    AgentExecutionContext,
    AgentPlan,
    AgentResponse,
    AgentStep,
    AgentStepResult,
    AgentToolResult,
    Intent,
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
        self._validate_registered_tools(plan)
        logger.info("Agent plan selected plan=%s steps=%s", plan.name, len(plan.steps))

        context = AgentExecutionContext()
        try:
            first = self._execute_step(plan.steps[0], request, context)
            step_results = (first,)
            if len(plan.steps) == 2:
                second = self._execute_step(plan.steps[1], request, context)
                step_results = (first, second)
        except Exception:
            logger.info("Agent execution completed plan=%s success=false", plan.name)
            raise

        logger.info("Agent execution completed plan=%s success=true", plan.name)
        final_result = step_results[-1].result
        return AgentResponse(
            request,
            intent,
            decision,
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
            raise ValueError(f"Agent tool is not configured: {missing[0]}")

    def _execute_step(
        self,
        step: AgentStep,
        request: str,
        context: AgentExecutionContext,
    ) -> AgentStepResult:
        """Execute one declared step with optional ephemeral context support."""
        logger.info("Agent tool execution tool=%s", step.tool)
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
