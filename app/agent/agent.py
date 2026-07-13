"""One-shot agent orchestration with no planning or execution loop."""

from collections.abc import Callable, Iterable

from app.agent.intent import classify_intent
from app.agent.models import AgentResponse, Intent
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


class Agent:
    """Classify a request, select one tool, and execute it exactly once."""

    def __init__(
        self,
        tools: Iterable[AgentTool],
        classifier: IntentClassifier = classify_intent,
        selector: ToolSelector | None = None,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._classifier = classifier
        self._selector = selector or ToolSelector()

    def run(self, request: str) -> AgentResponse:
        """Execute the single tool chosen for the classified request."""
        intent = self._classifier(request)
        decision = self._selector.select(intent)
        try:
            tool = self._tools[decision.tool]
        except KeyError as error:
            raise ValueError(f"Agent tool is not configured: {decision.tool}") from error
        result = tool.run(request)
        return AgentResponse(request, intent, decision, result)


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
