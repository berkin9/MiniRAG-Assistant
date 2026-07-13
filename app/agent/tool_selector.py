"""Map classified intent to one tool without executing it."""

from app.agent.models import Intent, ToolDecision


TOOL_DECISIONS: dict[Intent, ToolDecision] = {
    Intent.ASK: ToolDecision("ask", "Default grounded question answering."),
    Intent.SEARCH: ToolDecision("search", "The request asks to retrieve chunks."),
    Intent.COLLECTIONS: ToolDecision(
        "collections", "The request asks for configured collections."
    ),
    Intent.ROUTING: ToolDecision(
        "routing", "The request asks how a question would be routed."
    ),
}


class ToolSelector:
    """Choose exactly one registered tool for a classified intent."""

    def select(self, intent: Intent) -> ToolDecision:
        """Return a structured selection without running the tool."""
        return TOOL_DECISIONS[intent]
