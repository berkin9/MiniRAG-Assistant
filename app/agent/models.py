"""Typed models shared by the agent components."""

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from app.services.routing import RoutingDecision
from app.services.runtime import RoutedAnswer, RoutedSearch


class Intent(str, Enum):
    """Supported deterministic user-request intents."""

    ASK = "ask"
    SEARCH = "search"
    COLLECTIONS = "collections"
    ROUTING = "routing"


@dataclass(frozen=True)
class ToolDecision:
    """Explain which single tool should handle a request."""

    tool: str
    reason: str


AgentToolResult: TypeAlias = (
    RoutedAnswer | RoutedSearch | tuple[str, ...] | RoutingDecision
)


@dataclass(frozen=True)
class AgentResponse:
    """Result of one intent decision and one tool execution."""

    request: str
    intent: Intent
    decision: ToolDecision
    result: AgentToolResult
