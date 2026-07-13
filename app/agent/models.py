"""Typed models shared by the agent components."""

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from app.services.routing import RoutingDecision
from app.services.runtime import RoutedAnswer, RoutedSearch

SUPPORTED_AGENT_TOOLS = frozenset({"ask", "search", "collections", "routing"})
SUPPORTED_INPUT_MODES = frozenset({"original_request", "extracted_question"})
SUPPORTED_AGENT_PLANS: dict[str, tuple[str, ...]] = {
    "ask": ("ask",),
    "search": ("search",),
    "collections": ("collections",),
    "routing": ("routing",),
    "route_and_ask": ("routing", "ask"),
    "route_and_search": ("routing", "search"),
}


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


@dataclass(frozen=True)
class AgentStep:
    """One predefined tool invocation in a bounded plan."""

    tool: str
    input_mode: str = "original_request"

    def __post_init__(self) -> None:
        """Reject unknown tools and unsupported input behavior."""
        if self.tool not in SUPPORTED_AGENT_TOOLS:
            raise ValueError(f"Unsupported agent tool: {self.tool}")
        if self.input_mode not in SUPPORTED_INPUT_MODES:
            raise ValueError(f"Unsupported agent input mode: {self.input_mode}")


@dataclass(frozen=True)
class AgentPlan:
    """Immutable predefined plan containing one or two steps."""

    name: str
    steps: tuple[AgentStep, ...]
    reason: str

    def __post_init__(self) -> None:
        """Enforce the strict execution bound."""
        if not self.steps:
            raise ValueError("Agent plans must contain at least one step")
        if len(self.steps) > 2:
            raise ValueError("Agent plans cannot contain more than two steps")
        expected_tools = SUPPORTED_AGENT_PLANS.get(self.name)
        if expected_tools is None:
            raise ValueError(f"Unsupported agent plan: {self.name}")
        actual_tools = tuple(step.tool for step in self.steps)
        if actual_tools != expected_tools:
            raise ValueError(
                f"Agent plan {self.name!r} requires tools: "
                f"{', '.join(expected_tools)}"
            )


@dataclass(frozen=True)
class AgentStepResult:
    """Structured result from one completed plan step."""

    tool: str
    result: "AgentToolResult"

    def __post_init__(self) -> None:
        """Reject results attributed to invented tools."""
        if self.tool not in SUPPORTED_AGENT_TOOLS:
            raise ValueError(f"Unsupported agent tool result: {self.tool}")


AgentToolResult: TypeAlias = (
    RoutedAnswer | RoutedSearch | tuple[str, ...] | RoutingDecision
)


@dataclass(frozen=True)
class AgentResponse:
    """Backward-compatible result of a bounded agent execution."""

    request: str
    intent: Intent
    decision: ToolDecision
    result: AgentToolResult
    plan: AgentPlan | None = None
    steps: tuple[AgentStepResult, ...] = ()

    def __post_init__(self) -> None:
        """Synthesize one-step metadata for legacy response construction."""
        if self.plan is None:
            object.__setattr__(
                self,
                "plan",
                AgentPlan(
                    self.decision.tool,
                    (AgentStep(self.decision.tool),),
                    self.decision.reason,
                ),
            )
        plan = self.plan
        if plan is None:
            raise ValueError("Agent response requires a plan")
        if not self.steps:
            if len(plan.steps) != 1:
                raise ValueError("Multi-step responses require ordered step results")
            object.__setattr__(
                self,
                "steps",
                (AgentStepResult(self.decision.tool, self.result),),
            )
        result_tools = tuple(step.tool for step in self.steps)
        plan_tools = tuple(step.tool for step in plan.steps)
        if result_tools != plan_tools:
            raise ValueError("Agent step results must match the selected plan")

    @property
    def final_result(self) -> AgentToolResult:
        """Return the last completed step while preserving `.result`."""
        return self.steps[-1].result


@dataclass
class AgentExecutionContext:
    """Ephemeral state shared only within one Agent.run call."""

    selected_collection: str | None = None
