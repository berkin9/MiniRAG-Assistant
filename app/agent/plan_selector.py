"""Deterministic selection of predefined one- or two-step plans."""

import re
from dataclasses import dataclass

from app.agent.definitions import AGENT_PLAN_INPUT_MODES, SUPPORTED_AGENT_PLANS
from app.agent.models import AgentPlan, AgentStep, ToolDecision


@dataclass(frozen=True)
class CompoundPlanRule:
    """Signals that must both occur to choose one compound plan."""

    name: str
    routing_patterns: tuple[str, ...]
    action_patterns: tuple[str, ...]
    final_tool: str
    reason: str


ROUTING_PATTERNS = (
    r"\bwhich\s+collection\b",
    r"\broute\s+(?:this|that|the\s+(?:question|request))\b",
    r"\b(?:explain|show)\s+(?:the\s+)?rout(?:e|ing)\b",
)

COMPOUND_PLAN_RULES: tuple[CompoundPlanRule, ...] = (
    CompoundPlanRule(
        "route_and_search",
        ROUTING_PATTERNS,
        (
            r"\b(?:retrieve|show)\s+(?:the\s+)?(?:matching\s+|relevant\s+)?(?:chunks|sources)\b",
            r"\bsearch\s+for\b",
            r"\bthen\s+search\b",
        ),
        "search",
        "The request asks for routing information and retrieved chunks.",
    ),
    CompoundPlanRule(
        "route_and_ask",
        ROUTING_PATTERNS,
        (
            r"\b(?:then\s+)?answer(?:\s+the\s+question)?\b",
            r"\bwhat\s+(?:does|do)\s+(?:the\s+)?(?:documentation|documents|docs)\s+say\b",
            r"\btell\s+me\s+what\s+(?:the\s+)?(?:documentation|documents|docs)\s+say\b",
        ),
        "ask",
        "The request asks for routing information and a grounded answer.",
    ),
)


class PlanSelector:
    """Select only predefined plans without executing any tool."""

    def select(self, request: str, single_tool: ToolDecision) -> AgentPlan:
        """Prefer a clear compound match, otherwise retain single-tool behavior."""
        normalized = " ".join(request.casefold().split())
        for rule in COMPOUND_PLAN_RULES:
            if _matches_any(normalized, rule.routing_patterns) and _matches_any(
                normalized, rule.action_patterns
            ):
                return AgentPlan(
                    rule.name,
                    _registered_steps(rule.name),
                    rule.reason,
                )
        return AgentPlan(
            single_tool.tool,
            _registered_steps(single_tool.tool),
            single_tool.reason,
        )


def extract_question(request: str) -> str:
    """Extract text after an obvious command separator or keep the request."""
    match = re.search(
        r"(?:question|then\s+answer|then\s+search)\s*:\s*(.+)$",
        request,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    if ":" in request:
        prefix, candidate = request.rsplit(":", maxsplit=1)
        if candidate.strip() and re.search(
            r"\b(?:route|routing|answer|search|sources|chunks)\b",
            prefix,
            flags=re.IGNORECASE,
        ):
            return candidate.strip()
    return request


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether any centralized compound signal matches."""
    return any(re.search(pattern, text) for pattern in patterns)


def _registered_steps(plan_name: str) -> tuple[AgentStep, ...]:
    """Build steps from the application-owned plan input mapping."""
    return tuple(
        AgentStep(tool, input_mode)
        for tool, input_mode in zip(
            SUPPORTED_AGENT_PLANS[plan_name],
            AGENT_PLAN_INPUT_MODES[plan_name],
            strict=True,
        )
    )
