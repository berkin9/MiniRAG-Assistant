"""Tests for deterministic intent selection and one-shot orchestration."""

from dataclasses import dataclass

import pytest

from app.agent.agent import Agent
from app.agent.intent import classify_intent
from app.agent.models import Intent
from app.agent.tool_selector import ToolSelector


@pytest.mark.parametrize(
    ("user_input", "expected"),
    [
        ("list collections", Intent.COLLECTIONS),
        ("Show the collections", Intent.COLLECTIONS),
        ("Which collection handles authentication?", Intent.ROUTING),
        ("Why technical?", Intent.ROUTING),
        ("Route this question", Intent.ROUTING),
        ("Explain the routing", Intent.ROUTING),
        ("Search for authentication details", Intent.SEARCH),
        ("Find the project deadline", Intent.SEARCH),
        ("Retrieve relevant chunks", Intent.SEARCH),
        ("Show the matching sources", Intent.SEARCH),
        ("How is authentication implemented?", Intent.ASK),
    ],
)
def test_intent_classification(user_input: str, expected: Intent) -> None:
    """Centralized rules should classify supported phrases deterministically."""
    assert classify_intent(user_input) is expected


@pytest.mark.parametrize("intent", list(Intent))
def test_tool_selector_returns_structured_decision(intent: Intent) -> None:
    """Selection should map intent to one tool without executing anything."""
    decision = ToolSelector().select(intent)

    assert decision.tool == intent.value
    assert decision.reason


@dataclass
class CountingTool:
    """Record executions while returning a harmless structured result."""

    name: str
    calls: int = 0

    def run(self, request: str) -> tuple[str, ...]:
        self.calls += 1
        return (request,)


def test_agent_executes_exactly_one_tool_and_defaults_to_ask() -> None:
    """Unknown input should select AskTool behavior with no execution loop."""
    tools = [CountingTool(intent.value) for intent in Intent]
    agent = Agent(tools)

    response = agent.run("Explain the deployment process")

    assert response.intent is Intent.ASK
    assert response.decision.tool == "ask"
    assert [tool.calls for tool in tools] == [1, 0, 0, 0]
