"""Offline tests for deterministic and LLM agent planning."""

import json

import pytest

from app.agent.planner import (
    DeterministicAgentPlanner,
    LLMAgentPlanner,
    build_agent_planner,
)
from app.agent.planner_parser import AgentPlanningError, parse_agent_decision
from app.agent.planner_prompt import AgentPlanningPromptBuilder
from app.config import Settings
from app.services.llm_providers import LLMRequestError


class FakeProvider:
    """Return one deterministic response while recording planner prompts."""

    def __init__(self, response: str, fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.fail:
            raise LLMRequestError("fake provider failure")
        return self.response


def _decision_json(
    selected_plan: str,
    tools: tuple[str, ...],
    *,
    intent: str = "grounded_question",
    reason: str = "A short planning justification.",
    confidence: float = 0.9,
) -> str:
    return json.dumps(
        {
            "intent": intent,
            "selected_plan": selected_plan,
            "steps": [
                {"tool": tool, "purpose": f"Use {tool}"} for tool in tools
            ],
            "reason": reason,
            "confidence": confidence,
        }
    )


@pytest.mark.parametrize(
    ("selected_plan", "tools"),
    [
        ("ask", ("ask",)),
        ("route_and_ask", ("routing", "ask")),
        ("route_and_search", ("routing", "search")),
        ("collections", ("collections",)),
    ],
)
def test_llm_planner_parses_valid_registered_decisions(
    selected_plan: str, tools: tuple[str, ...]
) -> None:
    """Supported provider decisions should become validated Pydantic models."""
    provider = FakeProvider(_decision_json(selected_plan, tools))

    decision = LLMAgentPlanner(provider).create_plan("user request")

    assert decision.selected_plan == selected_plan
    assert tuple(step.tool for step in decision.steps) == tools
    assert decision.confidence == 0.9
    assert len(provider.calls) == 1


def test_markdown_fenced_json_is_unwrapped_defensively() -> None:
    """One complete optional JSON fence should parse consistently."""
    response = f"```json\n{_decision_json('ask', ('ask',))}\n```"

    decision = parse_agent_decision(response)

    assert decision.selected_plan == "ask"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("not-json", "invalid JSON"),
        ("", "empty response"),
        (
            _decision_json("ask", ("invented",)),
            "unknown agent tool",
        ),
        (
            _decision_json("invented", ("ask",)),
            "unknown agent plan",
        ),
        (
            _decision_json("ask", ("ask",), confidence=-0.1),
            "greater than or equal to 0",
        ),
        (
            _decision_json("ask", ("ask",), confidence=1.1),
            "less than or equal to 1",
        ),
        (
            _decision_json("route_and_ask", ("routing", "ask", "ask")),
            "at most 2",
        ),
    ],
)
def test_invalid_planner_outputs_raise_domain_error(
    response: str, message: str
) -> None:
    """Malformed or unsafe decisions should fail with explicit planner errors."""
    with pytest.raises(AgentPlanningError, match=message):
        parse_agent_decision(response)


def test_missing_required_field_raises_planning_error() -> None:
    """Incomplete JSON must not produce a partially valid decision."""
    payload = json.loads(_decision_json("ask", ("ask",)))
    del payload["reason"]

    with pytest.raises(AgentPlanningError, match="reason"):
        parse_agent_decision(json.dumps(payload))


@pytest.mark.parametrize("field", ["intent", "reason"])
def test_blank_required_text_raises_planning_error(field: str) -> None:
    """Required decision descriptions must contain meaningful text."""
    payload = json.loads(_decision_json("ask", ("ask",)))
    payload[field] = "   "

    with pytest.raises(AgentPlanningError, match=field):
        parse_agent_decision(json.dumps(payload))


def test_provider_failure_is_wrapped_without_fallback() -> None:
    """Sprint 1 should expose provider failure without executing or fallback."""
    planner = LLMAgentPlanner(FakeProvider("", fail=True))

    with pytest.raises(AgentPlanningError, match="planning request failed"):
        planner.create_plan("question")


def test_planning_prompt_is_bounded_json_only_and_not_an_answer_prompt() -> None:
    """The dedicated prompt should constrain tools, plans, and model behavior."""
    prompt = AgentPlanningPromptBuilder().build("How does authentication work?")
    system = prompt.system_prompt.lower()

    for tool in ("ask", "search", "routing", "collections"):
        assert tool in system
    for plan in (
        "ask",
        "search",
        "routing",
        "collections",
        "route_and_ask",
        "route_and_search",
    ):
        assert plan in system
    assert "not an answer generator" in system
    assert "do not answer" in system
    assert "raw json object only" in system
    assert "do not use markdown" in system
    assert "at most two steps" in system
    assert (
        "<user_query>How does authentication work?</user_query>"
        in prompt.user_prompt
    )


@pytest.mark.parametrize(
    ("query", "selected_plan", "tools"),
    [
        ("How is authentication implemented?", "ask", ("ask",)),
        ("Find authentication chunks", "search", ("search",)),
        (
            "Route this question and answer: how is authentication implemented?",
            "route_and_ask",
            ("routing", "ask"),
        ),
    ],
)
def test_deterministic_planner_preserves_existing_selection(
    query: str, selected_plan: str, tools: tuple[str, ...]
) -> None:
    """The deterministic planner wrapper should mirror current agent rules."""
    decision = DeterministicAgentPlanner().create_plan(query)

    assert decision.selected_plan == selected_plan
    assert tuple(step.tool for step in decision.steps) == tools
    assert decision.confidence == 1.0


def test_planner_factory_defaults_to_deterministic_without_provider() -> None:
    """Default construction must remain offline and backward compatible."""
    provider_calls = 0

    def provider_factory() -> FakeProvider:
        nonlocal provider_calls
        provider_calls += 1
        return FakeProvider(_decision_json("ask", ("ask",)))

    planner = build_agent_planner(Settings(), provider_factory)

    assert isinstance(planner, DeterministicAgentPlanner)
    assert provider_calls == 0


def test_llm_planner_factory_accepts_injected_provider() -> None:
    """LLM mode should remain independently testable without credentials."""
    provider = FakeProvider(_decision_json("collections", ("collections",)))
    planner = build_agent_planner(
        Settings(agent_planning_mode="llm"), lambda: provider
    )

    decision = planner.create_plan("List collections")

    assert isinstance(planner, LLMAgentPlanner)
    assert decision.selected_plan == "collections"
