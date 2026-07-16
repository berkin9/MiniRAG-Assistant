"""Safe JSON parsing for provider-generated agent decisions."""

import json
import re

from pydantic import ValidationError

from app.agent.planner_models import AgentDecision


class AgentPlanningError(RuntimeError):
    """Raised when an agent planner cannot produce a valid decision."""

    def __init__(self, message: str, code: str = "planning_error") -> None:
        super().__init__(message)
        self.code = code


def parse_agent_decision(response: str) -> AgentDecision:
    """Parse raw or simply fenced JSON into a validated decision."""
    if not response or not response.strip():
        raise AgentPlanningError(
            "Agent planner returned an empty response",
            code="empty_response",
        )
    payload_text = _remove_json_fence(response.strip())
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise AgentPlanningError(
            "Agent planner returned invalid JSON",
            code="invalid_json",
        ) from error
    try:
        return AgentDecision.model_validate(payload)
    except ValidationError as error:
        first = error.errors(include_input=False)[0]
        location = ".".join(str(part) for part in first["loc"]) or "decision"
        raise AgentPlanningError(
            f"Invalid agent decision at {location}: {first['msg']}",
            code="invalid_decision",
        ) from error


def _remove_json_fence(response: str) -> str:
    """Defensively unwrap one complete JSON Markdown fence."""
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced.group(1).strip() if fenced else response
