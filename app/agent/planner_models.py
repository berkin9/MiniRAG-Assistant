"""Validated structured output models for agent planning."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent.models import SUPPORTED_AGENT_PLANS, SUPPORTED_AGENT_TOOLS


class AgentPlanningStep(BaseModel):
    """One non-executed tool step proposed by an agent planner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    purpose: str | None = None

    @field_validator("tool")
    @classmethod
    def validate_tool(cls, value: str) -> str:
        """Reject tool names outside the existing bounded agent registry."""
        tool = value.strip()
        if tool not in SUPPORTED_AGENT_TOOLS:
            raise ValueError(f"unknown agent tool: {tool or '<empty>'}")
        return tool

    @field_validator("purpose")
    @classmethod
    def normalize_purpose(cls, value: str | None) -> str | None:
        """Normalize optional short step descriptions."""
        if value is None:
            return None
        purpose = value.strip()
        return purpose or None


class AgentDecision(BaseModel):
    """Validated planner decision that is not automatically executable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str
    selected_plan: str
    steps: tuple[AgentPlanningStep, ...] = Field(min_length=1, max_length=2)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("intent", "reason")
    @classmethod
    def require_text(cls, value: str) -> str:
        """Reject blank intent and high-level reasoning values."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("selected_plan")
    @classmethod
    def validate_plan_name(cls, value: str) -> str:
        """Reject plans outside the existing deterministic catalog."""
        plan = value.strip()
        if plan not in SUPPORTED_AGENT_PLANS:
            raise ValueError(f"unknown agent plan: {plan or '<empty>'}")
        return plan

    @model_validator(mode="after")
    def validate_plan_shape(self) -> "AgentDecision":
        """Require the exact registered tool sequence for the selected plan."""
        expected = SUPPORTED_AGENT_PLANS[self.selected_plan]
        actual = tuple(step.tool for step in self.steps)
        if actual != expected:
            raise ValueError(
                f"plan {self.selected_plan!r} requires tools: {', '.join(expected)}"
            )
        return self
