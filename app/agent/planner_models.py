"""Validated structured output models for agent planning."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent.definitions import (
    HARD_AGENT_MAX_STEPS,
    SUPPORTED_AGENT_PLANS,
    SUPPORTED_AGENT_TOOLS,
)


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
    steps: tuple[AgentPlanningStep, ...] = Field(
        min_length=1,
        max_length=HARD_AGENT_MAX_STEPS,
    )
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


class AgentDecisionPolicyResult(BaseModel):
    """Immutable acceptance outcome from the execution-readiness policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool
    reason: str | None = None


class AgentPlanningResult(BaseModel):
    """Safe structured result from one orchestrated planning request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: AgentDecision
    requested_strategy: Literal["deterministic", "llm"]
    used_strategy: Literal["deterministic", "llm"]
    fallback_used: bool
    fallback_reason: str | None = None
    primary_error_type: str | None = None
    policy_rejection_reason: str | None = None

    @model_validator(mode="after")
    def validate_fallback_metadata(self) -> "AgentPlanningResult":
        """Keep fallback flags, strategy, and safe reason consistent."""
        if self.fallback_used:
            if (
                self.requested_strategy != "llm"
                or self.used_strategy != "deterministic"
            ):
                raise ValueError(
                    "fallback must replace LLM planning with deterministic"
                )
            if not self.fallback_reason or not self.fallback_reason.strip():
                raise ValueError("fallback_reason is required when fallback is used")
        elif self.fallback_reason is not None:
            raise ValueError("fallback_reason requires fallback_used=true")
        elif self.requested_strategy != self.used_strategy:
            raise ValueError("strategy changes require fallback_used=true")
        elif self.primary_error_type is not None:
            raise ValueError("primary_error_type requires fallback_used=true")
        elif self.policy_rejection_reason is not None:
            raise ValueError("policy_rejection_reason requires fallback_used=true")
        return self
