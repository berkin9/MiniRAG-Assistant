"""Human-editable dataset models and JSON loading."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic import model_validator

from app.agent.definitions import SUPPORTED_AGENT_PLANS, SUPPORTED_AGENT_TOOLS
from app.agent_limits import HARD_AGENT_MAX_STEPS


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation dataset cannot be loaded safely."""


class EvaluationCase(BaseModel):
    """One expected planning outcome for a benchmark query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    query: str
    expected_plan: str
    expected_tools: tuple[str, ...] = Field(
        min_length=1,
        max_length=HARD_AGENT_MAX_STEPS,
    )
    description: str
    expected_collection: str | None = None
    notes: str | None = None

    @field_validator("id", "query", "description")
    @classmethod
    def require_text(cls, value: str) -> str:
        """Reject blank identifiers, queries, and descriptions."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("expected_plan")
    @classmethod
    def validate_plan(cls, value: str) -> str:
        """Require one centrally registered plan name."""
        plan = value.strip()
        if plan not in SUPPORTED_AGENT_PLANS:
            raise ValueError(f"unknown agent plan: {plan or '<empty>'}")
        return plan

    @field_validator("expected_tools")
    @classmethod
    def validate_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject tools outside the central registry."""
        unknown = tuple(tool for tool in value if tool not in SUPPORTED_AGENT_TOOLS)
        if unknown:
            raise ValueError(f"unknown agent tool: {unknown[0]}")
        return value

    @model_validator(mode="after")
    def validate_plan_shape(self) -> "EvaluationCase":
        """Keep expected tools aligned with the registered plan definition."""
        if self.expected_tools != SUPPORTED_AGENT_PLANS[self.expected_plan]:
            raise ValueError(
                f"expected tools do not match plan: {self.expected_plan}"
            )
        return self


class EvaluationDataset(BaseModel):
    """Immutable collection of uniquely identified benchmark cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cases: tuple[EvaluationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "EvaluationDataset":
        """Require stable unique case identifiers for failure reports."""
        ids = tuple(case.id for case in self.cases)
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation case ids must be unique")
        return self


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate one UTF-8 JSON evaluation dataset."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise EvaluationDatasetError(
            f"Could not read evaluation dataset: {source}"
        ) from error
    except json.JSONDecodeError as error:
        raise EvaluationDatasetError(
            f"Evaluation dataset contains invalid JSON: {source}"
        ) from error
    try:
        return EvaluationDataset.model_validate(payload)
    except ValidationError as error:
        first = error.errors(include_input=False)[0]
        location = ".".join(str(part) for part in first["loc"]) or "dataset"
        raise EvaluationDatasetError(
            f"Invalid evaluation dataset at {location}: {first['msg']}"
        ) from error
