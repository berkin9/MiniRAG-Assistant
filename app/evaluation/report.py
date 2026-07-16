"""Structured benchmark reports plus JSON and CSV export."""

import csv
import json
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetricStatistics(BaseModel):
    """Descriptive values shared by confidence and latency metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    average: float
    minimum: float
    maximum: float
    median: float


class ConfidenceCalibrationBucket(BaseModel):
    """Correctness within one confidence interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    cases: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)


class EvaluationCaseResult(BaseModel):
    """Observed planning outcome for one evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    query: str
    expected_plan: str
    actual_plan: str | None
    expected_tools: tuple[str, ...]
    actual_tools: tuple[str, ...] | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    fallback_used: bool
    latency_ms: float = Field(ge=0.0)
    plan_correct: bool
    tools_correct: bool
    passed: bool
    requested_strategy: str
    used_strategy: str | None
    error_type: str | None = None


class BenchmarkReport(BaseModel):
    """Complete aggregate and per-case benchmark result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    planner: Literal["deterministic", "llm"]
    timestamp: datetime
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    plan_accuracy: float = Field(ge=0.0, le=1.0)
    tool_accuracy: float = Field(ge=0.0, le=1.0)
    fallback_count: int = Field(ge=0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    confidence_statistics: MetricStatistics | None
    latency_statistics_ms: MetricStatistics | None
    deterministic_planner_calls: int = Field(ge=0)
    llm_planner_calls: int = Field(ge=0)
    tool_executions: int = Field(default=0, ge=0)
    selected_plan_distribution: dict[str, int]
    confidence_calibration: tuple[ConfidenceCalibrationBucket, ...]
    case_results: tuple[EvaluationCaseResult, ...]

    @model_validator(mode="after")
    def validate_case_counts(self) -> "BenchmarkReport":
        """Keep aggregate counts aligned with included case results."""
        if self.total_cases != len(self.case_results):
            raise ValueError("total_cases must match case_results")
        if self.passed_cases + self.failed_cases != self.total_cases:
            raise ValueError("passed and failed cases must equal total cases")
        return self

    @property
    def failures(self) -> tuple[EvaluationCaseResult, ...]:
        """Return every failed or errored case without stopping the benchmark."""
        return tuple(result for result in self.case_results if not result.passed)


def benchmark_report_payload(report: BenchmarkReport) -> dict[str, object]:
    """Build a stable machine-readable report without internal provider data."""
    confidence = (
        report.confidence_statistics.model_dump()
        if report.confidence_statistics
        else None
    )
    latency = (
        report.latency_statistics_ms.model_dump()
        if report.latency_statistics_ms
        else None
    )
    return {
        "planner": report.planner,
        "timestamp": report.timestamp.isoformat(),
        "summary": {
            "total_cases": report.total_cases,
            "passed_cases": report.passed_cases,
            "failed_cases": report.failed_cases,
        },
        "metrics": {
            "plan_accuracy": report.plan_accuracy,
            "tool_accuracy": report.tool_accuracy,
            "fallback_count": report.fallback_count,
            "fallback_rate": report.fallback_rate,
            "confidence": confidence,
            "latency_ms": latency,
            "deterministic_planner_calls": report.deterministic_planner_calls,
            "llm_planner_calls": report.llm_planner_calls,
            "tool_executions": report.tool_executions,
            "selected_plan_distribution": report.selected_plan_distribution,
            "confidence_calibration": [
                bucket.model_dump() for bucket in report.confidence_calibration
            ],
        },
        "failures": [failure.model_dump() for failure in report.failures],
        "cases": [result.model_dump() for result in report.case_results],
    }


def benchmark_report_json(report: BenchmarkReport) -> str:
    """Serialize one benchmark report as readable JSON."""
    return json.dumps(benchmark_report_payload(report), indent=2) + "\n"


def benchmark_report_csv(report: BenchmarkReport) -> str:
    """Serialize per-case benchmark results as CSV."""
    output = StringIO(newline="")
    fields = (
        "id",
        "query",
        "expected_plan",
        "actual_plan",
        "expected_tools",
        "actual_tools",
        "confidence",
        "fallback",
        "latency_ms",
        "passed",
        "error_type",
    )
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for result in report.case_results:
        writer.writerow(
            {
                "id": result.id,
                "query": result.query,
                "expected_plan": result.expected_plan,
                "actual_plan": result.actual_plan or "",
                "expected_tools": "|".join(result.expected_tools),
                "actual_tools": "|".join(result.actual_tools or ()),
                "confidence": "" if result.confidence is None else result.confidence,
                "fallback": result.fallback_used,
                "latency_ms": result.latency_ms,
                "passed": result.passed,
                "error_type": result.error_type or "",
            }
        )
    return output.getvalue()


def write_json_report(report: BenchmarkReport, path: str | Path) -> None:
    """Write a JSON benchmark report as UTF-8."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(benchmark_report_json(report), encoding="utf-8")


def write_csv_report(report: BenchmarkReport, path: str | Path) -> None:
    """Write a CSV benchmark report as UTF-8."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(benchmark_report_csv(report), encoding="utf-8")
