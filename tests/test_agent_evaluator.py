"""Offline tests for planning benchmark execution and aggregation."""

from datetime import datetime, timezone

import pytest

from app.agent.planner_models import (
    AgentDecision,
    AgentPlanningResult,
    AgentPlanningStep,
)
from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.evaluator import AgentEvaluator


def _case(case_id: str, plan: str, tools: tuple[str, ...]) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        query=f"query {case_id}",
        expected_plan=plan,
        expected_tools=tools,
        description="test case",
    )


def _decision(
    plan: str,
    tools: tuple[str, ...],
    confidence: float,
) -> AgentDecision:
    return AgentDecision(
        intent="test",
        selected_plan=plan,
        steps=tuple(AgentPlanningStep(tool=tool) for tool in tools),
        reason="test decision",
        confidence=confidence,
    )


def _llm_result(
    decision: AgentDecision,
    *,
    fallback: bool = False,
    primary: AgentDecision | None = None,
) -> AgentPlanningResult:
    return AgentPlanningResult(
        decision=decision,
        requested_strategy="llm",
        used_strategy="deterministic" if fallback else "llm",
        fallback_used=fallback,
        fallback_reason="Safe fallback." if fallback else None,
        primary_decision=primary,
    )


class FakePlanningService:
    """Return or raise queued planning outcomes while counting all cases."""

    def __init__(self, outcomes: list[AgentPlanningResult | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def create_plan(self, query: str) -> AgentPlanningResult:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClock:
    """Yield deterministic monotonic timestamps."""

    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def test_evaluator_calculates_accuracy_fallback_confidence_latency_and_usage() -> None:
    """One run should aggregate planner quality without executing any tools."""
    dataset = EvaluationDataset(
        cases=(
            _case("one", "ask", ("ask",)),
            _case("two", "search", ("search",)),
            _case("three", "collections", ("collections",)),
        )
    )
    correct = _decision("ask", ("ask",), 0.95)
    incorrect = _decision("ask", ("ask",), 0.80)
    low_confidence = _decision("collections", ("collections",), 0.40)
    fallback = _decision("collections", ("collections",), 1.0)
    service = FakePlanningService(
        [
            _llm_result(correct),
            _llm_result(incorrect),
            _llm_result(fallback, fallback=True, primary=low_confidence),
        ]
    )
    timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)

    report = AgentEvaluator(
        service,
        "llm",
        FakeClock((0.00, 0.01, 0.02, 0.05, 0.10, 0.11)),
        lambda: timestamp,
    ).evaluate(dataset)

    assert service.calls == 3
    assert report.timestamp == timestamp
    assert report.total_cases == 3
    assert report.passed_cases == 2
    assert report.plan_accuracy == pytest.approx(2 / 3)
    assert report.tool_accuracy == pytest.approx(2 / 3)
    assert report.fallback_count == 1
    assert report.fallback_rate == pytest.approx(1 / 3)
    assert report.deterministic_planner_calls == 1
    assert report.llm_planner_calls == 3
    assert report.tool_executions == 0
    assert report.selected_plan_distribution == {"ask": 2, "collections": 1}
    assert report.confidence_statistics is not None
    assert report.confidence_statistics.average == pytest.approx(0.7166667)
    assert report.confidence_statistics.minimum == 0.40
    assert report.confidence_statistics.maximum == 0.95
    assert report.confidence_statistics.median == 0.80
    assert report.latency_statistics_ms is not None
    assert report.latency_statistics_ms.average == pytest.approx(50 / 3)
    assert report.latency_statistics_ms.minimum == pytest.approx(10)
    assert report.latency_statistics_ms.maximum == pytest.approx(30)
    assert [bucket.accuracy for bucket in report.confidence_calibration] == [
        1.0,
        0.0,
        1.0,
    ]
    assert [failure.id for failure in report.failures] == ["two"]
    assert report.failures[0].expected_plan == "search"
    assert report.failures[0].actual_plan == "ask"


def test_evaluator_records_safe_error_and_continues_remaining_cases() -> None:
    """A failed case must not stop later planner evaluation."""
    dataset = EvaluationDataset(
        cases=(
            _case("failed", "ask", ("ask",)),
            _case("passed", "search", ("search",)),
        )
    )
    decision = _decision("search", ("search",), 1.0)
    deterministic = AgentPlanningResult(
        decision=decision,
        requested_strategy="deterministic",
        used_strategy="deterministic",
        fallback_used=False,
    )
    service = FakePlanningService([RuntimeError("secret details"), deterministic])

    report = AgentEvaluator(
        service,
        "deterministic",
        FakeClock((0.0, 0.01, 0.02, 0.03)),
    ).evaluate(dataset)

    assert service.calls == 2
    assert report.failed_cases == 1
    assert report.passed_cases == 1
    assert report.case_results[0].error_type == "RuntimeError"
    assert "secret details" not in str(report.case_results[0])
    assert report.case_results[1].passed is True
