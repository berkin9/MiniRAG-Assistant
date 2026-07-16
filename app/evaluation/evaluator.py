"""Read-only benchmark runner for agent-planning services."""

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Literal, Protocol

from app.agent.planner_models import AgentDecision, AgentPlanningResult
from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.metrics import (
    calculate_accuracy,
    calculate_confidence_calibration,
    calculate_distribution,
    calculate_rate,
    calculate_statistics,
)
from app.evaluation.report import BenchmarkReport, EvaluationCaseResult

logger = logging.getLogger(__name__)
PlannerStrategy = Literal["deterministic", "llm"]


class PlanningService(Protocol):
    """Minimal planning-only dependency used by the evaluator."""

    def create_plan(self, query: str) -> AgentPlanningResult:
        """Return one planning result without executing tools."""


class AgentEvaluator:
    """Evaluate every case while isolating failures and measuring planning time."""

    def __init__(
        self,
        planning_service: PlanningService,
        planner: PlannerStrategy,
        clock: Callable[[], float] = perf_counter,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if planner not in {"deterministic", "llm"}:
            raise ValueError(f"Unsupported benchmark planner: {planner}")
        self._planning_service = planning_service
        self._planner = planner
        self._clock = clock
        self._now = now or (lambda: datetime.now(timezone.utc))

    def evaluate(self, dataset: EvaluationDataset) -> BenchmarkReport:
        """Evaluate all cases without invoking the agent tool executor."""
        logger.info(
            "benchmark_started planner=%s cases=%s",
            self._planner,
            len(dataset.cases),
        )
        results: list[EvaluationCaseResult] = []
        deterministic_calls = 0
        llm_calls = 0

        for case in dataset.cases:
            started = self._clock()
            try:
                planning = self._planning_service.create_plan(case.query)
            except Exception as error:
                latency_ms = self._elapsed_ms(started)
                deterministic_calls += self._planner == "deterministic"
                llm_calls += self._planner == "llm"
                result = self._failed_case(case, latency_ms, type(error).__name__)
            else:
                latency_ms = self._elapsed_ms(started)
                deterministic_calls, llm_calls = _count_planner_usage(
                    planning,
                    deterministic_calls,
                    llm_calls,
                )
                result = self._completed_case(case, planning, latency_ms)
            results.append(result)
            logger.info(
                "benchmark_case id=%s passed=%s latency_ms=%.3f",
                case.id,
                str(result.passed).lower(),
                result.latency_ms,
            )

        report = _build_report(
            self._planner,
            self._now(),
            tuple(results),
            deterministic_calls,
            llm_calls,
        )
        logger.info(
            "benchmark_completed planner=%s cases=%s plan_accuracy=%.4f "
            "fallback_rate=%.4f",
            report.planner,
            report.total_cases,
            report.plan_accuracy,
            report.fallback_rate,
        )
        return report

    def _elapsed_ms(self, started: float) -> float:
        """Measure non-negative planning-only elapsed milliseconds."""
        return max((self._clock() - started) * 1_000, 0.0)

    def _completed_case(
        self,
        case: EvaluationCase,
        planning: AgentPlanningResult,
        latency_ms: float,
    ) -> EvaluationCaseResult:
        """Compare one returned decision with the expected planner outcome."""
        decision = _evaluated_decision(planning)
        actual_plan = decision.selected_plan if decision else None
        actual_tools = (
            tuple(step.tool for step in decision.steps) if decision else None
        )
        plan_correct = actual_plan == case.expected_plan
        tools_correct = actual_tools == case.expected_tools
        return EvaluationCaseResult(
            id=case.id,
            query=case.query,
            expected_plan=case.expected_plan,
            actual_plan=actual_plan,
            expected_tools=case.expected_tools,
            actual_tools=actual_tools,
            confidence=decision.confidence if decision else None,
            fallback_used=planning.fallback_used,
            latency_ms=latency_ms,
            plan_correct=plan_correct,
            tools_correct=tools_correct,
            passed=plan_correct and tools_correct,
            requested_strategy=planning.requested_strategy,
            used_strategy=planning.used_strategy,
        )

    def _failed_case(
        self,
        case: EvaluationCase,
        latency_ms: float,
        error_type: str,
    ) -> EvaluationCaseResult:
        """Record one safe failure and continue with the remaining dataset."""
        return EvaluationCaseResult(
            id=case.id,
            query=case.query,
            expected_plan=case.expected_plan,
            actual_plan=None,
            expected_tools=case.expected_tools,
            actual_tools=None,
            confidence=None,
            fallback_used=False,
            latency_ms=latency_ms,
            plan_correct=False,
            tools_correct=False,
            passed=False,
            requested_strategy=self._planner,
            used_strategy=None,
            error_type=error_type,
        )


def _evaluated_decision(planning: AgentPlanningResult) -> AgentDecision | None:
    """Measure the requested planner, not a replacement fallback decision."""
    if planning.requested_strategy == "llm" and planning.fallback_used:
        return planning.primary_decision
    return planning.decision


def _count_planner_usage(
    planning: AgentPlanningResult,
    deterministic_calls: int,
    llm_calls: int,
) -> tuple[int, int]:
    """Count calls implied by the bounded planning-service contract."""
    if planning.requested_strategy == "deterministic":
        deterministic_calls += 1
    else:
        llm_calls += 1
        if planning.fallback_used:
            deterministic_calls += 1
    return deterministic_calls, llm_calls


def _build_report(
    planner: PlannerStrategy,
    timestamp: datetime,
    results: tuple[EvaluationCaseResult, ...],
    deterministic_calls: int,
    llm_calls: int,
) -> BenchmarkReport:
    """Aggregate reusable metrics into one immutable report."""
    total = len(results)
    passed = sum(result.passed for result in results)
    fallback_count = sum(result.fallback_used for result in results)
    confidence_values = tuple(
        result.confidence for result in results if result.confidence is not None
    )
    latency_values = tuple(result.latency_ms for result in results)
    calibration_values = tuple(
        (result.confidence, result.plan_correct)
        for result in results
        if result.confidence is not None
    )
    return BenchmarkReport(
        planner=planner,
        timestamp=timestamp,
        total_cases=total,
        passed_cases=passed,
        failed_cases=total - passed,
        plan_accuracy=calculate_accuracy(
            (result.plan_correct for result in results), total
        ),
        tool_accuracy=calculate_accuracy(
            (result.tools_correct for result in results), total
        ),
        fallback_count=fallback_count,
        fallback_rate=calculate_rate(fallback_count, total),
        confidence_statistics=calculate_statistics(confidence_values),
        latency_statistics_ms=calculate_statistics(latency_values),
        deterministic_planner_calls=deterministic_calls,
        llm_planner_calls=llm_calls,
        tool_executions=0,
        selected_plan_distribution=calculate_distribution(
            result.actual_plan for result in results if result.actual_plan
        ),
        confidence_calibration=calculate_confidence_calibration(
            calibration_values
        ),
        case_results=results,
    )
