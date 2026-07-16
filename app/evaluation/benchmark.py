"""Benchmark construction and readable terminal formatting."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import perf_counter

from app.config import Settings
from app.evaluation.dataset import load_dataset
from app.evaluation.evaluator import AgentEvaluator
from app.evaluation.report import BenchmarkReport
from app.agent.planning_service import build_agent_planning_service
from app.services.llm_providers import LLMProvider

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "agent_planning.json"
)


def run_benchmark(
    settings: Settings,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    provider_factory: Callable[[], LLMProvider] | None = None,
    clock: Callable[[], float] = perf_counter,
    now: Callable[[], datetime] | None = None,
) -> BenchmarkReport:
    """Run the configured planner against a validated dataset without tools."""
    dataset = load_dataset(dataset_path)
    planning_service = build_agent_planning_service(settings, provider_factory)
    return AgentEvaluator(
        planning_service,
        settings.agent_planning_mode,
        clock,
        now,
    ).evaluate(dataset)


def format_benchmark_report(report: BenchmarkReport) -> str:
    """Format concise human-readable benchmark output."""
    confidence = report.confidence_statistics
    latency = report.latency_statistics_ms
    most_selected = next(iter(report.selected_plan_distribution), "none")
    lines = [
        "Running benchmark...",
        f"Planner: {report.planner}",
        f"Cases: {report.total_cases}",
        f"Plan Accuracy: {report.plan_accuracy:.1%}",
        f"Tool Accuracy: {report.tool_accuracy:.1%}",
        f"Fallback Rate: {report.fallback_rate:.1%}",
        (
            f"Average Confidence: {confidence.average:.2f}"
            if confidence
            else "Average Confidence: n/a"
        ),
        (
            f"Average Planning Latency: {latency.average:.2f} ms"
            if latency
            else "Average Planning Latency: n/a"
        ),
        f"Deterministic planner calls: {report.deterministic_planner_calls}",
        f"LLM planner calls: {report.llm_planner_calls}",
        f"Tool executions: {report.tool_executions}",
        f"Most Selected Plan: {most_selected}",
    ]
    if report.failures:
        lines.append("Failures:")
        for failure in report.failures:
            lines.append(
                f"- {failure.id}: expected {failure.expected_plan}, "
                f"actual {failure.actual_plan or failure.error_type or 'none'}"
            )
    else:
        lines.append("Failures: none")
    return "\n".join(lines)
