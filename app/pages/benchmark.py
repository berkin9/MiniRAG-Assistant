"""Streamlit page for read-only agent-planning benchmarks."""

from dataclasses import replace
from pathlib import Path

import streamlit as st

from app.config import ConfigurationError, Settings, get_settings
from app.evaluation.benchmark import DEFAULT_DATASET_PATH, run_benchmark
from app.evaluation.dataset import EvaluationDatasetError
from app.evaluation.report import (
    BenchmarkReport,
    benchmark_report_csv,
    benchmark_report_json,
)

_REPORT_KEY = "agent_benchmark_report"


def main() -> None:
    """Render a separate planning-only benchmark page."""
    st.set_page_config(page_title="Agent Benchmark", page_icon="📊")
    st.title("Agent Planning Benchmark")
    st.caption(
        "Evaluate planning quality and fallback behavior without executing tools."
    )
    try:
        settings = get_settings()
    except ConfigurationError as error:
        st.error(str(error))
        return

    planner = st.selectbox(
        "Planner",
        options=("deterministic", "llm"),
        index=0 if settings.agent_planning_mode == "deterministic" else 1,
    )
    dataset_path = Path(
        st.text_input("Dataset", value=str(DEFAULT_DATASET_PATH))
    )
    if st.button("Run benchmark", type="primary"):
        try:
            with st.spinner("Running planning benchmark..."):
                report = _run_selected_benchmark(
                    settings,
                    planner,
                    dataset_path,
                )
        except (ConfigurationError, EvaluationDatasetError, OSError) as error:
            st.error(str(error))
            st.session_state.pop(_REPORT_KEY, None)
        else:
            st.session_state[_REPORT_KEY] = report

    report = st.session_state.get(_REPORT_KEY)
    if isinstance(report, BenchmarkReport):
        _render_report(report)


def _run_selected_benchmark(
    settings: Settings,
    planner: str,
    dataset_path: Path,
) -> BenchmarkReport:
    """Apply the page selection only to this benchmark run."""
    benchmark_settings = replace(settings, agent_planning_mode=planner)
    return run_benchmark(benchmark_settings, dataset_path)


def _render_report(report: BenchmarkReport) -> None:
    """Display summary, distributions, failures, and download exports."""
    confidence = report.confidence_statistics
    latency = report.latency_statistics_ms
    columns = st.columns(4)
    columns[0].metric("Plan accuracy", f"{report.plan_accuracy:.1%}")
    columns[1].metric("Tool accuracy", f"{report.tool_accuracy:.1%}")
    columns[2].metric("Fallback rate", f"{report.fallback_rate:.1%}")
    columns[3].metric(
        "Avg planning latency",
        f"{latency.average:.2f} ms" if latency else "n/a",
    )
    st.caption(
        f"Planner: {report.planner} · Cases: {report.total_cases} · "
        f"Average confidence: {confidence.average:.2f}"
        if confidence
        else f"Planner: {report.planner} · Cases: {report.total_cases}"
    )

    st.subheader("Selected plan distribution")
    st.bar_chart(report.selected_plan_distribution)
    st.subheader("Confidence calibration")
    st.table(
        [
            {
                "Band": bucket.label,
                "Cases": bucket.cases,
                "Correct": bucket.correct,
                "Accuracy": f"{bucket.accuracy:.1%}",
            }
            for bucket in report.confidence_calibration
        ]
    )

    if report.failures:
        with st.expander("Misclassifications", expanded=False):
            for failure in report.failures:
                st.markdown(
                    f"**{failure.id}** — expected `{failure.expected_plan}`, "
                    f"actual `{failure.actual_plan or failure.error_type or 'none'}`"
                )

    st.download_button(
        "Download JSON",
        benchmark_report_json(report),
        file_name="agent-benchmark.json",
        mime="application/json",
    )
    st.download_button(
        "Download CSV",
        benchmark_report_csv(report),
        file_name="agent-benchmark.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
