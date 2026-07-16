"""Tests for the separate Streamlit planning benchmark page."""

from contextlib import contextmanager
from pathlib import Path

import pytest

from app.config import Settings
from app.evaluation.benchmark import run_benchmark
from app.pages import benchmark as page


class FakeColumn:
    """Record one Streamlit metric column."""

    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events

    def metric(self, label: str, value: str) -> None:
        self.events.append(("metric", (label, value)))


class FakeStreamlit:
    """Record report rendering without starting Streamlit."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def columns(self, count: int) -> list[FakeColumn]:
        return [FakeColumn(self.events) for _ in range(count)]

    def caption(self, value: str) -> None:
        self.events.append(("caption", value))

    def subheader(self, value: str) -> None:
        self.events.append(("subheader", value))

    def bar_chart(self, value: object) -> None:
        self.events.append(("bar_chart", value))

    def table(self, value: object) -> None:
        self.events.append(("table", value))

    @contextmanager
    def expander(self, label: str, expanded: bool = False):
        self.events.append(("expander", (label, expanded)))
        yield

    def markdown(self, value: str) -> None:
        self.events.append(("markdown", value))

    def download_button(
        self,
        label: str,
        data: str,
        *,
        file_name: str,
        mime: str,
    ) -> None:
        self.events.append(
            ("download", (label, data, file_name, mime))
        )


def test_benchmark_page_applies_selected_planner_only_to_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Page selection should not mutate the application's frozen settings."""
    settings = Settings(agent_planning_mode="deterministic")
    captured: list[tuple[str, Path]] = []
    expected = run_benchmark(settings)

    def fake_run(selected: Settings, path: Path):
        captured.append((selected.agent_planning_mode, path))
        return expected

    monkeypatch.setattr(page, "run_benchmark", fake_run)

    result = page._run_selected_benchmark(
        settings,
        "llm",
        Path("custom.json"),
    )

    assert result is expected
    assert captured == [("llm", Path("custom.json"))]
    assert settings.agent_planning_mode == "deterministic"


def test_benchmark_page_renders_metrics_chart_and_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The separate page should expose summary and both export formats."""
    report = run_benchmark(Settings())
    fake_st = FakeStreamlit()
    monkeypatch.setattr(page, "st", fake_st)

    page._render_report(report)

    assert ("metric", ("Plan accuracy", "100.0%")) in fake_st.events
    assert ("metric", ("Fallback rate", "0.0%")) in fake_st.events
    assert ("bar_chart", report.selected_plan_distribution) in fake_st.events
    downloads = [value for event, value in fake_st.events if event == "download"]
    assert len(downloads) == 2
    assert downloads[0][2] == "agent-benchmark.json"
    assert '"tool_executions": 0' in downloads[0][1]
    assert downloads[1][2] == "agent-benchmark.csv"
    assert "expected_plan,actual_plan" in downloads[1][1]
