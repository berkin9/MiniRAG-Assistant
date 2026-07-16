"""Tests for benchmark exports and CLI integration."""

import csv
import json

import pytest

from app import main as cli
from app.config import Settings
from app.evaluation.benchmark import run_benchmark
from app.evaluation.report import (
    benchmark_report_csv,
    benchmark_report_json,
    write_csv_report,
    write_json_report,
)


def test_json_and_csv_exports_include_summary_cases_and_failures(tmp_path) -> None:
    """Both report formats should be deterministic and machine-readable."""
    report = run_benchmark(Settings())
    json_text = benchmark_report_json(report)
    csv_text = benchmark_report_csv(report)
    payload = json.loads(json_text)
    rows = list(csv.DictReader(csv_text.splitlines()))

    assert payload["planner"] == "deterministic"
    assert payload["summary"]["total_cases"] == 12
    assert payload["metrics"]["plan_accuracy"] == 1.0
    assert payload["metrics"]["tool_executions"] == 0
    assert payload["failures"] == []
    assert len(payload["cases"]) == 12
    assert len(rows) == 12
    assert rows[0]["expected_plan"]
    assert rows[0]["actual_plan"]
    assert rows[0]["passed"] == "True"

    json_path = tmp_path / "nested" / "report.json"
    csv_path = tmp_path / "nested" / "report.csv"
    write_json_report(report, json_path)
    write_csv_report(report, csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"] == payload[
        "summary"
    ]
    assert len(list(csv.DictReader(csv_path.read_text().splitlines()))) == 12


def test_cli_benchmark_runs_offline_and_writes_both_exports(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default deterministic benchmark should need no provider or tools."""
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    monkeypatch.setattr(cli, "get_settings", Settings)

    exit_code = cli.main(
        [
            "benchmark",
            "--planner",
            "deterministic",
            "--json",
            str(json_path),
            "--csv",
            str(csv_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Cases: 12" in output
    assert "Plan Accuracy: 100.0%" in output
    assert "Tool Accuracy: 100.0%" in output
    assert "Tool executions: 0" in output
    assert "Failures: none" in output
    assert json_path.is_file()
    assert csv_path.is_file()
