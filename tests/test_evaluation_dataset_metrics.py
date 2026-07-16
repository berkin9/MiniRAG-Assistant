"""Tests for benchmark dataset loading and reusable metrics."""

import json

import pytest

from app.evaluation.benchmark import DEFAULT_DATASET_PATH
from app.evaluation.dataset import EvaluationDatasetError, load_dataset
from app.evaluation.metrics import (
    calculate_accuracy,
    calculate_confidence_calibration,
    calculate_distribution,
    calculate_rate,
    calculate_statistics,
)


def test_bundled_evaluation_dataset_loads_registered_cases() -> None:
    """The versioned human-editable dataset should validate successfully."""
    dataset = load_dataset(DEFAULT_DATASET_PATH)

    assert len(dataset.cases) == 12
    assert len({case.id for case in dataset.cases}) == 12
    assert {case.expected_plan for case in dataset.cases} == {
        "ask",
        "search",
        "collections",
        "routing",
        "route_and_ask",
        "route_and_search",
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-json", "invalid JSON"),
        (
            json.dumps(
                {
                    "cases": [
                        {
                            "id": "bad",
                            "query": "question",
                            "expected_plan": "ask",
                            "expected_tools": ["search"],
                            "description": "mismatch",
                        }
                    ]
                }
            ),
            "expected tools do not match",
        ),
        (
            json.dumps(
                {
                    "cases": [
                        {
                            "id": "same",
                            "query": "question one",
                            "expected_plan": "ask",
                            "expected_tools": ["ask"],
                            "description": "first",
                        },
                        {
                            "id": "same",
                            "query": "question two",
                            "expected_plan": "ask",
                            "expected_tools": ["ask"],
                            "description": "second",
                        },
                    ]
                }
            ),
            "case ids must be unique",
        ),
    ],
)
def test_invalid_evaluation_dataset_is_rejected(
    tmp_path, payload: str, message: str
) -> None:
    """Malformed benchmark data should fail before any planner call."""
    path = tmp_path / "cases.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(EvaluationDatasetError, match=message):
        load_dataset(path)


def test_accuracy_rate_statistics_and_distribution_metrics() -> None:
    """Core aggregate metrics should use all cases and deterministic ordering."""
    statistics = calculate_statistics((0.9, 0.5, 0.7))

    assert calculate_accuracy((True, False, True), 3) == pytest.approx(2 / 3)
    assert calculate_rate(1, 4) == 0.25
    assert statistics is not None
    assert statistics.average == pytest.approx(0.7)
    assert statistics.minimum == 0.5
    assert statistics.maximum == 0.9
    assert statistics.median == 0.7
    assert calculate_distribution(("ask", "search", "ask", "routing")) == {
        "ask": 2,
        "routing": 1,
        "search": 1,
    }


def test_confidence_calibration_uses_documented_bands() -> None:
    """Confidence grouping should calculate correctness independently per band."""
    buckets = calculate_confidence_calibration(
        ((0.95, True), (0.90, False), (0.80, True), (0.69, False))
    )

    assert [(bucket.cases, bucket.correct, bucket.accuracy) for bucket in buckets] == [
        (2, 1, 0.5),
        (1, 1, 1.0),
        (1, 0, 0.0),
    ]
