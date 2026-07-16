"""Reusable calculations for agent-planning benchmark metrics."""

from collections import Counter
from collections.abc import Iterable, Sequence
from statistics import fmean, median

from app.evaluation.report import ConfidenceCalibrationBucket, MetricStatistics


def calculate_accuracy(correct: Iterable[bool], total: int) -> float:
    """Return the fraction of all cases marked correct."""
    if total <= 0:
        return 0.0
    return sum(correct) / total


def calculate_rate(count: int, total: int) -> float:
    """Return a safe count-to-total ratio."""
    return count / total if total > 0 else 0.0


def calculate_statistics(values: Sequence[float]) -> MetricStatistics | None:
    """Return average, minimum, maximum, and median for non-empty values."""
    if not values:
        return None
    return MetricStatistics(
        average=fmean(values),
        minimum=min(values),
        maximum=max(values),
        median=median(values),
    )


def calculate_distribution(values: Iterable[str]) -> dict[str, int]:
    """Return deterministic descending-frequency distribution counts."""
    counts = Counter(values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def calculate_confidence_calibration(
    values: Sequence[tuple[float, bool]],
) -> tuple[ConfidenceCalibrationBucket, ...]:
    """Group confidence and plan correctness into three simple bands."""
    definitions = (
        ("confidence >= 0.90", lambda value: value >= 0.90),
        ("confidence 0.70-0.90", lambda value: 0.70 <= value < 0.90),
        ("confidence < 0.70", lambda value: value < 0.70),
    )
    buckets: list[ConfidenceCalibrationBucket] = []
    for label, includes in definitions:
        matching = tuple(correct for confidence, correct in values if includes(confidence))
        correct_count = sum(matching)
        buckets.append(
            ConfidenceCalibrationBucket(
                label=label,
                cases=len(matching),
                correct=correct_count,
                accuracy=calculate_rate(correct_count, len(matching)),
            )
        )
    return tuple(buckets)
