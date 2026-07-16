"""Separate dataset and metrics for cross-collection retrieval quality."""

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic import model_validator

from app.services.cross_collection import CrossCollectionRetrievalResponse


class MultiRAGEvaluationError(ValueError):
    """Raised when a Multi-RAG evaluation dataset is invalid."""


class MultiRAGEvaluationCase(BaseModel):
    """One expected collection and relevant-document retrieval outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    query: str
    expected_collections: tuple[str, ...] = Field(min_length=1)
    relevant_documents: tuple[str, ...] = Field(min_length=1)
    description: str | None = None

    @field_validator("id", "query")
    @classmethod
    def require_text(cls, value: str) -> str:
        """Reject blank identifiers and queries."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("expected_collections", "relevant_documents")
    @classmethod
    def reject_duplicates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Keep expected values stable and unique."""
        if len(value) != len(set(value)):
            raise ValueError("values must not contain duplicates")
        return value


class MultiRAGEvaluationDataset(BaseModel):
    """Human-editable set of uniquely identified retrieval cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cases: tuple[MultiRAGEvaluationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "MultiRAGEvaluationDataset":
        """Require stable unique identifiers."""
        ids = tuple(case.id for case in self.cases)
        if len(ids) != len(set(ids)):
            raise ValueError("Multi-RAG evaluation case ids must be unique")
        return self


class MultiRAGCaseResult(BaseModel):
    """Metrics and safe failure metadata for one retrieval case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    selected_collections: tuple[str, ...]
    collection_exact_match: bool
    collection_precision: float = Field(ge=0.0, le=1.0)
    collection_recall: float = Field(ge=0.0, le=1.0)
    relevant_document_recall_at_k: float = Field(ge=0.0, le=1.0)
    duplicate_removal_count: int = Field(ge=0)
    results_per_collection: dict[str, int]
    total_candidates: int = Field(ge=0)
    global_results: int = Field(ge=0)
    retrieval_latency_ms: float = Field(ge=0.0)
    error_type: str | None = None


class MultiRAGBenchmarkReport(BaseModel):
    """Aggregate retrieval-quality metrics kept separate from planner metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_cases: int
    failed_cases: int
    collection_exact_match_rate: float
    average_collection_precision: float
    average_collection_recall: float
    average_relevant_document_recall_at_k: float
    total_duplicates_removed: int
    average_retrieval_latency_ms: float
    case_results: tuple[MultiRAGCaseResult, ...]


class MultiRAGRunner(Protocol):
    """Minimal cross-collection retrieval dependency for offline evaluation."""

    def retrieve(
        self, query: str, expected_collections: tuple[str, ...]
    ) -> CrossCollectionRetrievalResponse:
        """Return one bounded fused retrieval response."""


class MultiRAGEvaluator:
    """Evaluate retrieval cases independently and continue after failures."""

    def __init__(
        self,
        runner: MultiRAGRunner,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._runner = runner
        self._clock = clock

    def evaluate(
        self, dataset: MultiRAGEvaluationDataset
    ) -> MultiRAGBenchmarkReport:
        """Calculate selection and retrieval metrics without planner metrics."""
        results: list[MultiRAGCaseResult] = []
        for case in dataset.cases:
            started = self._clock()
            try:
                response = self._runner.retrieve(
                    case.query, case.expected_collections
                )
            except Exception as error:
                results.append(
                    MultiRAGCaseResult(
                        id=case.id,
                        selected_collections=(),
                        collection_exact_match=False,
                        collection_precision=0.0,
                        collection_recall=0.0,
                        relevant_document_recall_at_k=0.0,
                        duplicate_removal_count=0,
                        results_per_collection={},
                        total_candidates=0,
                        global_results=0,
                        retrieval_latency_ms=_elapsed(self._clock, started),
                        error_type=type(error).__name__,
                    )
                )
                continue
            results.append(_evaluate_case(case, response))
        return _aggregate(tuple(results))


def load_multirag_dataset(path: str | Path) -> MultiRAGEvaluationDataset:
    """Load one UTF-8 JSON Multi-RAG retrieval dataset."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return MultiRAGEvaluationDataset.model_validate(payload)
    except OSError as error:
        raise MultiRAGEvaluationError(
            f"Could not read Multi-RAG evaluation dataset: {source}"
        ) from error
    except json.JSONDecodeError as error:
        raise MultiRAGEvaluationError(
            f"Multi-RAG evaluation dataset contains invalid JSON: {source}"
        ) from error
    except ValidationError as error:
        raise MultiRAGEvaluationError(
            "Invalid Multi-RAG evaluation dataset"
        ) from error


def collection_precision(
    selected: tuple[str, ...], expected: tuple[str, ...]
) -> float:
    """Return the selected collection fraction that was expected."""
    return len(set(selected) & set(expected)) / len(selected) if selected else 0.0


def collection_recall(
    selected: tuple[str, ...], expected: tuple[str, ...]
) -> float:
    """Return the expected collection fraction that was selected."""
    return len(set(selected) & set(expected)) / len(expected) if expected else 0.0


def relevant_document_recall(
    retrieved: tuple[str, ...], relevant: tuple[str, ...]
) -> float:
    """Return relevant-document recall over the globally limited result list."""
    normalized = {Path(name).name.casefold() for name in retrieved}
    expected = {Path(name).name.casefold() for name in relevant}
    return len(normalized & expected) / len(expected) if expected else 0.0


def _evaluate_case(
    case: MultiRAGEvaluationCase,
    response: CrossCollectionRetrievalResponse,
) -> MultiRAGCaseResult:
    """Compare one fused response with its human expectations."""
    selected = response.selected_collections
    documents = tuple(result.source_file for result in response.results)
    return MultiRAGCaseResult(
        id=case.id,
        selected_collections=selected,
        collection_exact_match=selected == case.expected_collections,
        collection_precision=collection_precision(
            selected, case.expected_collections
        ),
        collection_recall=collection_recall(selected, case.expected_collections),
        relevant_document_recall_at_k=relevant_document_recall(
            documents, case.relevant_documents
        ),
        duplicate_removal_count=response.duplicate_removal_count,
        results_per_collection=response.results_per_collection,
        total_candidates=response.total_candidates,
        global_results=response.returned_results,
        retrieval_latency_ms=response.latency_ms,
    )


def _aggregate(
    results: tuple[MultiRAGCaseResult, ...],
) -> MultiRAGBenchmarkReport:
    """Average each retrieval metric across every dataset case."""
    total = len(results)
    denominator = total or 1
    return MultiRAGBenchmarkReport(
        total_cases=total,
        failed_cases=sum(result.error_type is not None for result in results),
        collection_exact_match_rate=sum(
            result.collection_exact_match for result in results
        )
        / denominator,
        average_collection_precision=sum(
            result.collection_precision for result in results
        )
        / denominator,
        average_collection_recall=sum(
            result.collection_recall for result in results
        )
        / denominator,
        average_relevant_document_recall_at_k=sum(
            result.relevant_document_recall_at_k for result in results
        )
        / denominator,
        total_duplicates_removed=sum(
            result.duplicate_removal_count for result in results
        ),
        average_retrieval_latency_ms=sum(
            result.retrieval_latency_ms for result in results
        )
        / denominator,
        case_results=results,
    )


def _elapsed(clock: Callable[[], float], started: float) -> float:
    """Return safe elapsed milliseconds for failed cases."""
    return max((clock() - started) * 1_000, 0.0)
