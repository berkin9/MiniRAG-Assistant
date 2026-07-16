"""Offline tests for separate Multi-RAG retrieval evaluation."""

from pathlib import Path

import pytest

from app.evaluation.multirag import (
    MultiRAGEvaluationCase,
    MultiRAGEvaluationDataset,
    MultiRAGEvaluator,
    collection_precision,
    collection_recall,
    load_multirag_dataset,
    relevant_document_recall,
)
from app.services.collection_selection import CollectionSelectionResult
from app.services.cross_collection import CrossCollectionRetrievalResponse
from app.services.retrieval import RetrievalResult


class FakeRunner:
    """Return controlled responses and fail one case without a network."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(
        self, query: str, expected_collections: tuple[str, ...]
    ) -> CrossCollectionRetrievalResponse:
        del expected_collections
        self.calls.append(query)
        if query == "fail":
            raise RuntimeError("secret provider detail")
        selection = CollectionSelectionResult(
            collections=("technical", "policies"),
            strategy="deterministic",
        )
        result = RetrievalResult(
            text="evidence",
            source_file="architecture.pdf",
            file_type="pdf",
            page_number=1,
            chunk_index=0,
            document_hash="hash",
            distance=0.1,
            collection="technical",
            global_rank=1,
        )
        return CrossCollectionRetrievalResponse(
            query=query,
            results=(result,),
            selection=selection,
            selected_collections=selection.collections,
            collections_searched=selection.collections,
            results_per_collection={"technical": 2, "policies": 1},
            total_candidates=3,
            deduplicated_candidates=2,
            returned_results=1,
            collection_failures={},
            latency_ms=4.0,
        )


def test_bundled_multirag_dataset_loads() -> None:
    dataset = load_multirag_dataset("benchmarks/multirag_retrieval.json")

    assert len(dataset.cases) >= 2
    assert dataset.cases[0].expected_collections == ("technical", "policies")


def test_collection_and_document_metrics() -> None:
    selected = ("technical", "general")
    expected = ("technical", "policies")

    assert collection_precision(selected, expected) == 0.5
    assert collection_recall(selected, expected) == 0.5
    assert relevant_document_recall(
        ("/tmp/Architecture.PDF",),
        ("architecture.pdf", "policy.pdf"),
    ) == 0.5


def test_multirag_evaluator_aggregates_and_continues_after_failure() -> None:
    dataset = MultiRAGEvaluationDataset(
        cases=(
            MultiRAGEvaluationCase(
                id="good",
                query="good",
                expected_collections=("technical", "policies"),
                relevant_documents=("architecture.pdf",),
            ),
            MultiRAGEvaluationCase(
                id="failure",
                query="fail",
                expected_collections=("technical",),
                relevant_documents=("missing.pdf",),
            ),
        )
    )
    runner = FakeRunner()

    report = MultiRAGEvaluator(runner).evaluate(dataset)

    assert runner.calls == ["good", "fail"]
    assert report.total_cases == 2
    assert report.failed_cases == 1
    assert report.collection_exact_match_rate == 0.5
    assert report.average_collection_precision == 0.5
    assert report.average_collection_recall == 0.5
    assert report.average_relevant_document_recall_at_k == 0.5
    assert report.total_duplicates_removed == 1
    assert report.case_results[0].results_per_collection == {
        "technical": 2,
        "policies": 1,
    }
    failure = report.case_results[1]
    assert failure.error_type == "RuntimeError"
    assert "secret" not in failure.model_dump_json()


def test_invalid_multirag_dataset_has_safe_error(tmp_path: Path) -> None:
    source = tmp_path / "dataset.json"
    source.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_multirag_dataset(source)
