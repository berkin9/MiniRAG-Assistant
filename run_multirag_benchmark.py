import csv
import json
from pathlib import Path

from app.config import get_settings
from app.evaluation.multirag import (
    MultiRAGEvaluator,
    load_multirag_dataset,
)
from app.services.collection_selection import CollectionSelectionResult
from app.services.runtime import _cross_collection_retrieve


class RealMultiRAGRunner:
    def __init__(self, settings):
        self.settings = settings

    def retrieve(
        self,
        query: str,
        expected_collections: tuple[str, ...],
    ):
        selection = CollectionSelectionResult(
            collections=expected_collections,
            strategy="benchmark_manual",
        )

        return _cross_collection_retrieve(
            query=query,
            settings=self.settings,
            selection=selection,
        )


def main():
    settings = get_settings()

    dataset = load_multirag_dataset(
        "benchmarks/multirag_retrieval.json"
    )

    runner = RealMultiRAGRunner(settings)
    evaluator = MultiRAGEvaluator(runner)

    report = evaluator.evaluate(dataset)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # JSON export
    json_path = reports_dir / "multirag-benchmark.json"
    json_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # CSV export
    csv_path = reports_dir / "multirag-benchmark.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "id",
                "selected_collections",
                "collection_exact_match",
                "collection_precision",
                "collection_recall",
                "relevant_document_recall_at_k",
                "duplicate_removal_count",
                "total_candidates",
                "global_results",
                "retrieval_latency_ms",
                "error_type",
            ]
        )

        for result in report.case_results:
            writer.writerow(
                [
                    result.id,
                    "|".join(result.selected_collections),
                    result.collection_exact_match,
                    result.collection_precision,
                    result.collection_recall,
                    result.relevant_document_recall_at_k,
                    result.duplicate_removal_count,
                    result.total_candidates,
                    result.global_results,
                    result.retrieval_latency_ms,
                    result.error_type or "",
                ]
            )

    print()
    print("=== Multi-RAG Retrieval Benchmark ===")
    print(f"Total cases: {report.total_cases}")
    print(f"Failed cases: {report.failed_cases}")
    print(
        f"Collection exact match rate: "
        f"{report.collection_exact_match_rate:.2%}"
    )
    print(
        f"Average collection precision: "
        f"{report.average_collection_precision:.2%}"
    )
    print(
        f"Average collection recall: "
        f"{report.average_collection_recall:.2%}"
    )
    print(
        f"Relevant document recall@K: "
        f"{report.average_relevant_document_recall_at_k:.2%}"
    )
    print(
        f"Duplicates removed: "
        f"{report.total_duplicates_removed}"
    )
    print(
        f"Average retrieval latency: "
        f"{report.average_retrieval_latency_ms:.2f} ms"
    )

    print()
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()