"""Command-line entry point for MiniRAG Assistant."""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from app.agent import (
    PlannedAgentExecutionError,
    PlannedAgentResult,
    build_planned_agent_service,
)
from app.agent.decision_adapter import AgentExecutionPreparationError
from app.agent.planning_service import AgentPlanningServiceError
from app.config import ConfigurationError, Settings, get_settings
from app.evaluation.benchmark import (
    DEFAULT_DATASET_PATH,
    format_benchmark_report,
    run_benchmark,
)
from app.evaluation.dataset import EvaluationDatasetError
from app.evaluation.report import write_csv_report, write_json_report
from app.services.answering import AnswerGenerationError, AnswerResult
from app.services.cross_collection import (
    CrossCollectionRetrievalError,
    CrossCollectionRetrievalResponse,
)
from app.services.document_loader import DocumentLoadError
from app.services.demo_indexing import (
    DemoIndexingError,
    DemoIndexingResult,
    ensure_demo_documents_indexed,
)
from app.services.embeddings import EmbeddingError, EmbeddingService
from app.services.indexing import IndexingError, index_directory, index_document
from app.services.ingestion import (
    DirectoryNotFoundError,
    discover_documents,
    ingest_directory,
)
from app.services.llm_providers import LLMProviderError
from app.services.runtime import (
    RoutedAnswer,
    RoutedSearch,
    answer_with_routing,
    ask_with_settings,
    build_collection_registry,
    build_index_services,
    route_with_settings,
    search_with_settings,
)
from app.services.routing import RoutingDecision
from app.services.vector_store import ChromaVectorStore, VectorStoreError


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Ingest local documents for RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a directory")
    ingest_parser.add_argument("directory", nargs="?", help="Directory to ingest")
    index_parser = subparsers.add_parser("index", help="Index a file or directory")
    index_parser.add_argument("path", nargs="?", help="File or directory to index")
    index_parser.add_argument("--collection", help="Logical RAG collection")
    subparsers.add_parser(
        "demo-index", help="Index predefined shared demo documents"
    )
    search_parser = subparsers.add_parser("search", help="Search indexed documents")
    search_parser.add_argument("query", help="Natural-language search query")
    search_parser.add_argument("--top-k", type=int, help="Maximum results to return")
    search_mode = search_parser.add_mutually_exclusive_group()
    search_mode.add_argument("--collection", help="Logical RAG collection")
    search_mode.add_argument(
        "--collections",
        help="Comma-separated bounded collection list for cross-collection mode",
    )
    search_mode.add_argument(
        "--auto-route", action="store_true", help="Route the query automatically"
    )
    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question")
    ask_parser.add_argument("question", help="Question about indexed documents")
    ask_parser.add_argument("--top-k", type=int, help="Maximum context chunks")
    ask_mode = ask_parser.add_mutually_exclusive_group()
    ask_mode.add_argument("--collection", help="Logical RAG collection")
    ask_mode.add_argument(
        "--collections",
        help="Comma-separated bounded collection list for cross-collection mode",
    )
    ask_mode.add_argument(
        "--auto-route", action="store_true", help="Route the question automatically"
    )
    subparsers.add_parser("collections", help="List configured RAG collections")
    route_parser = subparsers.add_parser(
        "route", help="Inspect automatic collection routing"
    )
    route_parser.add_argument("question", help="Question to route without retrieval")
    agent_parser = subparsers.add_parser(
        "agent", help="Select and run a bounded agent plan"
    )
    agent_parser.add_argument("request", help="Request for the agent to handle")
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Evaluate agent planning without executing tools"
    )
    benchmark_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Evaluation dataset JSON path",
    )
    benchmark_parser.add_argument(
        "--planner",
        choices=("deterministic", "llm"),
        help="Override AGENT_PLANNING_MODE for this benchmark",
    )
    benchmark_parser.add_argument(
        "--json",
        type=Path,
        dest="json_report",
        help="Write a machine-readable JSON report",
    )
    benchmark_parser.add_argument(
        "--csv",
        type=Path,
        dest="csv_report",
        help="Write per-case CSV results",
    )
    return parser


def _build_rag_services(
    settings: Settings, collection: str | None = None
) -> tuple[EmbeddingService, ChromaVectorStore]:
    """Create local embedding and persistent vector-store services."""
    return build_index_services(settings, collection)


def _run_ingest(directory: str | Path, chunk_size: int, overlap: int) -> None:
    """Run the non-persistent ingestion command."""
    document_count = len(discover_documents(directory))
    chunks = ingest_directory(directory, chunk_size, overlap)
    print(f"Documents loaded: {document_count}")
    print(f"Chunks created: {len(chunks)}")


def _run_index(
    path: str | Path, settings: Settings, collection: str | None = None
) -> None:
    """Index one file or every supported file in a directory."""
    logical_collection = build_collection_registry(settings).resolve_logical_name(
        collection
    )
    source = Path(path)
    embedder, vector_store = _build_rag_services(settings, logical_collection)
    if source.is_file():
        results = [
            index_document(
                source,
                settings.chunk_size,
                settings.chunk_overlap,
                embedder,
                vector_store,
                logical_collection,
            )
        ]
    else:
        results = index_directory(
            source,
            settings.chunk_size,
            settings.chunk_overlap,
            embedder,
            vector_store,
            logical_collection,
        )
    for result in results:
        print(
            f"{result.source}: {result.status} in {result.collection} "
            f"({result.stored_chunks} chunk(s))"
        )
    print(f"Documents processed: {len(results)}")
    print(f"Chunks stored: {sum(result.stored_chunks for result in results)}")


def _run_demo_index(settings: Settings) -> DemoIndexingResult:
    """Run the shared demo service and print its safe aggregate outcome."""
    result = ensure_demo_documents_indexed(
        settings,
        build_collection_registry(settings),
    )
    print(f"Discovered: {result.discovered_documents}")
    print(f"Indexed: {result.indexed_documents}")
    print(f"Skipped: {result.skipped_documents}")
    print(f"Failed: {result.failed_documents}")
    for error in result.errors:
        print(f"Warning: {error}", file=sys.stderr)
    return result


def _run_search(
    query: str,
    top_k: int,
    settings: Settings,
    query_mode: str = "manual",
    collection: str | None = None,
    collections: tuple[str, ...] | None = None,
) -> None:
    """Search the persistent index and print readable matches."""
    if collections is None:
        routed = search_with_settings(
            query, top_k, settings, query_mode, collection
        )
    else:
        routed = search_with_settings(
            query,
            top_k,
            settings,
            query_mode,
            collection,
            collections=collections,
        )
    if routed.selection is not None:
        _print_selection(routed.selection)
    elif routed.routing.strategy != "manual":
        _print_routing(routed.routing)
    _print_search_response(routed)


def _print_search_response(routed: RoutedSearch) -> None:
    """Print readable matches from a structured routed search."""
    response = routed.response
    if isinstance(response, CrossCollectionRetrievalResponse):
        _print_cross_collection_metadata(response)
    if not response.results:
        print("No relevant results found.")
        return
    for rank, result in enumerate(response.results, start=1):
        page = f", page {result.page_number}" if result.page_number else ""
        preview = " ".join(result.text.split())[:200]
        collection_prefix = (
            f"{result.collection} — "
            if isinstance(response, CrossCollectionRetrievalResponse)
            else ""
        )
        score = (
            f", relevance {result.normalized_score:.4f}"
            if result.normalized_score is not None
            else ""
        )
        print(
            f"{rank}. {collection_prefix}{result.source_file}{page}, "
            f"chunk {result.chunk_index}, distance {result.distance:.4f}{score}\n"
            f"   {preview}"
        )


def _print_answer(result: AnswerResult) -> None:
    """Print a grounded answer and its citations."""
    print(result.answer)
    if not result.sources:
        return
    print("\nSources:")
    for source in result.sources:
        page = f", page {source.page_number}" if source.page_number else ""
        citation_identity = (
            f" [{source.citation_id}]" if source.citation_id else ""
        )
        collections = (
            f"{', '.join(source.matched_collections or (source.collection,))} — "
            if result.selected_collections
            else ""
        )
        print(
            f"- [{source.label}]{citation_identity} "
            f"{collections}{Path(source.source_file).name}{page}, "
            f"chunk {source.chunk_index}, distance {source.distance:.4f}"
        )


def _run_ask(
    question: str,
    top_k: int,
    settings: Settings,
    query_mode: str = "manual",
    collection: str | None = None,
    collections: tuple[str, ...] | None = None,
) -> None:
    """Generate and print an answer grounded in indexed chunks."""
    if (
        settings.rag_retrieval_strategy == "single_collection"
        and query_mode == "manual"
        and collections is None
    ):
        _print_answer(ask_with_settings(question, top_k, settings, collection))
        return
    if collections is None:
        routed = answer_with_routing(
            question, top_k, settings, query_mode, collection
        )
    else:
        routed = answer_with_routing(
            question,
            top_k,
            settings,
            query_mode,
            collection,
            collections=collections,
        )
    if routed.selection is not None:
        _print_selection(routed.selection)
        if routed.retrieval is not None:
            _print_cross_collection_metadata(routed.retrieval)
    else:
        _print_routing(routed.routing)
    _print_answer(routed.answer)


def _print_selection(selection: object) -> None:
    """Print bounded multi-collection selection metadata."""
    from app.services.collection_selection import CollectionSelectionResult

    if not isinstance(selection, CollectionSelectionResult):
        return
    print("Selected collections:")
    for name in selection.collections:
        print(f"- {name}")
    print(f"Selection strategy: {selection.strategy}")
    if selection.confidence is not None:
        print(f"Selection confidence: {selection.confidence:.2f}")
    if selection.fallback_used:
        print("Selection fallback used: yes")


def _print_cross_collection_metadata(
    response: CrossCollectionRetrievalResponse,
) -> None:
    """Print safe cross-collection candidate and fusion counts."""
    if response.selection:
        print("Results per collection:")
        for collection in response.selected_collections:
            print(f"- {collection}: {response.results_per_collection[collection]}")
    print(f"Candidates before fusion: {response.total_candidates}")
    print(f"Duplicates removed: {response.duplicate_removal_count}")
    print(f"Global results: {response.returned_results}")


def _run_collections(settings: Settings) -> None:
    """Print configured logical collections deterministically."""
    registry = build_collection_registry(settings)
    for collection in registry.list_collections():
        suffix = " (default)" if collection == registry.default_collection else ""
        print(f"{collection}{suffix}")


def _run_route(question: str, settings: Settings) -> None:
    """Inspect automatic routing without retrieval or answer generation."""
    _print_routing(
        route_with_settings(question, settings, query_mode="automatic")
    )


def _run_agent(request: str, settings: Settings) -> None:
    """Select and execute one predefined bounded plan for a request."""
    result = build_planned_agent_service(settings).run(request)
    _print_agent_planning(result)
    response = result.execution
    if len(response.steps) == 1:
        print(f"Selected tool: {response.decision.tool}")
        print(f"Reason: {response.decision.reason}")
        _print_agent_tool_result(response.result, show_routing=True)
        return

    plan = response.plan
    if plan is None:
        raise ValueError("Agent response is missing its execution plan")
    print(f"Plan: {plan.name}")
    print(f"Reason: {plan.reason}")
    for index, step in enumerate(response.steps, start=1):
        print(f"\nStep {index}: {step.tool}")
        _print_agent_tool_result(
            step.result,
            show_routing=step.tool == "routing",
            label_answer=True,
        )


def _run_benchmark_command(
    settings: Settings,
    dataset: Path,
    planner: str | None = None,
    json_report: Path | None = None,
    csv_report: Path | None = None,
) -> None:
    """Evaluate planning only and optionally write JSON or CSV reports."""
    benchmark_settings = (
        replace(settings, agent_planning_mode=planner) if planner else settings
    )
    report = run_benchmark(benchmark_settings, dataset)
    print(format_benchmark_report(report))
    if json_report is not None:
        write_json_report(report, json_report)
        print(f"JSON report: {json_report}")
    if csv_report is not None:
        write_csv_report(report, csv_report)
        print(f"CSV report: {csv_report}")


def _print_agent_planning(result: PlannedAgentResult) -> None:
    """Print safe planner metadata without raw provider details."""
    planning = result.planning
    print(f"Planning requested: {planning.requested_strategy}")
    print(f"Planning used: {planning.used_strategy}")
    print(f"Selected plan: {planning.decision.selected_plan}")
    print(f"Confidence: {planning.decision.confidence:.2f}")
    print(f"Planning reason: {planning.decision.reason}")
    print(f"Fallback used: {'yes' if planning.fallback_used else 'no'}")
    if planning.fallback_reason:
        print(f"Fallback reason: {planning.fallback_reason}")


def _print_agent_tool_result(
    result: object,
    show_routing: bool = True,
    label_answer: bool = False,
) -> None:
    """Print one agent tool result without exposing internal objects."""
    if isinstance(result, RoutedAnswer):
        if show_routing:
            _print_routing(result.routing)
        if label_answer:
            print("Answer:")
        _print_answer(result.answer)
    elif isinstance(result, RoutedSearch):
        if show_routing:
            _print_routing(result.routing)
        _print_search_response(result)
    elif isinstance(result, RoutingDecision):
        _print_routing(result)
    else:
        print("Configured collections:")
        for collection in result:
            print(f"- {collection}")


def _print_routing(decision: RoutingDecision) -> None:
    """Print safe, observable routing metadata."""
    print(f"Selected collection: {decision.collection}")
    print(f"Routing strategy: {decision.strategy}")
    print(f"Reason: {decision.reason}")
    if decision.confidence is not None:
        print(f"Confidence: {decision.confidence:.2f}")


def _resolve_query_mode(
    auto_route: bool, collection: str | None, settings: Settings
) -> str:
    """Resolve CLI flags while retaining the configured default mode."""
    if collection is not None:
        return "manual"
    if auto_route:
        return "automatic"
    return settings.default_query_mode


def _parse_collections(value: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated CLI collection list without normalizing silently."""
    if value is None:
        return None
    collections = tuple(name.strip() for name in value.split(",") if name.strip())
    if not collections:
        raise ValueError("--collections must include at least one collection")
    return collections


def main(argv: list[str] | None = None) -> int:
    """Run the selected command and return its exit status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    try:
        settings = get_settings()
        if args.command == "ingest":
            _run_ingest(
                args.directory or settings.data_dir,
                settings.chunk_size,
                settings.chunk_overlap,
            )
        elif args.command == "index":
            _run_index(args.path or settings.data_dir, settings, args.collection)
        elif args.command == "demo-index":
            _run_demo_index(settings)
        elif args.command == "search":
            top_k = args.top_k if args.top_k is not None else settings.default_top_k
            collections = _parse_collections(args.collections)
            mode = _resolve_query_mode(
                args.auto_route, args.collection or args.collections, settings
            )
            if collections is None:
                _run_search(args.query, top_k, settings, mode, args.collection)
            else:
                _run_search(
                    args.query,
                    top_k,
                    settings,
                    mode,
                    args.collection,
                    collections,
                )
        elif args.command == "ask":
            top_k = args.top_k if args.top_k is not None else settings.default_top_k
            collections = _parse_collections(args.collections)
            mode = _resolve_query_mode(
                args.auto_route, args.collection or args.collections, settings
            )
            if collections is None:
                _run_ask(args.question, top_k, settings, mode, args.collection)
            else:
                _run_ask(
                    args.question,
                    top_k,
                    settings,
                    mode,
                    args.collection,
                    collections,
                )
        elif args.command == "collections":
            _run_collections(settings)
        elif args.command == "route":
            _run_route(args.question, settings)
        elif args.command == "agent":
            _run_agent(args.request, settings)
        elif args.command == "benchmark":
            _run_benchmark_command(
                settings,
                args.dataset,
                args.planner,
                args.json_report,
                args.csv_report,
            )
    except (
        ConfigurationError,
        DemoIndexingError,
        DirectoryNotFoundError,
        DocumentLoadError,
        EmbeddingError,
        EvaluationDatasetError,
        AnswerGenerationError,
        IndexingError,
        AgentExecutionPreparationError,
        AgentPlanningServiceError,
        PlannedAgentExecutionError,
        LLMProviderError,
        VectorStoreError,
        CrossCollectionRetrievalError,
        OSError,
        UnicodeError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
