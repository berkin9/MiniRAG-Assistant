"""Command-line entry point for MiniRAG Assistant."""

import argparse
import logging
import sys
from pathlib import Path

from app.config import ConfigurationError, Settings, get_settings
from app.services.answering import AnswerGenerationError, AnswerResult
from app.services.document_loader import DocumentLoadError
from app.services.embeddings import EmbeddingError, EmbeddingService
from app.services.indexing import IndexingError, index_directory, index_document
from app.services.ingestion import (
    DirectoryNotFoundError,
    discover_documents,
    ingest_directory,
)
from app.services.llm_providers import LLMProviderError
from app.services.retrieval import retrieve
from app.services.runtime import (
    ask_with_settings,
    build_collection_registry,
    build_index_services,
)
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
    search_parser = subparsers.add_parser("search", help="Search indexed documents")
    search_parser.add_argument("query", help="Natural-language search query")
    search_parser.add_argument("--top-k", type=int, help="Maximum results to return")
    search_parser.add_argument("--collection", help="Logical RAG collection")
    ask_parser = subparsers.add_parser("ask", help="Ask a grounded question")
    ask_parser.add_argument("question", help="Question about indexed documents")
    ask_parser.add_argument("--top-k", type=int, help="Maximum context chunks")
    ask_parser.add_argument("--collection", help="Logical RAG collection")
    subparsers.add_parser("collections", help="List configured RAG collections")
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


def _run_search(
    query: str,
    top_k: int,
    settings: Settings,
    collection: str | None = None,
) -> None:
    """Search the persistent index and print readable matches."""
    logical_collection = build_collection_registry(settings).resolve_logical_name(
        collection
    )
    embedder, vector_store = _build_rag_services(settings, logical_collection)
    response = retrieve(
        query,
        top_k,
        settings.max_retrieval_distance,
        embedder,
        vector_store,
        logical_collection,
    )
    if not response.results:
        print("No relevant results found.")
        return
    for rank, result in enumerate(response.results, start=1):
        page = f", page {result.page_number}" if result.page_number else ""
        preview = " ".join(result.text.split())[:200]
        print(
            f"{rank}. {result.source_file}{page}, chunk {result.chunk_index}, "
            f"distance {result.distance:.4f}\n   {preview}"
        )


def _print_answer(result: AnswerResult) -> None:
    """Print a grounded answer and its citations."""
    print(result.answer)
    if not result.sources:
        return
    print("\nSources:")
    for source in result.sources:
        page = f", page {source.page_number}" if source.page_number else ""
        print(
            f"- [{source.label}] {Path(source.source_file).name}{page}, "
            f"chunk {source.chunk_index}, distance {source.distance:.4f}"
        )


def _run_ask(
    question: str,
    top_k: int,
    settings: Settings,
    collection: str | None = None,
) -> None:
    """Generate and print an answer grounded in indexed chunks."""
    _print_answer(ask_with_settings(question, top_k, settings, collection))


def _run_collections(settings: Settings) -> None:
    """Print configured logical collections deterministically."""
    registry = build_collection_registry(settings)
    for collection in registry.list_collections():
        suffix = " (default)" if collection == registry.default_collection else ""
        print(f"{collection}{suffix}")


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
        elif args.command == "search":
            top_k = args.top_k if args.top_k is not None else settings.default_top_k
            _run_search(args.query, top_k, settings, args.collection)
        elif args.command == "ask":
            top_k = args.top_k if args.top_k is not None else settings.default_top_k
            _run_ask(args.question, top_k, settings, args.collection)
        elif args.command == "collections":
            _run_collections(settings)
    except (
        ConfigurationError,
        DirectoryNotFoundError,
        DocumentLoadError,
        EmbeddingError,
        AnswerGenerationError,
        IndexingError,
        LLMProviderError,
        VectorStoreError,
        OSError,
        UnicodeError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
