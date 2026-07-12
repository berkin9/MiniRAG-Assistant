"""Command-line entry point for MiniRAG Assistant."""

import argparse
import logging
import sys

from app.config import ConfigurationError, get_settings
from app.services.document_loader import DocumentLoadError
from app.services.ingestion import (
    DirectoryNotFoundError,
    discover_documents,
    ingest_directory,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Ingest local documents for RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a directory")
    ingest_parser.add_argument("directory", nargs="?", help="Directory to ingest")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected command and return its exit status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    try:
        settings = get_settings()
        directory = args.directory or settings.data_dir
        document_count = len(discover_documents(directory))
        chunks = ingest_directory(
            directory, settings.chunk_size, settings.chunk_overlap
        )
    except (
        ConfigurationError,
        DirectoryNotFoundError,
        DocumentLoadError,
        OSError,
        UnicodeError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Documents loaded: {document_count}")
    print(f"Chunks created: {len(chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
