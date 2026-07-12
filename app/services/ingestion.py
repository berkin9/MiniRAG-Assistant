"""Document discovery and ingestion services."""

import logging
from dataclasses import dataclass
from pathlib import Path

from app.services.document_loader import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    load_document,
)
from app.services.text_splitter import split_text

logger = logging.getLogger(__name__)


class DirectoryNotFoundError(FileNotFoundError):
    """Raised when an ingestion directory does not exist."""


@dataclass(frozen=True)
class IngestionChunk:
    """A retrieval-ready text chunk and its document context."""

    text: str
    source: Path
    chunk_index: int
    metadata: dict[str, str | int]


def discover_documents(directory: str | Path) -> list[Path]:
    """Find supported, non-hidden files below a directory."""
    root = Path(directory)
    if not root.is_dir():
        raise DirectoryNotFoundError(f"Directory not found: {root}")

    paths: list[Path] = []
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in relative_parts):
            logger.debug("Skipping hidden path %s", path)
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(path)
        elif path.is_file():
            logger.debug("Skipping unsupported file %s", path)
    return sorted(paths, key=lambda path: path.as_posix())


def ingest_directory(
    directory: str | Path, chunk_size: int, chunk_overlap: int
) -> list[IngestionChunk]:
    """Load and split all supported documents in a directory."""
    results: list[IngestionChunk] = []
    for path in discover_documents(directory):
        try:
            document = load_document(path)
            chunks = split_text(document.content, chunk_size, chunk_overlap)
        except (DocumentLoadError, OSError, UnicodeError, ValueError):
            logger.error("Failed to ingest document %s", path)
            raise

        results.extend(
            IngestionChunk(
                text=text,
                source=document.source,
                chunk_index=index,
                metadata=dict(document.metadata),
            )
            for index, text in enumerate(chunks)
        )
    return results
