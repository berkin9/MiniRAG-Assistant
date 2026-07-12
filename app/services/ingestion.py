"""Document discovery and ingestion services."""

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import SUPPORTED_EXTENSIONS
from app.services.document_loader import (
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
    metadata: dict[str, str | int | None]


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
            results.extend(ingest_document(path, chunk_size, chunk_overlap))
        except (DocumentLoadError, OSError, UnicodeError, ValueError):
            logger.error("Failed to ingest document %s", path)
            raise
    return results


def ingest_document(
    path: str | Path, chunk_size: int, chunk_overlap: int
) -> list[IngestionChunk]:
    """Load and split one supported document into page-aware chunks."""
    document = load_document(path)
    results: list[IngestionChunk] = []
    chunk_index = 0
    for page in document.pages:
        for text in split_text(page.content, chunk_size, chunk_overlap):
            metadata: dict[str, str | int | None] = {
                **document.metadata,
                "source_file": str(document.source),
                "file_type": document.file_type,
                "page_number": page.page_number,
                "chunk_index": chunk_index,
            }
            results.append(
                IngestionChunk(
                    text=text,
                    source=document.source,
                    chunk_index=chunk_index,
                    metadata=metadata,
                )
            )
            chunk_index += 1
    return results
