"""Utilities for loading supported local documents."""

import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.models.document import Document

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".pdf"})
logger = logging.getLogger(__name__)


class UnsupportedFileTypeError(ValueError):
    """Raised when a document type is not supported."""


class DocumentLoadError(RuntimeError):
    """Raised when text cannot be extracted from a supported document."""


def load_document(path: str | Path) -> Document:
    """Load text and metadata from a supported local document."""
    document_path = Path(path)
    if not document_path.is_file():
        raise FileNotFoundError(f"Document not found: {document_path}")

    extension = document_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type {extension or '<none>'!r}: {document_path}"
        )

    metadata: dict[str, str | int] = {
        "filename": document_path.name,
        "extension": extension,
        "file_size_bytes": document_path.stat().st_size,
    }
    if extension == ".pdf":
        try:
            reader = PdfReader(document_path)
            content = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except PyPdfError as error:
            raise DocumentLoadError(
                f"Could not read PDF document: {document_path}"
            ) from error
        metadata["page_count"] = len(reader.pages)
    else:
        content = document_path.read_text(encoding="utf-8")

    logger.info("Loaded document %s", document_path)
    return Document(
        content=content,
        source=document_path,
        file_type=extension.removeprefix("."),
        metadata=metadata,
    )
