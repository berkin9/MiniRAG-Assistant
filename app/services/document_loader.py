"""Utilities for loading supported local documents."""

import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.config import SUPPORTED_EXTENSIONS
from app.models.document import Document, DocumentPage

logger = logging.getLogger(__name__)


class UnsupportedFileTypeError(ValueError):
    """Raised when a document type is not supported."""


class DocumentLoadError(RuntimeError):
    """Raised when text cannot be extracted from a supported document."""


class EmptyDocumentError(DocumentLoadError):
    """Raised when a document contains no extractable text."""


def _load_pdf(path: Path) -> tuple[DocumentPage, ...]:
    """Extract text from a PDF one page at a time."""
    try:
        reader = PdfReader(path)
        return tuple(
            DocumentPage(content=page.extract_text() or "", page_number=index)
            for index, page in enumerate(reader.pages, start=1)
        )
    except PyPdfError as error:
        raise DocumentLoadError(f"Could not read PDF document: {path}") from error


def _load_text(path: Path) -> tuple[DocumentPage, ...]:
    """Read a UTF-8 text document as a single logical page."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise DocumentLoadError(
            f"Document is not valid UTF-8 text: {path}"
        ) from error
    return (DocumentPage(content=content),)


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
    pages = _load_pdf(document_path) if extension == ".pdf" else _load_text(document_path)
    if not any(page.content.strip() for page in pages):
        raise EmptyDocumentError(f"Document contains no extractable text: {document_path}")
    if extension == ".pdf":
        metadata["page_count"] = len(pages)

    content = "\n\n".join(page.content for page in pages)

    logger.info("Loaded document %s", document_path)
    return Document(
        content=content,
        source=document_path,
        file_type=extension.removeprefix("."),
        metadata=metadata,
        pages=pages,
    )
