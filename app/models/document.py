"""Document models used by the ingestion pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentPage:
    """Text extracted from one document page or text file."""

    content: str
    page_number: int | None = None


@dataclass(frozen=True)
class Document:
    """Text and metadata extracted from a local document."""

    content: str
    source: Path
    file_type: str
    metadata: dict[str, str | int]
    pages: tuple[DocumentPage, ...]
