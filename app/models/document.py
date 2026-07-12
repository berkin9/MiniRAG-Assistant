"""Document models used by the ingestion pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """Text and metadata extracted from a local document."""

    content: str
    source: Path
    file_type: str
    metadata: dict[str, str | int]
