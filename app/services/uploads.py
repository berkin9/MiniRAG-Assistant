"""Safe upload persistence and indexing orchestration."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.config import SUPPORTED_EXTENSIONS
from app.services.indexing import (
    DocumentEmbedder,
    IndexVectorStore,
    IndexingResult,
    index_document,
)


class UploadError(ValueError):
    """Raised when uploaded content is unsafe or unsupported."""


@dataclass(frozen=True)
class UploadData:
    """Provider-independent uploaded file bytes."""

    filename: str
    content: bytes


@dataclass(frozen=True)
class UploadIndexResult:
    """Outcome of saving and indexing one upload."""

    filename: str
    saved_path: Path | None
    status: Literal["indexed", "already_indexed", "failed"]
    stored_chunks: int
    error: str | None = None


def sanitize_filename(filename: str) -> str:
    """Return a traversal-safe filename with conservative characters."""
    basename = Path(filename.replace("\\", "/")).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    if not safe_name:
        raise UploadError("Upload filename is invalid")
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UploadError(f"Unsupported upload file type: {extension or '<none>'}")
    return safe_name


def save_upload(
    upload: UploadData, upload_directory: str | Path, max_size_bytes: int
) -> Path:
    """Save upload bytes under a content-addressed, sanitized filename."""
    if max_size_bytes <= 0:
        raise ValueError("max_size_bytes must be greater than zero")
    if not upload.content:
        raise UploadError(f"Uploaded file is empty: {upload.filename}")
    if len(upload.content) > max_size_bytes:
        raise UploadError(f"Uploaded file exceeds the size limit: {upload.filename}")

    safe_name = sanitize_filename(upload.filename)
    digest = hashlib.sha256(upload.content).hexdigest()[:16]
    directory = Path(upload_directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}-{safe_name}"
    if destination.exists():
        if destination.read_bytes() != upload.content:
            raise UploadError(f"Upload destination conflict: {safe_name}")
        return destination
    destination.write_bytes(upload.content)
    return destination


def index_uploads(
    uploads: Sequence[UploadData],
    upload_directory: str | Path,
    max_size_bytes: int,
    chunk_size: int,
    chunk_overlap: int,
    embedder: DocumentEmbedder,
    vector_store: IndexVectorStore,
) -> list[UploadIndexResult]:
    """Save and independently index multiple uploads using existing services."""
    results: list[UploadIndexResult] = []
    for upload in uploads:
        saved_path: Path | None = None
        try:
            saved_path = save_upload(upload, upload_directory, max_size_bytes)
            indexed: IndexingResult = index_document(
                saved_path, chunk_size, chunk_overlap, embedder, vector_store
            )
            results.append(
                UploadIndexResult(
                    filename=upload.filename,
                    saved_path=saved_path,
                    status=indexed.status,
                    stored_chunks=indexed.stored_chunks,
                )
            )
        except (OSError, RuntimeError, ValueError) as error:
            results.append(
                UploadIndexResult(
                    filename=upload.filename,
                    saved_path=saved_path,
                    status="failed",
                    stored_chunks=0,
                    error=str(error),
                )
            )
    return results
