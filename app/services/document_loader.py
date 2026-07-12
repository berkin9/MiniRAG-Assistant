"""Utilities for loading local text documents."""

from pathlib import Path


def load_document(path: str | Path) -> str:
    """Read a UTF-8 text document from disk."""
    document_path = Path(path)
    if not document_path.is_file():
        raise FileNotFoundError(f"Document not found: {document_path}")
    return document_path.read_text(encoding="utf-8")
