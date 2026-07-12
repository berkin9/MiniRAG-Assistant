"""Tests for document loading and discovery."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import document_loader
from app.services.document_loader import UnsupportedFileTypeError, load_document
from app.services.ingestion import DirectoryNotFoundError, discover_documents


@pytest.mark.parametrize(
    ("filename", "content", "file_type"),
    [("notes.txt", "plain text", "txt"), ("guide.md", "# Guide", "md")],
)
def test_load_text_documents(
    tmp_path: Path, filename: str, content: str, file_type: str
) -> None:
    """UTF-8 text and Markdown files should include useful metadata."""
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")

    document = load_document(path)

    assert document.content == content
    assert document.source == path
    assert document.file_type == file_type
    assert document.metadata == {
        "filename": filename,
        "extension": path.suffix,
        "file_size_bytes": len(content.encode("utf-8")),
    }


def test_load_pdf_extracts_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PDF pages should be extracted and counted."""
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"fake pdf")
    pages = [
        SimpleNamespace(extract_text=lambda: "Page one"),
        SimpleNamespace(extract_text=lambda: "Page two"),
    ]
    monkeypatch.setattr(
        document_loader, "PdfReader", lambda _: SimpleNamespace(pages=pages)
    )

    document = load_document(path)

    assert document.content == "Page one\n\nPage two"
    assert document.file_type == "pdf"
    assert document.metadata["page_count"] == 2


def test_load_document_rejects_unsupported_type(tmp_path: Path) -> None:
    """Unsupported extensions should produce a clear error."""
    path = tmp_path / "image.png"
    path.write_bytes(b"content")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        load_document(path)


def test_discover_documents_recursively_and_ignore_hidden(tmp_path: Path) -> None:
    """Discovery should be sorted, recursive, and exclude hidden paths."""
    nested = tmp_path / "nested"
    nested.mkdir()
    hidden_directory = tmp_path / ".private"
    hidden_directory.mkdir()
    for relative_path in (
        "z.md",
        "nested/a.txt",
        ".hidden.txt",
        ".private/secret.pdf",
        "ignored.csv",
    ):
        path = tmp_path / relative_path
        path.write_text("content", encoding="utf-8")

    assert discover_documents(tmp_path) == [nested / "a.txt", tmp_path / "z.md"]


def test_discover_documents_rejects_missing_directory(tmp_path: Path) -> None:
    """A missing discovery root should produce a clear error."""
    with pytest.raises(DirectoryNotFoundError, match="Directory not found"):
        discover_documents(tmp_path / "missing")
