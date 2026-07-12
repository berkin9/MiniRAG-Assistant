"""Tests for document loading and discovery."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf.errors import PdfReadError

from app.services import document_loader
from app.services.document_loader import (
    DocumentLoadError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    load_document,
)
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
    assert [page.page_number for page in document.pages] == [1, 2]
    assert [page.content for page in document.pages] == ["Page one", "Page two"]


def test_load_document_rejects_unsupported_type(tmp_path: Path) -> None:
    """Unsupported extensions should produce a clear error."""
    path = tmp_path / "image.png"
    path.write_bytes(b"content")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        load_document(path)


@pytest.mark.parametrize("filename", ["empty.txt", "empty.pdf"])
def test_load_document_rejects_empty_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    """Files without extractable text should produce a clear error."""
    path = tmp_path / filename
    path.write_bytes(b"" if path.suffix == ".txt" else b"fake pdf")
    if path.suffix == ".pdf":
        page = SimpleNamespace(extract_text=lambda: "  ")
        monkeypatch.setattr(
            document_loader, "PdfReader", lambda _: SimpleNamespace(pages=[page])
        )

    with pytest.raises(EmptyDocumentError, match="no extractable text"):
        load_document(path)


def test_load_document_rejects_corrupted_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unreadable PDFs should produce a meaningful loading error."""
    path = tmp_path / "corrupted.pdf"
    path.write_bytes(b"not a pdf")

    def raise_pdf_error(_: Path) -> None:
        raise PdfReadError("broken PDF")

    monkeypatch.setattr(document_loader, "PdfReader", raise_pdf_error)

    with pytest.raises(DocumentLoadError, match="Could not read PDF"):
        load_document(path)


def test_load_document_rejects_non_utf8_text(tmp_path: Path) -> None:
    """Invalid UTF-8 text should be reported as a corrupted document."""
    path = tmp_path / "corrupted.txt"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(DocumentLoadError, match="not valid UTF-8"):
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
