"""Tests for the ingestion pipeline."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import document_loader
from app.services.ingestion import ingest_directory


def test_ingestion_preserves_metadata_and_chunk_indexes(tmp_path: Path) -> None:
    """Every chunk should retain source metadata and a sequential index."""
    source = tmp_path / "sample.txt"
    source.write_text("abcdefghij", encoding="utf-8")

    results = ingest_directory(tmp_path, chunk_size=4, chunk_overlap=1)

    assert [result.text for result in results] == ["abcd", "defg", "ghij"]
    assert [result.chunk_index for result in results] == [0, 1, 2]
    assert all(result.source == source for result in results)
    assert all(result.metadata["filename"] == "sample.txt" for result in results)
    assert [result.metadata["chunk_index"] for result in results] == [0, 1, 2]
    assert all(result.metadata["source_file"] == str(source) for result in results)
    assert all(result.metadata["file_type"] == "txt" for result in results)
    assert all(result.metadata["page_number"] is None for result in results)


def test_ingestion_keeps_pdf_page_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PDF chunks should reference the page that produced them."""
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"fake pdf")
    pages = [
        SimpleNamespace(extract_text=lambda: "First page"),
        SimpleNamespace(extract_text=lambda: "Second page"),
    ]
    monkeypatch.setattr(
        document_loader, "PdfReader", lambda _: SimpleNamespace(pages=pages)
    )

    results = ingest_directory(tmp_path, chunk_size=100, chunk_overlap=10)

    assert [result.metadata["page_number"] for result in results] == [1, 2]
    assert [result.metadata["chunk_index"] for result in results] == [0, 1]
