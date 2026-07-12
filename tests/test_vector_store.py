"""Integration tests for persistent ChromaDB storage."""

from pathlib import Path

import pytest

from app.services.retrieval import retrieve
from app.services.vector_store import ChromaVectorStore, normalize_metadata


class FixedQueryEmbedder:
    """Return a vector nearest to the PDF test chunk."""

    def embed_query(self, query: str) -> list[float]:
        """Return the fixed PDF query vector."""
        del query
        return [0.0, 1.0]


def test_metadata_normalizes_optional_page_number() -> None:
    """A missing page should use zero, which PDF page numbering never uses."""
    assert normalize_metadata({"page_number": None, "value": None}) == {
        "page_number": 0,
        "value": "",
    }


def test_persistent_store_reopens_and_preserves_pdf_metadata(tmp_path: Path) -> None:
    """Stored chunks and PDF page numbers should survive reopening."""
    persist_directory = tmp_path / "chroma"
    store = ChromaVectorStore(persist_directory, "test_documents")
    store.add_chunks(
        ids=["txt:0", "pdf:0"],
        texts=["text notes", "project deadline"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {
                "source_file": "notes.txt",
                "file_type": "txt",
                "page_number": None,
                "chunk_index": 0,
                "document_hash": "txt",
            },
            {
                "source_file": "guide.pdf",
                "file_type": "pdf",
                "page_number": 4,
                "chunk_index": 0,
                "document_hash": "pdf",
            },
        ],
    )

    reopened = ChromaVectorStore(persist_directory, "test_documents")
    response = retrieve(
        "deadline", 1, 1.2, FixedQueryEmbedder(), reopened
    )

    assert reopened.count() == 2
    assert reopened.has_document("pdf")
    assert len(response.results) == 1
    assert response.results[0].source_file == "guide.pdf"
    assert response.results[0].page_number == 4
    assert response.results[0].distance == pytest.approx(0.0)
