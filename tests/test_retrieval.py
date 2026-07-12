"""Tests for structured semantic retrieval."""

from collections.abc import Sequence

import pytest

from app.services.retrieval import retrieve
from app.services.vector_store import VectorSearchResult


class FakeQueryEmbedder:
    """Return a deterministic query vector."""

    def embed_query(self, query: str) -> list[float]:
        """Return one fixed vector."""
        del query
        return [1.0, 0.0]


class FakeSearchStore:
    """Return deterministic results while recording top-k."""

    def __init__(self) -> None:
        self.top_k = 0

    def search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchResult]:
        """Return ranked fake matches."""
        del query_embedding
        self.top_k = top_k
        matches = [
            VectorSearchResult(
                "closest",
                {
                    "source_file": "guide.pdf",
                    "file_type": "pdf",
                    "page_number": 2,
                    "chunk_index": 3,
                    "document_hash": "abc",
                },
                0.2,
            ),
            VectorSearchResult(
                "too far",
                {
                    "source_file": "notes.txt",
                    "file_type": "txt",
                    "page_number": 0,
                    "chunk_index": 0,
                    "document_hash": "def",
                },
                1.5,
            ),
        ]
        return matches[:top_k]


def test_retrieval_is_structured_ranked_and_thresholded() -> None:
    """Relevant matches should retain metadata and ascending distance."""
    store = FakeSearchStore()

    response = retrieve("deadline", 2, 1.2, FakeQueryEmbedder(), store)

    assert store.top_k == 2
    assert len(response.results) == 1
    result = response.results[0]
    assert result.text == "closest"
    assert result.page_number == 2
    assert result.chunk_index == 3
    assert result.distance == 0.2


def test_no_relevant_results_returns_structured_empty_response() -> None:
    """A strict threshold should return an empty results tuple."""
    response = retrieve("deadline", 4, 0.1, FakeQueryEmbedder(), FakeSearchStore())

    assert response.query == "deadline"
    assert response.results == ()


def test_empty_query_is_rejected() -> None:
    """Whitespace-only searches should not be embedded."""
    with pytest.raises(ValueError, match="must not be empty"):
        retrieve("   ", 4, 1.2, FakeQueryEmbedder(), FakeSearchStore())
