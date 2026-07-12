"""Tests for text chunking."""

import pytest

from app.services.text_splitter import split_text


def test_split_text_creates_overlapping_chunks() -> None:
    """Each chunk should retain the configured preceding overlap."""
    assert split_text("abcdefghij", chunk_size=4, overlap=1) == [
        "abcd",
        "defg",
        "ghij",
    ]


def test_split_text_rejects_invalid_overlap() -> None:
    """Overlap must leave room to advance through the text."""
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        split_text("text", chunk_size=4, overlap=4)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (4, -1)],
)
def test_split_text_rejects_invalid_sizes(chunk_size: int, overlap: int) -> None:
    """Chunk sizes and overlaps must be valid."""
    with pytest.raises(ValueError):
        split_text("text", chunk_size=chunk_size, overlap=overlap)


def test_split_text_prefers_paragraph_boundaries() -> None:
    """Paragraph breaks should be preferred over character cuts."""
    text = "First paragraph.\n\nSecond paragraph."
    chunks = split_text(text, chunk_size=20, overlap=0)

    assert chunks == ["First paragraph.\n\n", "Second paragraph."]
