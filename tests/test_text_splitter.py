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
