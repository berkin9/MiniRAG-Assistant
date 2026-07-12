"""Tests for the ingestion pipeline."""

from pathlib import Path

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
