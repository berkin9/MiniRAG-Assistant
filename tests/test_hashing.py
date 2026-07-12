"""Tests for stable content hashing."""

from pathlib import Path

from app.services.hashing import hash_file


def test_file_hash_is_stable_and_content_based(tmp_path: Path) -> None:
    """Equal bytes should hash equally and changed bytes should not."""
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same content", encoding="utf-8")
    second.write_text("same content", encoding="utf-8")

    original_hash = hash_file(first)

    assert original_hash == hash_file(first)
    assert original_hash == hash_file(second)

    second.write_text("different content", encoding="utf-8")
    assert original_hash != hash_file(second)
