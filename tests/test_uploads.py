"""Tests for safe uploaded-document orchestration."""

from collections.abc import Sequence
from pathlib import Path

from app.services.uploads import UploadData, index_uploads, save_upload
from app.services.vector_store import ChunkMetadata


class FakeEmbedder:
    """Create deterministic upload embeddings."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class FakeStore:
    """Track content hashes and stored chunks in memory."""

    def __init__(self) -> None:
        self.hashes: set[str] = set()
        self.chunk_count = 0

    def has_document(self, document_hash: str) -> bool:
        return document_hash in self.hashes

    def add_chunks(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[ChunkMetadata],
    ) -> None:
        del texts, embeddings
        self.chunk_count += len(ids)
        self.hashes.add(str(metadatas[0]["document_hash"]))


def test_safe_and_traversal_filenames_are_stored_inside_upload_dir(
    tmp_path: Path,
) -> None:
    """Traversal components should never control the destination directory."""
    upload_directory = tmp_path / "uploads"

    regular = save_upload(UploadData("notes.txt", b"notes"), upload_directory, 100)
    traversal = save_upload(
        UploadData("../../secret.txt", b"secret"), upload_directory, 100
    )

    assert regular.parent == upload_directory
    assert traversal.parent == upload_directory
    assert ".." not in traversal.name
    assert traversal.name.endswith("-secret.txt")


def test_unsupported_upload_is_reported_as_failed(tmp_path: Path) -> None:
    """Unsupported content should fail without stopping other orchestration."""
    results = index_uploads(
        [UploadData("image.png", b"image")],
        tmp_path,
        100,
        100,
        10,
        FakeEmbedder(),
        FakeStore(),
    )

    assert results[0].status == "failed"
    assert "Unsupported" in str(results[0].error)


def test_duplicate_and_multiple_uploads_index_independently(tmp_path: Path) -> None:
    """Duplicate bytes should skip storage while distinct uploads still index."""
    store = FakeStore()
    uploads = [
        UploadData("first.txt", b"same content"),
        UploadData("duplicate.txt", b"same content"),
        UploadData("second.md", b"different content"),
    ]

    results = index_uploads(
        uploads, tmp_path, 1_000, 100, 10, FakeEmbedder(), store
    )

    assert [result.status for result in results] == [
        "indexed",
        "already_indexed",
        "indexed",
    ]
    assert store.chunk_count == 2
