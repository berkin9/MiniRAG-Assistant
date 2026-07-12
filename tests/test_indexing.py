"""Tests for document indexing and duplicate prevention."""

from collections.abc import Sequence
from pathlib import Path

from app.services.indexing import index_document
from app.services.vector_store import ChunkMetadata


class FakeEmbedder:
    """Deterministic embedder that records calls."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Represent each text using its length."""
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


class FakeVectorStore:
    """In-memory vector store for indexing tests."""

    def __init__(self) -> None:
        self.hashes: set[str] = set()
        self.ids: list[str] = []
        self.metadatas: list[ChunkMetadata] = []

    def has_document(self, document_hash: str) -> bool:
        """Check an in-memory hash set."""
        return document_hash in self.hashes

    def add_chunks(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[ChunkMetadata],
    ) -> None:
        """Record stored IDs and metadata."""
        del texts, embeddings
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)
        self.hashes.add(str(metadatas[0]["document_hash"]))


def test_reindexing_same_file_does_not_embed_or_duplicate(tmp_path: Path) -> None:
    """A known content hash should skip embedding and storage."""
    source = tmp_path / "document.txt"
    source.write_text("alpha beta gamma", encoding="utf-8")
    embedder = FakeEmbedder()
    store = FakeVectorStore()

    first = index_document(source, 10, 2, embedder, store)
    stored_ids = list(store.ids)
    second = index_document(source, 10, 2, embedder, store)

    assert first.status == "indexed"
    assert first.stored_chunks > 0
    assert second.status == "already_indexed"
    assert second.stored_chunks == 0
    assert store.ids == stored_ids
    assert embedder.calls == 1
    assert all(item_id.startswith(first.document_hash) for item_id in store.ids)
    assert all("document_hash" in metadata for metadata in store.metadatas)


def test_changed_file_with_same_name_is_new_content(tmp_path: Path) -> None:
    """Deduplication should use bytes rather than the filename."""
    source = tmp_path / "document.txt"
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    source.write_text("first version", encoding="utf-8")
    first = index_document(source, 100, 10, embedder, store)

    source.write_text("second version", encoding="utf-8")
    second = index_document(source, 100, 10, embedder, store)

    assert first.document_hash != second.document_hash
    assert second.status == "indexed"
    assert embedder.calls == 2
