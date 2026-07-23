"""Tests for predefined shared demo document indexing."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.config import Settings
from app.services.collections import CollectionRegistry
from app.services.demo_indexing import (
    DemoIndexingResult,
    discover_demo_documents,
    ensure_demo_documents_indexed,
)
from app.services.vector_store import ChunkMetadata


class FixedEmbedder:
    """Create deterministic vectors without downloading a model."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


class MemoryStore:
    """Retain document hashes across repeated demo-index checks."""

    def __init__(self) -> None:
        self.document_hashes: set[str] = set()
        self.chunk_count = 0

    def has_document(self, document_hash: str) -> bool:
        return document_hash in self.document_hashes

    def add_chunks(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[ChunkMetadata],
    ) -> None:
        del texts, embeddings
        self.chunk_count += len(ids)
        self.document_hashes.update(
            str(metadata["document_hash"]) for metadata in metadatas
        )


def _registry() -> CollectionRegistry:
    return CollectionRegistry(
        "demo_test",
        "general",
        ("general", "project", "technical", "policies"),
    )


def _write(path: Path, content: str = "Fictional demo content.") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discovery_maps_registered_directories_in_deterministic_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "demo"
    _write(root / "technical" / "zulu.md")
    _write(root / "technical" / "alpha.txt")
    _write(root / "general" / "overview.md")
    _write(root / "unknown" / "ignored.md")
    _write(root / "project" / "ignored.json")

    documents = discover_demo_documents(root, _registry())

    assert tuple((item.collection, item.path.name) for item in documents) == (
        ("general", "overview.md"),
        ("technical", "alpha.txt"),
        ("technical", "zulu.md"),
    )


def test_first_run_indexes_and_second_run_skips_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    _write(root / "general" / "overview.md")
    _write(root / "project" / "plan.md")
    stores: dict[str, MemoryStore] = {}
    embedder = FixedEmbedder()
    settings = Settings(demo_data_dir=root)

    first = ensure_demo_documents_indexed(
        settings,
        _registry(),
        embedder_factory=lambda model: embedder,
        store_factory=lambda collection: stores.setdefault(
            collection, MemoryStore()
        ),
    )
    second = ensure_demo_documents_indexed(
        settings,
        _registry(),
        embedder_factory=lambda model: embedder,
        store_factory=lambda collection: stores.setdefault(
            collection, MemoryStore()
        ),
    )

    assert first == DemoIndexingResult(2, 2, 0, 0)
    assert second == DemoIndexingResult(2, 0, 2, 0)
    assert set(stores) == {"general", "project"}
    assert embedder.calls == 2


def test_one_failed_document_does_not_block_later_documents(
    tmp_path: Path,
) -> None:
    root = tmp_path / "demo"
    _write(root / "general" / "empty.md", "")
    _write(root / "technical" / "valid.md")
    stores: dict[str, MemoryStore] = {}

    result = ensure_demo_documents_indexed(
        Settings(demo_data_dir=root),
        _registry(),
        embedder_factory=lambda model: FixedEmbedder(),
        store_factory=lambda collection: stores.setdefault(
            collection, MemoryStore()
        ),
    )

    assert result.discovered_documents == 2
    assert result.indexed_documents == 1
    assert result.failed_documents == 1
    assert result.errors == ("general/empty.md: EmptyDocumentError",)
    assert str(tmp_path) not in repr(result)
    assert "Fictional demo content" not in repr(result)


def test_disabled_mode_does_not_scan_or_build_services(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    _write(root / "general" / "document.md")

    result = ensure_demo_documents_indexed(
        Settings(
            demo_data_dir=root,
            auto_index_demo_documents=False,
        ),
        _registry(),
        embedder_factory=lambda model: (_ for _ in ()).throw(
            AssertionError("embedding service must not be built")
        ),
        store_factory=lambda collection: (_ for _ in ()).throw(
            AssertionError("vector store must not be built")
        ),
    )

    assert result == DemoIndexingResult(0, 0, 0, 0)


def test_upload_directory_is_never_scanned(tmp_path: Path) -> None:
    demo_root = tmp_path / "demo"
    upload_root = tmp_path / "uploads"
    _write(demo_root / "policies" / "policy.md")
    _write(upload_root / "private-upload.md")

    documents = discover_demo_documents(demo_root, _registry())
    result = ensure_demo_documents_indexed(
        Settings(demo_data_dir=demo_root, upload_dir=upload_root),
        _registry(),
        embedder_factory=lambda model: FixedEmbedder(),
        store_factory=lambda collection: MemoryStore(),
    )

    assert tuple(item.path.name for item in documents) == ("policy.md",)
    assert result.discovered_documents == 1
    assert "private-upload.md" not in repr(result)


def test_demo_indexing_never_constructs_an_llm_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import llm_providers

    root = tmp_path / "demo"
    _write(root / "general" / "document.md")
    monkeypatch.setattr(
        llm_providers,
        "build_llm_provider",
        lambda settings: (_ for _ in ()).throw(
            AssertionError("LLM provider must not be constructed")
        ),
    )

    result = ensure_demo_documents_indexed(
        Settings(demo_data_dir=root),
        _registry(),
        embedder_factory=lambda model: FixedEmbedder(),
        store_factory=lambda collection: MemoryStore(),
    )

    assert result.indexed_documents == 1
