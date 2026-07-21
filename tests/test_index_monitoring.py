"""Tests for safe, read-only indexed-document monitoring."""

from collections.abc import Mapping
from pathlib import Path

from app.services.collections import CollectionRegistry
from app.services.index_monitoring import (
    load_indexed_data_summary,
    summarize_indexed_data,
)
from app.services.vector_store import ChromaCollectionInspection


class FakeInspector:
    """Return controlled existing collections and metadata snapshots."""

    def __init__(
        self,
        collections: Mapping[str, ChromaCollectionInspection],
        failing: tuple[str, ...] = (),
    ) -> None:
        self.collections = collections
        self.failing = failing

    def list_collection_names(self) -> tuple[str, ...]:
        return tuple(self.collections)

    def inspect_collection(self, name: str) -> ChromaCollectionInspection:
        if name in self.failing:
            raise RuntimeError("database detail must not be exposed")
        return self.collections[name]


def _registry() -> CollectionRegistry:
    return CollectionRegistry(
        "minirag_documents",
        "general",
        ("general", "project", "technical", "policies"),
    )


def test_all_configured_collections_are_shown_in_registry_order() -> None:
    summary = summarize_indexed_data(_registry(), FakeInspector({}))

    assert tuple(item.logical_name for item in summary.collections) == (
        "general",
        "project",
        "technical",
        "policies",
    )
    assert all(item.status == "missing" for item in summary.collections)


def test_empty_and_missing_physical_collections_are_distinguished() -> None:
    summary = summarize_indexed_data(
        _registry(),
        FakeInspector(
            {"minirag_documents": ChromaCollectionInspection(0, ())}
        ),
    )

    assert summary.collections[0].status == "empty"
    assert summary.collections[0].chunk_count == 0
    assert summary.collections[1].status == "missing"


def test_filenames_are_deduplicated_sorted_and_chunks_are_counted() -> None:
    metadata = (
        {"filename": "Zulu.pdf", "source_file": "/private/Zulu.pdf"},
        {"filename": "alpha.txt"},
        {"filename": "Zulu.pdf"},
        {"source_file": "/secret/path/fallback.md"},
    )
    summary = summarize_indexed_data(
        _registry(),
        FakeInspector(
            {"minirag_documents": ChromaCollectionInspection(42, metadata)}
        ),
    )
    general = summary.collections[0]

    assert general.status == "indexed"
    assert general.chunk_count == 42
    assert general.document_count == 3
    assert general.filenames == ("alpha.txt", "fallback.md", "Zulu.pdf")
    assert summary.active_collection_count == 1
    assert summary.unique_document_count == 3
    assert summary.total_chunk_count == 42


def test_unexpected_metadata_is_ignored_without_exposing_content() -> None:
    metadata = (
        {},
        {"filename": None},
        {"filename": 123},
        {"source_file": ""},
        {"document": "TOP SECRET FULL CHUNK"},
        {"filename": "/hidden/location/safe.pdf", "api_key": "secret"},
    )
    summary = summarize_indexed_data(
        _registry(),
        FakeInspector(
            {"minirag_documents": ChromaCollectionInspection(6, metadata)}
        ),
    )
    rendered_values = repr(summary)

    assert summary.collections[0].filenames == ("safe.pdf",)
    assert "/hidden/location" not in rendered_values
    assert "TOP SECRET" not in rendered_values
    assert "api_key" not in rendered_values


def test_collection_and_database_access_failures_are_graceful() -> None:
    physical = "minirag_documents"
    partial = summarize_indexed_data(
        _registry(),
        FakeInspector(
            {physical: ChromaCollectionInspection(1, ())}, failing=(physical,)
        ),
    )
    unavailable = load_indexed_data_summary(
        Path("unused"),
        _registry(),
        inspector_factory=lambda path: (_ for _ in ()).throw(
            RuntimeError(f"cannot access {path}")
        ),
    )

    assert partial.collections[0].status == "unavailable"
    assert all(item.status == "unavailable" for item in unavailable.collections)
    assert "cannot access" not in repr(unavailable)
