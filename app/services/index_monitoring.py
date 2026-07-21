"""Read-only summaries of configured Chroma document collections."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from app.services.collections import CollectionRegistry
from app.services.vector_store import (
    ChromaCollectionInspection,
    ChromaIndexInspector,
    VectorStoreError,
)

CollectionStatus = Literal["indexed", "empty", "missing", "unavailable"]


class IndexInspector(Protocol):
    """Read behavior required by the index monitoring service."""

    def list_collection_names(self) -> tuple[str, ...]:
        """Return physical collection names that currently exist."""

    def inspect_collection(self, name: str) -> ChromaCollectionInspection:
        """Return a read-only count and metadata snapshot."""


@dataclass(frozen=True)
class IndexedCollectionSummary:
    """Public-safe monitoring data for one configured collection."""

    logical_name: str
    physical_name: str
    status: CollectionStatus
    chunk_count: int = 0
    filenames: tuple[str, ...] = ()

    @property
    def document_count(self) -> int:
        """Return the number of unique safe filenames."""
        return len(self.filenames)


@dataclass(frozen=True)
class IndexedDataSummary:
    """Aggregate public-safe monitoring data for the configured registry."""

    collections: tuple[IndexedCollectionSummary, ...]

    @property
    def configured_collection_count(self) -> int:
        return len(self.collections)

    @property
    def active_collection_count(self) -> int:
        return sum(item.status == "indexed" for item in self.collections)

    @property
    def unique_document_count(self) -> int:
        return len(
            {
                filename
                for item in self.collections
                for filename in item.filenames
            }
        )

    @property
    def total_chunk_count(self) -> int:
        return sum(item.chunk_count for item in self.collections)

    @property
    def has_access_errors(self) -> bool:
        return any(item.status == "unavailable" for item in self.collections)


def load_indexed_data_summary(
    persist_directory: str | Path,
    registry: CollectionRegistry,
    inspector_factory: Callable[[str | Path], IndexInspector] = ChromaIndexInspector,
) -> IndexedDataSummary:
    """Build a safe summary, keeping the app usable when Chroma is unavailable."""
    try:
        inspector = inspector_factory(persist_directory)
        return summarize_indexed_data(registry, inspector)
    except (OSError, RuntimeError, ValueError):
        return _all_unavailable(registry)


def summarize_indexed_data(
    registry: CollectionRegistry, inspector: IndexInspector
) -> IndexedDataSummary:
    """Inspect configured collections in registry order without mutating Chroma."""
    try:
        existing_names = set(inspector.list_collection_names())
    except (OSError, RuntimeError, ValueError):
        return _all_unavailable(registry)

    summaries: list[IndexedCollectionSummary] = []
    for logical_name in registry.list_collections():
        physical_name = registry.physical_name(logical_name)
        if physical_name not in existing_names:
            summaries.append(
                IndexedCollectionSummary(logical_name, physical_name, "missing")
            )
            continue
        try:
            inspection = inspector.inspect_collection(physical_name)
            filenames = _extract_filenames(inspection.metadatas)
            status: CollectionStatus = (
                "indexed" if inspection.chunk_count > 0 else "empty"
            )
            summaries.append(
                IndexedCollectionSummary(
                    logical_name,
                    physical_name,
                    status,
                    max(0, inspection.chunk_count),
                    filenames,
                )
            )
        except (OSError, RuntimeError, ValueError):
            summaries.append(
                IndexedCollectionSummary(logical_name, physical_name, "unavailable")
            )
    return IndexedDataSummary(tuple(summaries))


def _extract_filenames(metadatas: tuple[Mapping[str, object], ...]) -> tuple[str, ...]:
    """Return deduplicated safe basenames from the established metadata fields."""
    filenames: set[str] = set()
    for metadata in metadatas:
        candidate = metadata.get("filename")
        if not isinstance(candidate, str) or not candidate.strip():
            candidate = metadata.get("source_file")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        basename = Path(candidate.replace("\\", "/")).name.strip()
        if basename and basename not in {".", ".."}:
            filenames.add(basename)
    return tuple(sorted(filenames, key=str.casefold))


def _all_unavailable(registry: CollectionRegistry) -> IndexedDataSummary:
    """Represent a database-level failure without leaking its internal details."""
    return IndexedDataSummary(
        tuple(
            IndexedCollectionSummary(
                logical_name,
                registry.physical_name(logical_name),
                "unavailable",
            )
            for logical_name in registry.list_collections()
        )
    )
