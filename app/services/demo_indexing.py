"""Discovery and indexing for predefined shared demo documents."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.services.collections import CollectionRegistry
from app.services.embeddings import EmbeddingService
from app.services.indexing import (
    DocumentEmbedder,
    IndexVectorStore,
    IndexingResult,
    index_document,
)
from app.services.ingestion import discover_documents
from app.services.vector_store import ChromaVectorStore


class DemoIndexingError(RuntimeError):
    """Raised when demo document discovery cannot start safely."""


class DocumentIndexer(Protocol):
    """Existing indexing operation required by the demo service."""

    def __call__(
        self,
        path: str | Path,
        chunk_size: int,
        chunk_overlap: int,
        embedder: DocumentEmbedder,
        vector_store: IndexVectorStore,
        collection: str = "general",
    ) -> IndexingResult:
        """Index one document through the shared pipeline."""


@dataclass(frozen=True)
class DemoDocument:
    """One discovered demo file and its registered logical collection."""

    path: Path
    collection: str


@dataclass(frozen=True)
class DemoIndexingResult:
    """Safe aggregate outcome for one demo-indexing check."""

    discovered_documents: int
    indexed_documents: int
    skipped_documents: int
    failed_documents: int
    errors: tuple[str, ...] = ()


def discover_demo_documents(
    demo_directory: str | Path,
    registry: CollectionRegistry,
) -> tuple[DemoDocument, ...]:
    """Find supported files only below registered collection directories."""
    root = Path(demo_directory)
    if not root.exists():
        return ()
    if not root.is_dir():
        raise DemoIndexingError("Demo data path is not a directory")

    discovered: list[DemoDocument] = []
    try:
        for collection in sorted(registry.list_collections()):
            collection_directory = root / collection
            if not collection_directory.is_dir():
                continue
            discovered.extend(
                DemoDocument(path, collection)
                for path in discover_documents(collection_directory)
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise DemoIndexingError(
            f"Demo document discovery failed: {type(error).__name__}"
        ) from error
    return tuple(
        sorted(
            discovered,
            key=lambda item: (item.collection, item.path.as_posix()),
        )
    )


def ensure_demo_documents_indexed(
    settings: Settings,
    registry: CollectionRegistry,
    *,
    embedder_factory: Callable[[str], DocumentEmbedder] = EmbeddingService,
    store_factory: Callable[[str], IndexVectorStore] | None = None,
    indexer: DocumentIndexer = index_document,
) -> DemoIndexingResult:
    """Index each predefined demo file once and isolate expected file failures."""
    if not settings.auto_index_demo_documents:
        return DemoIndexingResult(0, 0, 0, 0)

    documents = discover_demo_documents(settings.demo_data_dir, registry)
    if not documents:
        return DemoIndexingResult(0, 0, 0, 0)

    build_store = store_factory or (
        lambda collection: ChromaVectorStore(
            settings.chroma_persist_dir,
            registry.physical_name(collection),
        )
    )
    embedder: DocumentEmbedder | None = None
    stores: dict[str, IndexVectorStore] = {}
    indexed = 0
    skipped = 0
    failed = 0
    errors: list[str] = []

    for document in documents:
        try:
            if embedder is None:
                embedder = embedder_factory(settings.embedding_model)
            if document.collection not in stores:
                stores[document.collection] = build_store(document.collection)
            result = indexer(
                document.path,
                settings.chunk_size,
                settings.chunk_overlap,
                embedder,
                stores[document.collection],
                document.collection,
            )
            if result.status == "indexed":
                indexed += 1
            else:
                skipped += 1
        except (OSError, RuntimeError, UnicodeError, ValueError) as error:
            failed += 1
            errors.append(_safe_error(document, error))

    return DemoIndexingResult(
        discovered_documents=len(documents),
        indexed_documents=indexed,
        skipped_documents=skipped,
        failed_documents=failed,
        errors=tuple(errors),
    )


def _safe_error(document: DemoDocument, error: Exception) -> str:
    """Describe a failed demo file without paths, content, or exception details."""
    return f"{document.collection}/{document.path.name}: {type(error).__name__}"
