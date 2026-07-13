"""Orchestration for embedding and indexing ingested documents."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from app.services.collections import normalize_collection_name
from app.services.hashing import hash_file
from app.services.ingestion import discover_documents, ingest_document
from app.services.vector_store import ChunkMetadata


class IndexingError(RuntimeError):
    """Raised when a document fails during directory indexing."""


class DocumentEmbedder(Protocol):
    """Embedding behavior needed by document indexing."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunk texts."""


class IndexVectorStore(Protocol):
    """Vector-store behavior needed by document indexing."""

    def has_document(self, document_hash: str) -> bool:
        """Check whether a document is already indexed."""

    def add_chunks(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[ChunkMetadata],
    ) -> None:
        """Store embedded document chunks."""


@dataclass(frozen=True)
class IndexingResult:
    """Outcome of indexing one local document."""

    source: Path
    document_hash: str
    stored_chunks: int
    status: Literal["indexed", "already_indexed"]
    collection: str = "general"


def index_document(
    path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    embedder: DocumentEmbedder,
    vector_store: IndexVectorStore,
    collection: str = "general",
) -> IndexingResult:
    """Ingest, deduplicate, embed, and store one document."""
    logical_collection = normalize_collection_name(collection)
    source = Path(path)
    chunks = ingest_document(source, chunk_size, chunk_overlap)
    document_hash = hash_file(source)
    if vector_store.has_document(document_hash):
        return IndexingResult(
            source, document_hash, 0, "already_indexed", logical_collection
        )

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_documents(texts)
    metadatas: list[dict[str, str | int | float | bool | None]] = []
    for chunk in chunks:
        metadatas.append(
            {
                **chunk.metadata,
                "document_hash": document_hash,
                "rag_collection": logical_collection,
            }
        )
    ids = [f"{document_hash}:{chunk.chunk_index}" for chunk in chunks]
    vector_store.add_chunks(ids, texts, embeddings, metadatas)
    return IndexingResult(
        source, document_hash, len(chunks), "indexed", logical_collection
    )


def index_directory(
    directory: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    embedder: DocumentEmbedder,
    vector_store: IndexVectorStore,
    collection: str = "general",
) -> list[IndexingResult]:
    """Index every supported document below a directory."""
    logical_collection = normalize_collection_name(collection)
    results: list[IndexingResult] = []
    for path in discover_documents(directory):
        try:
            results.append(
                index_document(
                    path,
                    chunk_size,
                    chunk_overlap,
                    embedder,
                    vector_store,
                    logical_collection,
                )
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise IndexingError(f"Failed to index {path}: {error}") from error
    return results
