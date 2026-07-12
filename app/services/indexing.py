"""Orchestration for embedding and indexing ingested documents."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from app.services.hashing import hash_file
from app.services.ingestion import discover_documents, ingest_document
from app.services.vector_store import ChunkMetadata


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


def index_document(
    path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    embedder: DocumentEmbedder,
    vector_store: IndexVectorStore,
) -> IndexingResult:
    """Ingest, deduplicate, embed, and store one document."""
    source = Path(path)
    chunks = ingest_document(source, chunk_size, chunk_overlap)
    document_hash = hash_file(source)
    if vector_store.has_document(document_hash):
        return IndexingResult(source, document_hash, 0, "already_indexed")

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_documents(texts)
    metadatas: list[dict[str, str | int | float | bool | None]] = []
    for chunk in chunks:
        metadatas.append({**chunk.metadata, "document_hash": document_hash})
    ids = [f"{document_hash}:{chunk.chunk_index}" for chunk in chunks]
    vector_store.add_chunks(ids, texts, embeddings, metadatas)
    return IndexingResult(source, document_hash, len(chunks), "indexed")


def index_directory(
    directory: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    embedder: DocumentEmbedder,
    vector_store: IndexVectorStore,
) -> list[IndexingResult]:
    """Index every supported document below a directory."""
    return [
        index_document(path, chunk_size, chunk_overlap, embedder, vector_store)
        for path in discover_documents(directory)
    ]
