"""Persistent ChromaDB storage isolated behind domain-friendly methods."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ChromaValue = str | int | float | bool
ChunkMetadata = Mapping[str, str | int | float | bool | None]


class VectorStoreError(RuntimeError):
    """Raised when vector persistence or querying fails."""


@dataclass(frozen=True)
class VectorSearchResult:
    """A vector match independent of ChromaDB's response format."""

    text: str
    metadata: dict[str, ChromaValue]
    distance: float
    chunk_id: str = ""


def normalize_metadata(metadata: ChunkMetadata) -> dict[str, ChromaValue]:
    """Convert optional values into Chroma-compatible scalar metadata."""
    normalized: dict[str, ChromaValue] = {}
    for key, value in metadata.items():
        if key == "page_number" and value is None:
            normalized[key] = 0
        else:
            normalized[key] = "" if value is None else value
    return normalized


class ChromaVectorStore:
    """Store and search chunk embeddings in a persistent Chroma collection."""

    def __init__(self, persist_directory: str | Path, collection_name: str) -> None:
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(persist_directory))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise VectorStoreError("Could not initialize ChromaDB") from error

    def has_document(self, document_hash: str) -> bool:
        """Return whether any stored chunk has the given content hash."""
        try:
            response = self._collection.get(
                where={"document_hash": document_hash}, limit=1, include=[]
            )
            return bool(response["ids"])
        except Exception as error:
            raise VectorStoreError("Could not check the ChromaDB index") from error

    def add_chunks(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[ChunkMetadata],
    ) -> None:
        """Persist chunk texts, embeddings, and normalized metadata."""
        if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("Chunk IDs, texts, embeddings, and metadata must align")
        if not ids:
            return
        try:
            self._collection.add(
                ids=list(ids),
                documents=list(texts),
                embeddings=[list(vector) for vector in embeddings],
                metadatas=[normalize_metadata(metadata) for metadata in metadatas],
            )
        except Exception as error:
            raise VectorStoreError("Could not store chunks in ChromaDB") from error

    def search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchResult]:
        """Return nearest chunks ordered by ascending cosine distance."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        count = self.count()
        if count == 0:
            return []
        try:
            response: dict[str, Any] = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            documents = (response.get("documents") or [[]])[0]
            metadatas = (response.get("metadatas") or [[]])[0]
            distances = (response.get("distances") or [[]])[0]
            ids = (response.get("ids") or [[]])[0]
            results = [
                VectorSearchResult(
                    text=str(text),
                    metadata=dict(metadata or {}),
                    distance=float(distance),
                    chunk_id=str(chunk_id),
                )
                for text, metadata, distance, chunk_id in zip(
                    documents, metadatas, distances, ids, strict=True
                )
            ]
            return sorted(results, key=lambda result: result.distance)
        except Exception as error:
            raise VectorStoreError("Could not search the ChromaDB index") from error

    def count(self) -> int:
        """Return the total number of stored chunks."""
        try:
            return int(self._collection.count())
        except Exception as error:
            raise VectorStoreError("Could not count the ChromaDB index") from error
