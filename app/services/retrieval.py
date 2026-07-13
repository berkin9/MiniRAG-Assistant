"""Semantic retrieval over indexed document chunks."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from app.services.collections import normalize_collection_name
from app.services.vector_store import VectorSearchResult


class QueryEmbedder(Protocol):
    """Embedding behavior needed for retrieval."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""


class SearchVectorStore(Protocol):
    """Vector-store behavior needed for retrieval."""

    def search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchResult]:
        """Find nearest stored chunks."""


@dataclass(frozen=True)
class RetrievalResult:
    """A relevant document chunk returned for a query."""

    text: str
    source_file: str
    file_type: str
    page_number: int | None
    chunk_index: int
    document_hash: str
    distance: float
    collection: str = "general"


@dataclass(frozen=True)
class RetrievalResponse:
    """Structured search response that may contain no relevant matches."""

    query: str
    results: tuple[RetrievalResult, ...]
    collection: str = "general"


def retrieve(
    query: str,
    top_k: int,
    max_distance: float,
    embedder: QueryEmbedder,
    vector_store: SearchVectorStore,
    collection: str = "general",
) -> RetrievalResponse:
    """Return chunks whose cosine distance is within the configured maximum."""
    if not query.strip():
        raise ValueError("Query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not isfinite(max_distance) or max_distance < 0:
        raise ValueError("max_distance must be finite and non-negative")
    logical_collection = normalize_collection_name(collection)

    matches = vector_store.search(embedder.embed_query(query), top_k)
    results = tuple(
        _to_retrieval_result(match, logical_collection)
        for match in sorted(matches, key=lambda item: item.distance)
        if match.distance <= max_distance
    )
    return RetrievalResponse(
        query=query, results=results, collection=logical_collection
    )


def _to_retrieval_result(
    match: VectorSearchResult, collection: str
) -> RetrievalResult:
    """Map normalized vector metadata into the retrieval domain model."""
    page_number = int(match.metadata.get("page_number", 0)) or None
    return RetrievalResult(
        text=match.text,
        source_file=str(match.metadata["source_file"]),
        file_type=str(match.metadata["file_type"]),
        page_number=page_number,
        chunk_index=int(match.metadata["chunk_index"]),
        document_hash=str(match.metadata["document_hash"]),
        distance=match.distance,
        collection=str(match.metadata.get("rag_collection", collection)),
    )
