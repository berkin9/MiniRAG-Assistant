"""Deterministic cross-collection retrieval, deduplication, and fusion."""

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from time import perf_counter

from app.services.collection_selection import CollectionSelectionResult
from app.services.collections import CollectionRegistry
from app.services.retrieval import (
    QueryEmbedder,
    RetrievalResult,
    SearchVectorStore,
    to_retrieval_result,
)
from app.services.vector_store import VectorSearchResult, VectorStoreError

logger = logging.getLogger(__name__)
RRF_K = 60


class CrossCollectionRetrievalError(RuntimeError):
    """Raised when bounded cross-collection retrieval cannot proceed safely."""


@dataclass(frozen=True)
class CrossCollectionRetrievalResponse:
    """Fused evidence and observable cross-collection retrieval metadata."""

    query: str
    results: tuple[RetrievalResult, ...]
    selection: CollectionSelectionResult
    selected_collections: tuple[str, ...]
    collections_searched: tuple[str, ...]
    results_per_collection: dict[str, int]
    total_candidates: int
    deduplicated_candidates: int
    returned_results: int
    collection_failures: dict[str, str]
    latency_ms: float

    @property
    def duplicate_removal_count(self) -> int:
        """Return candidates removed by exact deduplication."""
        return self.total_candidates - self.deduplicated_candidates


class CrossCollectionRetrievalService:
    """Search a validated bounded collection list and globally fuse evidence."""

    def __init__(
        self,
        registry: CollectionRegistry,
        embedder: QueryEmbedder,
        store_factory: Callable[[str], SearchVectorStore],
        top_k_per_collection: int,
        global_top_k: int,
        max_distance: float,
        deduplication_enabled: bool = True,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if top_k_per_collection < 1 or global_top_k < 1:
            raise ValueError("Cross-collection retrieval limits must be positive")
        if not isfinite(max_distance) or max_distance < 0:
            raise ValueError("max_distance must be finite and non-negative")
        self._registry = registry
        self._embedder = embedder
        self._store_factory = store_factory
        self._top_k_per_collection = top_k_per_collection
        self._global_top_k = global_top_k
        self._max_distance = max_distance
        self._deduplication_enabled = deduplication_enabled
        self._clock = clock

    def retrieve(
        self,
        query: str,
        selection: CollectionSelectionResult,
    ) -> CrossCollectionRetrievalResponse:
        """Retrieve once per collection, then normalize, deduplicate, and fuse."""
        if not query.strip():
            raise ValueError("Query must not be empty")
        collections = self._validate_collections(selection.collections)
        started = self._clock()
        logger.info(
            "multirag_retrieval_started collections=%s top_k_per_collection=%s",
            len(collections),
            self._top_k_per_collection,
        )
        query_embedding = self._embedder.embed_query(query)
        candidates: list[RetrievalResult] = []
        searched: list[str] = []
        counts = {collection: 0 for collection in collections}
        failures: dict[str, str] = {}

        for collection in collections:
            try:
                matches = self._store_factory(collection).search(
                    query_embedding, self._top_k_per_collection
                )
                ranked = _prepare_collection_results(
                    matches,
                    collection,
                    self._max_distance,
                )
            except VectorStoreError as error:
                failures[collection] = type(error).__name__
                logger.warning(
                    "multirag_collection_failed collection=%s error_type=%s",
                    collection,
                    type(error).__name__,
                )
                continue
            searched.append(collection)
            counts[collection] = len(ranked)
            candidates.extend(ranked)
            logger.info(
                "multirag_collection_completed collection=%s results=%s",
                collection,
                len(ranked),
            )

        if failures and not searched:
            raise CrossCollectionRetrievalError(
                "Retrieval failed for every selected collection"
            )
        fused, deduplicated_count = fuse_results(
            candidates,
            self._global_top_k,
            self._deduplication_enabled,
        )
        latency_ms = max((self._clock() - started) * 1_000, 0.0)
        logger.info(
            "multirag_fusion_completed candidates=%s deduplicated=%s "
            "returned=%s latency_ms=%.3f",
            len(candidates),
            deduplicated_count,
            len(fused),
            latency_ms,
        )
        return CrossCollectionRetrievalResponse(
            query=query.strip(),
            results=fused,
            selection=selection,
            selected_collections=collections,
            collections_searched=tuple(searched),
            results_per_collection=counts,
            total_candidates=len(candidates),
            deduplicated_candidates=deduplicated_count,
            returned_results=len(fused),
            collection_failures=failures,
            latency_ms=latency_ms,
        )

    def _validate_collections(
        self, collections: Sequence[str]
    ) -> tuple[str, ...]:
        """Reject the complete invalid list before embedding or store access."""
        if not collections:
            raise ValueError("At least one collection must be selected")
        if len(collections) != len(set(collections)):
            raise ValueError("Selected collections must not contain duplicates")
        registered = set(self._registry.list_collections())
        unknown = tuple(name for name in collections if name not in registered)
        if unknown:
            raise ValueError(f"Unknown collection selected: {unknown[0]}")
        return tuple(collections)


def normalize_cosine_distances(distances: Sequence[float]) -> tuple[float, ...]:
    """Convert cosine distances to stable relevance scores where higher is better."""
    normalized: list[float] = []
    for distance in distances:
        if not isfinite(distance) or distance < 0:
            raise ValueError("Cosine distances must be finite and non-negative")
        normalized.append(1.0 / (1.0 + distance))
    return tuple(normalized)


def fuse_results(
    candidates: Sequence[RetrievalResult],
    global_top_k: int,
    deduplication_enabled: bool = True,
) -> tuple[tuple[RetrievalResult, ...], int]:
    """Apply exact deduplication and deterministic reciprocal-rank fusion."""
    if global_top_k < 1:
        raise ValueError("global_top_k must be greater than zero")
    prepared = tuple(
        replace(
            candidate,
            fusion_score=1.0 / (RRF_K + (candidate.rank_within_collection or 1)),
        )
        for candidate in candidates
    )
    merged = _deduplicate(prepared) if deduplication_enabled else prepared
    ordered = sorted(merged, key=_fusion_sort_key)
    ranked = tuple(
        replace(candidate, global_rank=rank)
        for rank, candidate in enumerate(ordered[:global_top_k], start=1)
    )
    return ranked, len(merged)


def _prepare_collection_results(
    matches: Sequence[VectorSearchResult],
    collection: str,
    max_distance: float,
) -> tuple[RetrievalResult, ...]:
    """Filter and annotate one collection's independently ranked candidates."""
    normalize_cosine_distances(tuple(match.distance for match in matches))
    ordered = sorted(matches, key=lambda match: (match.distance, match.chunk_id))
    relevant = tuple(match for match in ordered if match.distance <= max_distance)
    normalized = normalize_cosine_distances(
        tuple(match.distance for match in relevant)
    )
    return tuple(
        replace(
            to_retrieval_result(match, collection),
            collection=collection,
            raw_score=match.distance,
            normalized_score=score,
            rank_within_collection=rank,
            matched_collections=(collection,),
        )
        for rank, (match, score) in enumerate(
            zip(relevant, normalized, strict=True), start=1
        )
    )


def _deduplicate(
    candidates: Sequence[RetrievalResult],
) -> tuple[RetrievalResult, ...]:
    """Merge exact identifiers, locations, or normalized text deterministically."""
    strongest_first = sorted(candidates, key=_strongest_sort_key)
    merged: list[RetrievalResult] = []
    key_to_index: dict[tuple[str, ...], int] = {}
    for candidate in strongest_first:
        keys = _duplicate_keys(candidate)
        duplicate_index = next(
            (key_to_index[key] for key in keys if key in key_to_index), None
        )
        if duplicate_index is None:
            duplicate_index = len(merged)
            merged.append(candidate)
        else:
            existing = merged[duplicate_index]
            collections = tuple(
                sorted(set(existing.matched_collections + candidate.matched_collections))
            )
            merged[duplicate_index] = replace(
                existing,
                matched_collections=collections,
                fusion_score=(existing.fusion_score or 0.0)
                + (candidate.fusion_score or 0.0),
            )
        for key in keys:
            key_to_index[key] = duplicate_index
    return tuple(merged)


def _duplicate_keys(result: RetrievalResult) -> tuple[tuple[str, ...], ...]:
    """Build exact, stable duplicate keys without retaining document text."""
    location = (
        "location",
        Path(result.source_file).name.casefold(),
        str(result.page_number or 0),
        str(result.chunk_index),
    )
    normalized_text = " ".join(result.text.casefold().split()).encode("utf-8")
    text_key = ("text", hashlib.sha256(normalized_text).hexdigest())
    if result.chunk_id:
        return (("chunk_id", result.chunk_id), location, text_key)
    return (location, text_key)


def _strongest_sort_key(result: RetrievalResult) -> tuple[object, ...]:
    """Order duplicate candidates so the strongest representative is retained."""
    return (
        -(result.normalized_score or 0.0),
        result.source_file.casefold(),
        result.page_number or 0,
        result.chunk_index,
        result.chunk_id,
        result.collection,
    )


def _fusion_sort_key(result: RetrievalResult) -> tuple[object, ...]:
    """Rank globally without depending on selected-collection input order."""
    return (
        -(result.fusion_score or 0.0),
        -(result.normalized_score or 0.0),
        Path(result.source_file).name.casefold(),
        result.page_number or 0,
        result.chunk_index,
        result.chunk_id,
        result.collection,
    )
