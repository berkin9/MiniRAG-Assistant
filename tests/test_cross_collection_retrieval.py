"""Tests for cross-collection retrieval, normalization, fusion, and failures."""

from collections.abc import Sequence

import pytest

from app.services.collection_selection import (
    CollectionSelectionResult,
    DeterministicMultiCollectionSelector,
)
from app.services.collections import CollectionRegistry
from app.services.cross_collection import (
    CrossCollectionRetrievalError,
    CrossCollectionRetrievalService,
    fuse_results,
    normalize_cosine_distances,
)
from app.services.retrieval import RetrievalResult
from app.services.vector_store import VectorSearchResult, VectorStoreError


class FakeEmbedder:
    """Return one vector and record query embedding calls."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, query: str) -> list[float]:
        del query
        self.calls += 1
        return [1.0, 0.0]


class FakeStore:
    """Return controlled matches or one expected storage failure."""

    def __init__(
        self, matches: list[VectorSearchResult] | None = None, fail: bool = False
    ) -> None:
        self.matches = matches or []
        self.fail = fail
        self.calls = 0
        self.top_k = 0

    def search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchResult]:
        del query_embedding
        self.calls += 1
        self.top_k = top_k
        if self.fail:
            raise VectorStoreError("safe fake failure")
        return self.matches[:top_k]


def _registry() -> CollectionRegistry:
    return CollectionRegistry(
        "documents", "general", ("general", "technical", "policies")
    )


def _match(
    source: str,
    text: str,
    distance: float,
    chunk_id: str,
    chunk: int = 0,
) -> VectorSearchResult:
    return VectorSearchResult(
        text=text,
        metadata={
            "source_file": source,
            "file_type": "txt",
            "page_number": 0,
            "chunk_index": chunk,
            "document_hash": source,
        },
        distance=distance,
        chunk_id=chunk_id,
    )


def _selection(*collections: str) -> CollectionSelectionResult:
    return CollectionSelectionResult(
        collections=collections,
        strategy="manual",
        reason="Test selection.",
    )


def _service(
    stores: dict[str, FakeStore], embedder: FakeEmbedder | None = None, **kwargs: object
) -> CrossCollectionRetrievalService:
    return CrossCollectionRetrievalService(
        _registry(),
        embedder or FakeEmbedder(),
        lambda collection: stores[collection],
        top_k_per_collection=int(kwargs.get("top_k", 2)),
        global_top_k=int(kwargs.get("global_top_k", 4)),
        max_distance=1.2,
        deduplication_enabled=bool(kwargs.get("deduplication", True)),
    )


def test_every_collection_is_searched_once_with_per_collection_limit() -> None:
    stores = {
        "technical": FakeStore([_match("a.txt", "A", 0.1, "a")]),
        "policies": FakeStore([_match("b.txt", "B", 0.2, "b")]),
    }
    embedder = FakeEmbedder()

    response = _service(stores, embedder).retrieve(
        "compare", _selection("technical", "policies")
    )

    assert embedder.calls == 1
    assert [stores[name].calls for name in stores] == [1, 1]
    assert [stores[name].top_k for name in stores] == [2, 2]
    assert response.results_per_collection == {"technical": 1, "policies": 1}
    assert {result.collection for result in response.results} == {
        "technical",
        "policies",
    }
    assert [result.global_rank for result in response.results] == [1, 2]


def test_comparison_query_selection_searches_technical_and_policies() -> None:
    registry = _registry()
    selection = DeterministicMultiCollectionSelector(registry, 3).select(
        "Compare authentication implementation with security policy requirements."
    )
    stores = {
        "technical": FakeStore([_match("auth.txt", "Auth", 0.1, "auth")]),
        "policies": FakeStore([_match("policy.txt", "Policy", 0.2, "policy")]),
    }

    response = _service(stores).retrieve("compare", selection)

    assert selection.collections == ("technical", "policies")
    assert response.selected_collections == ("technical", "policies")
    assert response.collections_searched == ("technical", "policies")
    assert {result.collection for result in response.results} == {
        "technical",
        "policies",
    }


def test_unknown_collection_is_rejected_before_embedding_or_store_access() -> None:
    embedder = FakeEmbedder()
    stores: dict[str, FakeStore] = {}
    service = _service(stores, embedder)

    with pytest.raises(ValueError, match="Unknown collection"):
        service.retrieve("query", _selection("secret"))

    assert embedder.calls == 0


def test_empty_and_partial_results_are_safe() -> None:
    stores = {
        "technical": FakeStore([]),
        "policies": FakeStore([_match("policy.txt", "Policy", 0.3, "p")]),
    }

    response = _service(stores).retrieve(
        "query", _selection("technical", "policies")
    )

    assert response.results_per_collection == {"technical": 0, "policies": 1}
    assert response.returned_results == 1

    empty = _service(
        {"technical": FakeStore([]), "policies": FakeStore([])}
    ).retrieve("query", _selection("technical", "policies"))
    assert empty.results == ()
    assert empty.returned_results == 0


def test_one_failure_continues_but_all_failures_raise_without_retry() -> None:
    stores = {
        "technical": FakeStore(fail=True),
        "policies": FakeStore([_match("policy.txt", "Policy", 0.2, "p")]),
    }
    response = _service(stores).retrieve(
        "query", _selection("technical", "policies")
    )
    assert response.collection_failures == {"technical": "VectorStoreError"}
    assert response.returned_results == 1
    assert stores["technical"].calls == 1

    failed = {
        "technical": FakeStore(fail=True),
        "policies": FakeStore(fail=True),
    }
    with pytest.raises(CrossCollectionRetrievalError, match="every"):
        _service(failed).retrieve(
            "query", _selection("technical", "policies")
        )
    assert [store.calls for store in failed.values()] == [1, 1]


def test_cosine_distance_normalization_handles_edge_cases() -> None:
    assert normalize_cosine_distances(()) == ()
    assert normalize_cosine_distances((0.0,)) == (1.0,)
    equal = normalize_cosine_distances((0.5, 0.5))
    assert equal[0] == equal[1]
    scores = normalize_cosine_distances((0.1, 0.8))
    assert scores[0] > scores[1]
    for invalid in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError, match="finite and non-negative"):
            normalize_cosine_distances((invalid,))


def _result(
    source: str,
    collection: str,
    score: float,
    rank: int,
    *,
    text: str = "text",
    chunk_id: str = "",
) -> RetrievalResult:
    return RetrievalResult(
        text=text,
        source_file=source,
        file_type="txt",
        page_number=None,
        chunk_index=0,
        document_hash=source,
        distance=(1 / score) - 1,
        collection=collection,
        chunk_id=chunk_id,
        normalized_score=score,
        rank_within_collection=rank,
        matched_collections=(collection,),
        raw_score=(1 / score) - 1,
    )


def test_fusion_is_global_stable_and_enforces_limit() -> None:
    candidates = (
        _result("z.txt", "technical", 0.9, 2, text="z"),
        _result("b.txt", "policies", 0.8, 1, text="b"),
        _result("a.txt", "technical", 0.8, 1, text="a"),
    )

    fused, count = fuse_results(candidates, 2, False)

    assert count == 3
    assert [item.source_file for item in fused] == ["a.txt", "b.txt"]
    assert fused[0].fusion_score == pytest.approx(1 / 61)
    assert fused[0].raw_score is not None


def test_deduplication_keeps_strongest_and_matched_collections() -> None:
    candidates = (
        _result(
            "same.txt", "technical", 0.9, 1, text="same", chunk_id="same:0"
        ),
        _result(
            "same.txt", "policies", 0.7, 2, text="same", chunk_id="same:0"
        ),
    )

    fused, count = fuse_results(candidates, 5, True)

    assert count == 1
    assert fused[0].collection == "technical"
    assert fused[0].normalized_score == 0.9
    assert fused[0].matched_collections == ("policies", "technical")
    assert fused[0].fusion_score == pytest.approx((1 / 61) + (1 / 62))


def test_deduplication_can_be_disabled() -> None:
    candidates = (
        _result("same.txt", "technical", 0.9, 1, chunk_id="same"),
        _result("same.txt", "policies", 0.8, 1, chunk_id="same"),
    )

    fused, count = fuse_results(candidates, 5, False)

    assert count == 2
    assert len(fused) == 2


def test_location_and_normalized_text_are_independent_duplicate_keys() -> None:
    same_location = (
        _result("same.txt", "technical", 0.9, 1, text="first"),
        _result("same.txt", "policies", 0.8, 1, text="changed"),
    )
    same_text = (
        _result("a.txt", "technical", 0.9, 1, text=" Same   Text "),
        _result("b.txt", "policies", 0.8, 1, text="same text"),
    )

    assert fuse_results(same_location, 5, True)[1] == 1
    assert fuse_results(same_text, 5, True)[1] == 1


def test_collection_input_order_does_not_change_global_ranking() -> None:
    first = (
        _result("b.txt", "technical", 0.8, 1, text="b"),
        _result("a.txt", "policies", 0.8, 1, text="a"),
    )

    forward = fuse_results(first, 5, False)[0]
    reversed_order = fuse_results(tuple(reversed(first)), 5, False)[0]

    assert [result.source_file for result in forward] == ["a.txt", "b.txt"]
    assert [result.source_file for result in reversed_order] == ["a.txt", "b.txt"]
