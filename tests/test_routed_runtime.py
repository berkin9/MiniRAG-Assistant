"""Tests for manual and automatic runtime orchestration."""

from collections.abc import Sequence

import pytest

from app.config import Settings
from app.services.routing import RoutingDecision
from app.services.runtime import answer_with_routing, route_with_settings
from app.services.vector_store import VectorSearchResult


class FakeRouter:
    """Return one observable decision and count invocations."""

    def __init__(self, collection: str = "technical") -> None:
        self.collection = collection
        self.calls = 0

    def route(self, question: str) -> RoutingDecision:
        del question
        self.calls += 1
        return RoutingDecision(
            self.collection,
            "Fake automatic decision.",
            0.8,
            "fake",
        )


class FakeEmbedder:
    """Embed queries deterministically without loading a model."""

    def embed_query(self, query: str) -> list[float]:
        del query
        return [1.0, 0.0]


class EmptyStore:
    """Return no relevant chunks from the selected collection."""

    def search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchResult]:
        del query_embedding, top_k
        return []


def test_explicit_manual_collection_bypasses_router() -> None:
    """An explicit manual selection must never be overridden."""
    router = FakeRouter()

    decision = route_with_settings(
        "How is authentication implemented?",
        Settings(),
        query_mode="automatic",
        collection="project",
        router=router,
    )

    assert decision.collection == "project"
    assert decision.strategy == "manual"
    assert router.calls == 0


def test_automatic_mode_invokes_router() -> None:
    """Automatic mode should use the injected router exactly once."""
    router = FakeRouter("technical")

    decision = route_with_settings(
        "How is authentication implemented?",
        Settings(),
        query_mode="automatic",
        router=router,
    )

    assert decision.collection == "technical"
    assert router.calls == 1


def test_routed_no_context_preserves_metadata_without_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One routed collection should return grounded no-context metadata."""
    from app.services import runtime

    router = FakeRouter("technical")
    provider_calls = 0

    def provider_factory() -> object:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("Provider must not be built without context")

    monkeypatch.setattr(
        runtime,
        "build_index_services",
        lambda settings, collection: (FakeEmbedder(), EmptyStore()),
    )

    result = answer_with_routing(
        "Unrelated question",
        4,
        Settings(),
        query_mode="automatic",
        router=router,
        provider_factory=provider_factory,
    )

    assert result.routing.collection == "technical"
    assert result.answer.collection == "technical"
    assert result.answer.has_relevant_context is False
    assert provider_calls == 0
