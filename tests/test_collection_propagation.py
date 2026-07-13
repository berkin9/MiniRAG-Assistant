"""Regression tests for logical-to-physical collection propagation."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from app import main as cli
from app.config import Settings
from app.services.collections import CollectionRegistry
from app.services.routing import RoutingDecision
from app.services.runtime import (
    answer_with_routing,
    build_collection_registry,
    route_with_settings,
    search_with_settings,
)
from app.services.vector_store import ChromaVectorStore


class FixedEmbedder:
    """Provide deterministic document and query vectors without a model."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class FixedRouter:
    """Return one configured collection while recording route calls."""

    def __init__(self, collection: str) -> None:
        self.collection = collection
        self.calls = 0

    def route(self, question: str) -> RoutingDecision:
        self.calls += 1
        return RoutingDecision(self.collection, "Fixed regression route.")


class FixedProvider:
    """Return a grounded answer without any network request."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "The technical context was retrieved [Source 1]."


@pytest.fixture
def collection_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Settings:
    """Create isolated physical collections with distinct documents."""
    from app.services import runtime

    settings = Settings(
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection_name="propagation_documents",
    )
    registry = build_collection_registry(settings)
    _store_document(settings, registry, "general", "general document")
    _store_document(settings, registry, "technical", "technical document")
    monkeypatch.setattr(runtime, "EmbeddingService", lambda model: FixedEmbedder())
    return settings


def _store_document(
    settings: Settings,
    registry: CollectionRegistry,
    logical_collection: str,
    text: str,
) -> None:
    """Store one identifiable chunk in its resolved physical collection."""
    store = ChromaVectorStore(
        settings.chroma_persist_dir,
        registry.physical_name(logical_collection),
    )
    store.add_chunks(
        ids=[f"{logical_collection}:0"],
        texts=[text],
        embeddings=[[1.0, 0.0]],
        metadatas=[
            {
                "source_file": f"{logical_collection}.txt",
                "file_type": "txt",
                "page_number": None,
                "chunk_index": 0,
                "document_hash": logical_collection,
                "rag_collection": logical_collection,
            }
        ],
    )


@pytest.mark.parametrize(
    ("logical_collection", "expected_text"),
    [
        ("general", "general document"),
        ("technical", "technical document"),
    ],
)
def test_manual_search_uses_requested_physical_collection(
    collection_settings: Settings,
    logical_collection: str,
    expected_text: str,
) -> None:
    """Manual general and technical searches must remain physically isolated."""
    result = search_with_settings(
        "document",
        4,
        collection_settings,
        query_mode="manual",
        collection=logical_collection,
    )

    assert result.routing.collection == logical_collection
    assert result.response.collection == logical_collection
    assert [match.text for match in result.response.results] == [expected_text]
    assert all(
        match.collection == logical_collection
        for match in result.response.results
    )


def test_routed_search_uses_technical_physical_collection(
    collection_settings: Settings,
) -> None:
    """Automatic routing should bind retrieval to technical, never general."""
    router = FixedRouter("technical")

    result = search_with_settings(
        "authentication",
        4,
        collection_settings,
        query_mode="automatic",
        router=router,
    )

    assert router.calls == 1
    assert result.routing.collection == "technical"
    assert [match.text for match in result.response.results] == [
        "technical document"
    ]


def test_routed_ask_uses_technical_physical_collection(
    collection_settings: Settings,
) -> None:
    """Grounded answering should retrieve technical context before generation."""
    router = FixedRouter("technical")

    result = answer_with_routing(
        "How is authentication implemented?",
        4,
        collection_settings,
        query_mode="automatic",
        router=router,
        provider_factory=FixedProvider,
    )

    assert router.calls == 1
    assert result.routing.collection == "technical"
    assert result.answer.collection == "technical"
    assert result.answer.has_relevant_context is True
    assert result.answer.sources[0].source_file == "technical.txt"


def test_manual_collection_override_never_invokes_or_uses_default_route(
    collection_settings: Settings,
) -> None:
    """An explicit technical override must bypass a router choosing general."""
    router = FixedRouter("general")

    decision = route_with_settings(
        "authentication",
        collection_settings,
        query_mode="automatic",
        collection="technical",
        router=router,
    )
    result = search_with_settings(
        "authentication",
        4,
        collection_settings,
        query_mode="automatic",
        collection="technical",
        router=router,
    )

    assert router.calls == 0
    assert decision.collection == "technical"
    assert result.response.collection == "technical"
    assert result.response.results[0].source_file == "technical.txt"


def test_general_and_technical_resolve_to_distinct_physical_collections() -> None:
    """The default name stays compatible while technical receives a suffix."""
    registry = CollectionRegistry(
        "minirag_documents", "general", ("general", "technical")
    )

    assert registry.physical_name("general") == "minirag_documents"
    assert registry.physical_name("technical") == "minirag_documents__technical"
    assert registry.physical_name("general") != registry.physical_name("technical")


def test_cli_index_collection_writes_only_to_technical_physical_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI indexing boundary must preserve its explicit collection."""
    from app.services import runtime

    settings = Settings(
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection_name="cli_propagation_documents",
    )
    registry = build_collection_registry(settings)
    source = tmp_path / "technical.txt"
    source.write_text("technical implementation", encoding="utf-8")
    monkeypatch.setattr(runtime, "EmbeddingService", lambda model: FixedEmbedder())

    cli._run_index(source, settings, "technical")

    general = ChromaVectorStore(
        settings.chroma_persist_dir, registry.physical_name("general")
    )
    technical = ChromaVectorStore(
        settings.chroma_persist_dir, registry.physical_name("technical")
    )
    assert general.count() == 0
    assert technical.count() == 1
