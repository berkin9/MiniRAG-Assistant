"""Shared construction and orchestration for CLI and UI entry points."""

from collections.abc import Callable
from dataclasses import dataclass

from app.config import Settings
from app.services.answering import AnswerResult, answer_question
from app.services.collections import CollectionRegistry
from app.services.embeddings import EmbeddingService
from app.services.llm_providers import LLMProvider, build_llm_provider
from app.services.retrieval import RetrievalResponse, retrieve
from app.services.routing import (
    QueryRouter,
    RoutingDecision,
    build_query_router,
)
from app.services.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RoutedAnswer:
    """Grounded answer paired with its observable routing decision."""

    answer: AnswerResult
    routing: RoutingDecision


@dataclass(frozen=True)
class RoutedSearch:
    """Semantic search response paired with its routing decision."""

    response: RetrievalResponse
    routing: RoutingDecision


def build_index_services(
    settings: Settings, collection: str | None = None
) -> tuple[EmbeddingService, ChromaVectorStore]:
    """Create services backed by the configured local index."""
    registry = build_collection_registry(settings)
    return EmbeddingService(settings.embedding_model), ChromaVectorStore(
        settings.chroma_persist_dir, registry.physical_name(collection)
    )


def build_collection_registry(settings: Settings) -> CollectionRegistry:
    """Build the logical collection registry from validated settings."""
    return CollectionRegistry(
        settings.chroma_collection_name,
        settings.default_rag_collection,
        settings.rag_collections,
    )


def ask_with_settings(
    question: str,
    top_k: int,
    settings: Settings,
    collection: str | None = None,
) -> AnswerResult:
    """Run grounded answering with lazily built configured providers."""
    registry = build_collection_registry(settings)
    logical_collection = registry.resolve_logical_name(collection)
    embedder, vector_store = build_index_services(settings, logical_collection)
    return answer_question(
        question=question,
        top_k=top_k,
        max_distance=settings.max_retrieval_distance,
        max_context_characters=settings.max_context_characters,
        embedder=embedder,
        vector_store=vector_store,
        provider_factory=lambda: build_llm_provider(settings),
        collection=logical_collection,
    )


def route_with_settings(
    question: str,
    settings: Settings,
    query_mode: str | None = None,
    collection: str | None = None,
    router: QueryRouter | None = None,
    provider_factory: Callable[[], LLMProvider] | None = None,
) -> RoutingDecision:
    """Resolve manual selection or route automatically to one collection."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    registry = build_collection_registry(settings)
    mode = query_mode or settings.default_query_mode
    if mode not in {"manual", "automatic"}:
        raise ValueError(f"Unsupported query mode: {mode}")
    if collection is not None or mode == "manual":
        selected = registry.resolve_logical_name(collection)
        return RoutingDecision(
            collection=selected,
            reason="Used the explicitly selected collection."
            if collection is not None
            else "Used the configured default collection in manual mode.",
            strategy="manual",
        )

    factory = provider_factory or _cached_provider_factory(settings)
    selected_router = router or build_query_router(
        settings.rag_routing_mode, registry, factory
    )
    return selected_router.route(question)


def search_with_settings(
    query: str,
    top_k: int,
    settings: Settings,
    query_mode: str | None = None,
    collection: str | None = None,
    router: QueryRouter | None = None,
    provider_factory: Callable[[], LLMProvider] | None = None,
) -> RoutedSearch:
    """Route once and search only the selected logical collection."""
    routing = route_with_settings(
        query, settings, query_mode, collection, router, provider_factory
    )
    embedder, vector_store = build_index_services(settings, routing.collection)
    response = retrieve(
        query,
        top_k,
        settings.max_retrieval_distance,
        embedder,
        vector_store,
        routing.collection,
    )
    return RoutedSearch(response=response, routing=routing)


def answer_with_routing(
    question: str,
    top_k: int,
    settings: Settings,
    query_mode: str | None = None,
    collection: str | None = None,
    router: QueryRouter | None = None,
    provider_factory: Callable[[], LLMProvider] | None = None,
) -> RoutedAnswer:
    """Route once and generate an answer from only that collection."""
    factory = provider_factory or _cached_provider_factory(settings)
    routing = route_with_settings(
        question, settings, query_mode, collection, router, factory
    )
    embedder, vector_store = build_index_services(settings, routing.collection)
    answer = answer_question(
        question=question,
        top_k=top_k,
        max_distance=settings.max_retrieval_distance,
        max_context_characters=settings.max_context_characters,
        embedder=embedder,
        vector_store=vector_store,
        provider_factory=factory,
        collection=routing.collection,
    )
    return RoutedAnswer(answer=answer, routing=routing)


def _cached_provider_factory(
    settings: Settings,
) -> Callable[[], LLMProvider]:
    """Reuse one lazily built provider within a routed operation."""
    provider: LLMProvider | None = None

    def factory() -> LLMProvider:
        nonlocal provider
        if provider is None:
            provider = build_llm_provider(settings)
        return provider

    return factory
