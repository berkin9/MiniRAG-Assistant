"""Shared construction and orchestration for CLI and UI entry points."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.config import Settings
from app.services.collection_selection import (
    CollectionSelectionResult,
    CollectionSelector,
    ManualCollectionSelector,
    build_multi_collection_selector,
)
from app.services.answering import AnswerResult, answer_question
from app.services.collections import CollectionRegistry
from app.services.cross_collection import (
    CrossCollectionRetrievalResponse,
    CrossCollectionRetrievalService,
)
from app.services.embeddings import EmbeddingService
from app.services.llm_providers import LLMProvider, build_llm_provider
from app.services.multirag_answering import answer_cross_collection
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
    selection: CollectionSelectionResult | None = None
    retrieval: CrossCollectionRetrievalResponse | None = None


@dataclass(frozen=True)
class RoutedSearch:
    """Semantic search response paired with its routing decision."""

    response: RetrievalResponse | CrossCollectionRetrievalResponse
    routing: RoutingDecision
    selection: CollectionSelectionResult | None = None


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
    if settings.rag_retrieval_strategy == "cross_collection":
        return answer_with_routing(
            question,
            top_k,
            settings,
            query_mode="manual",
            collection=collection,
        ).answer
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
    collections: Sequence[str] | None = None,
    selector: CollectionSelector | None = None,
) -> RoutedSearch:
    """Run the configured single- or cross-collection search strategy."""
    if settings.rag_retrieval_strategy == "cross_collection":
        selection = select_collections_with_settings(
            query,
            settings,
            query_mode,
            collection,
            collections,
            selector,
            provider_factory,
        )
        response = _cross_collection_retrieve(query, settings, selection)
        return RoutedSearch(
            response=response,
            routing=_routing_from_selection(selection),
            selection=selection,
        )
    if collections is not None:
        raise ValueError(
            "Multiple collections require cross_collection retrieval strategy"
        )
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
    collections: Sequence[str] | None = None,
    selector: CollectionSelector | None = None,
) -> RoutedAnswer:
    """Run configured retrieval and generate one grounded answer."""
    factory = provider_factory or _cached_provider_factory(settings)
    if settings.rag_retrieval_strategy == "cross_collection":
        selection = select_collections_with_settings(
            question,
            settings,
            query_mode,
            collection,
            collections,
            selector,
            factory,
        )
        retrieval = _cross_collection_retrieve(question, settings, selection)
        answer = answer_cross_collection(
            retrieval,
            settings.max_context_characters,
            factory,
        )
        return RoutedAnswer(
            answer=answer,
            routing=_routing_from_selection(selection),
            selection=selection,
            retrieval=retrieval,
        )
    if collections is not None:
        raise ValueError(
            "Multiple collections require cross_collection retrieval strategy"
        )
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


def select_collections_with_settings(
    query: str,
    settings: Settings,
    query_mode: str | None = None,
    collection: str | None = None,
    collections: Sequence[str] | None = None,
    selector: CollectionSelector | None = None,
    provider_factory: Callable[[], LLMProvider] | None = None,
) -> CollectionSelectionResult:
    """Select a bounded registered collection list for cross-collection retrieval."""
    registry = build_collection_registry(settings)
    mode = query_mode or settings.default_query_mode
    if mode not in {"manual", "automatic"}:
        raise ValueError(f"Unsupported query mode: {mode}")
    if collections is not None and collection is not None:
        raise ValueError("Use either collection or collections, not both")
    if collections is not None or collection is not None or mode == "manual":
        selected = (
            tuple(collections)
            if collections is not None
            else (registry.resolve_logical_name(collection),)
        )
        return ManualCollectionSelector(
            registry,
            settings.multirag_max_collections,
            selected,
        ).select(query)
    factory = provider_factory or _cached_provider_factory(settings)
    automatic = selector or build_multi_collection_selector(
        settings.rag_routing_mode,
        registry,
        settings.multirag_max_collections,
        settings.multirag_min_selection_confidence,
        factory,
    )
    return automatic.select(query)


def _cross_collection_retrieve(
    query: str,
    settings: Settings,
    selection: CollectionSelectionResult,
) -> CrossCollectionRetrievalResponse:
    """Build one bounded cross-collection retrieval service per request."""
    registry = build_collection_registry(settings)
    embedder = EmbeddingService(settings.embedding_model)
    service = CrossCollectionRetrievalService(
        registry=registry,
        embedder=embedder,
        store_factory=lambda collection: ChromaVectorStore(
            settings.chroma_persist_dir,
            registry.physical_name(collection),
        ),
        top_k_per_collection=settings.multirag_top_k_per_collection,
        global_top_k=settings.multirag_global_top_k,
        max_distance=settings.max_retrieval_distance,
        deduplication_enabled=settings.multirag_deduplication_enabled,
    )
    return service.retrieve(query, selection)


def _routing_from_selection(
    selection: CollectionSelectionResult,
) -> RoutingDecision:
    """Preserve the existing routed wrapper while exposing full selection separately."""
    return RoutingDecision(
        collection=selection.collections[0],
        reason=selection.reason or "Selected registered collections.",
        confidence=selection.confidence,
        strategy=selection.strategy,
        fallback_used=selection.fallback_used,
    )


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
