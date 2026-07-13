"""Shared construction and orchestration for CLI and UI entry points."""

from app.config import Settings
from app.services.answering import AnswerResult, answer_question
from app.services.collections import CollectionRegistry
from app.services.embeddings import EmbeddingService
from app.services.llm_providers import build_llm_provider
from app.services.vector_store import ChromaVectorStore


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
