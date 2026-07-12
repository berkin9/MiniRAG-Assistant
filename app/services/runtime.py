"""Shared construction and orchestration for CLI and UI entry points."""

from app.config import Settings
from app.services.answering import AnswerResult, answer_question
from app.services.embeddings import EmbeddingService
from app.services.llm_providers import build_llm_provider
from app.services.vector_store import ChromaVectorStore


def build_index_services(
    settings: Settings,
) -> tuple[EmbeddingService, ChromaVectorStore]:
    """Create services backed by the configured local index."""
    return EmbeddingService(settings.embedding_model), ChromaVectorStore(
        settings.chroma_persist_dir, settings.chroma_collection_name
    )


def ask_with_settings(question: str, top_k: int, settings: Settings) -> AnswerResult:
    """Run grounded answering with lazily built configured providers."""
    embedder, vector_store = build_index_services(settings)
    return answer_question(
        question=question,
        top_k=top_k,
        max_distance=settings.max_retrieval_distance,
        max_context_characters=settings.max_context_characters,
        embedder=embedder,
        vector_store=vector_store,
        provider_factory=lambda: build_llm_provider(settings),
    )
