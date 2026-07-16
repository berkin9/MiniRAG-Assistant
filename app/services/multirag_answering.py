"""Grounded answering over already fused cross-collection evidence."""

import logging
from collections.abc import Callable

from app.services.answering import AnswerResult, answer_from_retrieval
from app.services.cross_collection import CrossCollectionRetrievalResponse
from app.services.llm_providers import LLMProvider
from app.services.retrieval import RetrievalResponse

logger = logging.getLogger(__name__)

MULTIRAG_GROUNDED_SYSTEM_PROMPT = """You answer questions only from the supplied evidence.
Evidence may come from multiple document collections. Use collection names only as source
metadata, never as evidence. Never invent facts or make unsupported comparisons. If the
evidence is incomplete or conflicting, say so explicitly. Cite supporting statements with
the exact supplied labels, such as [Source 1]. Do not reveal hidden reasoning."""


def answer_cross_collection(
    retrieval: CrossCollectionRetrievalResponse,
    max_context_characters: int,
    provider_factory: Callable[[], LLMProvider],
) -> AnswerResult:
    """Generate at most one answer from globally ranked fused evidence."""
    response = RetrievalResponse(
        query=retrieval.query,
        results=retrieval.results,
        collection=retrieval.selected_collections[0],
    )
    result = answer_from_retrieval(
        response,
        max_context_characters,
        provider_factory,
        system_prompt=MULTIRAG_GROUNDED_SYSTEM_PROMPT,
        include_collections=True,
        selected_collections=retrieval.selected_collections,
        results_per_collection=retrieval.results_per_collection,
    )
    logger.info(
        "multirag_answer_completed collections=%s sources=%s has_context=%s",
        len(retrieval.selected_collections),
        len(result.sources),
        result.has_relevant_context,
    )
    return result
