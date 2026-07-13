"""Grounded answer orchestration over semantic retrieval."""

from collections.abc import Callable
from dataclasses import dataclass

from app.services.context_builder import AnswerSource, build_context
from app.services.llm_providers import LLMProvider, LLMRequestError
from app.services.retrieval import QueryEmbedder, RetrievalResponse, SearchVectorStore, retrieve

GROUNDED_SYSTEM_PROMPT = """You answer questions only from the supplied context.
Never invent facts or use outside knowledge. If the context is insufficient, say so clearly.
Keep the answer concise and understandable. Preserve the user's language where practical.
Do not claim to have read documents that are absent from the context.
Cite supporting statements with the provided labels, such as [Source 1]."""


class AnswerGenerationError(RuntimeError):
    """Raised when grounded answer generation fails."""


@dataclass(frozen=True)
class AnswerResult:
    """Provider-independent grounded answer and citations."""

    question: str
    answer: str
    sources: tuple[AnswerSource, ...]
    has_relevant_context: bool
    collection: str = "general"


def answer_question(
    question: str,
    top_k: int,
    max_distance: float,
    max_context_characters: int,
    embedder: QueryEmbedder,
    vector_store: SearchVectorStore,
    provider_factory: Callable[[], LLMProvider],
    collection: str = "general",
) -> AnswerResult:
    """Retrieve relevant chunks and generate a strictly grounded answer."""
    if not question.strip():
        raise ValueError("Question must not be empty")

    retrieval = retrieve(
        question, top_k, max_distance, embedder, vector_store, collection
    )
    if not retrieval.results:
        return _no_context_result(retrieval)

    context = build_context(retrieval.results, max_context_characters)
    user_prompt = (
        "Use only the context below to answer the question. Cite sources using "
        "their exact labels.\n\n"
        f"Context:\n{context.text}\n\nQuestion:\n{question.strip()}"
    )
    try:
        answer = provider_factory().generate(GROUNDED_SYSTEM_PROMPT, user_prompt)
    except LLMRequestError as error:
        raise AnswerGenerationError("The selected LLM provider request failed") from error
    return AnswerResult(
        question=question.strip(),
        answer=answer,
        sources=context.sources,
        has_relevant_context=True,
        collection=retrieval.collection,
    )


def _no_context_result(retrieval: RetrievalResponse) -> AnswerResult:
    """Return a deterministic response without contacting an LLM."""
    question = retrieval.query.strip()
    lowered = question.lower()
    if any(character in lowered for character in "çğıış"):
        message = "İndekslenen belgelerde ilgili bilgi bulamadım."
    elif any(character in lowered for character in "äöüß"):
        message = "Ich konnte in den indexierten Dokumenten keine relevanten Informationen finden."
    else:
        message = "I could not find relevant information in the indexed documents."
    return AnswerResult(
        question=question,
        answer=message,
        sources=(),
        has_relevant_context=False,
        collection=retrieval.collection,
    )
