"""Grounded answer orchestration over semantic retrieval."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from app.services.citations import (
    CitationValidationResult,
    render_display_citations,
    validate_and_repair_citations,
)
from app.services.context_builder import AnswerSource, build_context
from app.services.llm_providers import LLMProvider, LLMRequestError
from app.services.retrieval import QueryEmbedder, RetrievalResponse, SearchVectorStore, retrieve

CITATION_SYSTEM_RULES = """Use only the supplied evidence.
Cite the exact citation ID attached to the evidence that explicitly supports each claim.
Never infer a citation from evidence order or cite loosely related evidence from the same
collection. Never change, renumber, shorten, or invent citation IDs. Keep IDs exactly as
provided. If a claim uses multiple evidence blocks, cite every relevant ID. Every externally
verifiable claim from the evidence should have at least one citation. If evidence is
insufficient, say so instead of attaching an unrelated citation.
The following IDs are illustrative format examples only; never use them unless supplied in
the current evidence.
Good attribution: Access tokens expire after 15 minutes [TECH-02-C5-A1B2C3].
Bad attribution: Access tokens expire after 15 minutes [TECH-01-C1-D4E5F6] when that block
does not contain the expiration fact."""

GROUNDED_SYSTEM_PROMPT = f"""You answer questions only from the supplied context.
Never invent facts or use outside knowledge. If the context is insufficient, say so clearly.
Keep the answer concise and understandable. Preserve the user's language where practical.
Do not claim to have read documents that are absent from the context.
{CITATION_SYSTEM_RULES}"""


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
    selected_collections: tuple[str, ...] = ()
    results_per_collection: dict[str, int] = field(default_factory=dict)
    citation_id_to_display_label: dict[str, str] = field(default_factory=dict)
    citation_validation: CitationValidationResult | None = None


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
    return answer_from_retrieval(
        retrieval,
        max_context_characters,
        provider_factory,
    )


def answer_from_retrieval(
    retrieval: RetrievalResponse,
    max_context_characters: int,
    provider_factory: Callable[[], LLMProvider],
    *,
    system_prompt: str = GROUNDED_SYSTEM_PROMPT,
    include_collections: bool = False,
    selected_collections: tuple[str, ...] = (),
    results_per_collection: dict[str, int] | None = None,
) -> AnswerResult:
    """Generate one grounded answer from an already bounded retrieval response."""
    if not retrieval.results:
        empty = _no_context_result(retrieval)
        return replace(
            empty,
            selected_collections=selected_collections,
            results_per_collection=results_per_collection or {},
        )

    context = build_context(
        retrieval.results,
        max_context_characters,
        include_collections=include_collections,
    )
    if not context.sources:
        empty = _no_context_result(retrieval)
        return replace(
            empty,
            selected_collections=selected_collections,
            results_per_collection=results_per_collection or {},
        )
    user_prompt = (
        "Use only the evidence below. Cite its exact stable citation IDs; do not "
        "cite display labels or infer IDs from block order.\n\n"
        f"Context:\n{context.text}\n\nQuestion:\n{retrieval.query.strip()}"
    )
    try:
        provider_answer = provider_factory().generate(system_prompt, user_prompt)
    except LLMRequestError as error:
        raise AnswerGenerationError("The selected LLM provider request failed") from error
    normalized_answer, validation = validate_and_repair_citations(
        provider_answer,
        context.citation_id_to_display_label,
    )
    if not validation.valid:
        raise AnswerGenerationError(
            "The generated answer contained invalid citation identifiers"
        )
    answer = render_display_citations(
        normalized_answer,
        context.citation_id_to_display_label,
    )
    return AnswerResult(
        question=retrieval.query.strip(),
        answer=answer,
        sources=context.sources,
        has_relevant_context=True,
        collection=retrieval.collection,
        selected_collections=selected_collections,
        results_per_collection=results_per_collection or {},
        citation_id_to_display_label=context.citation_id_to_display_label,
        citation_validation=validation,
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
