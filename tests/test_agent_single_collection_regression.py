"""Regression coverage for routed single-collection agent answers."""

from collections.abc import Sequence

import pytest

from app.agent.planned_agent import build_planned_agent_service
from app.config import Settings
from app.services.collection_selection import CollectionSelectionResult
from app.services.cross_collection import CrossCollectionRetrievalResponse
from app.services.retrieval import RetrievalResult, to_retrieval_result
from app.services.vector_store import VectorSearchResult


class FixedEmbedder:
    """Avoid model loading while exercising the real retrieval pipeline."""

    def embed_query(self, query: str) -> list[float]:
        del query
        return [1.0, 0.0]


class CollectionStore:
    """Return two valid chunks from the routed logical collection."""

    def __init__(self, collection: str) -> None:
        self.collection = collection

    def search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchResult]:
        del query_embedding, top_k
        return [
            _match(self.collection, 0, "First supporting fact."),
            _match(self.collection, 1, "Second supporting fact."),
        ]


class GroupedCitationProvider:
    """Model a valid summary that groups two known display citations."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        assert "<CITATION" in user_prompt
        return "Grounded answer [Source 1, Source 2]."


def _match(collection: str, index: int, text: str) -> VectorSearchResult:
    return VectorSearchResult(
        text=text,
        metadata={
            "source_file": f"{collection}.pdf",
            "file_type": "pdf",
            "page_number": index + 1,
            "chunk_index": index,
            "document_hash": f"{collection}-hash",
            "rag_collection": collection,
        },
        distance=0.1 + index / 100,
        chunk_id=f"{collection}:{index}",
    )


@pytest.mark.parametrize(
    ("query", "expected_collection"),
    (
        ("Summarize the information security policy.", "policies"),
        ("Explain the authentication implementation.", "technical"),
        ("Who is the project sponsor?", "project"),
    ),
)
def test_agent_ask_routes_and_answers_single_collection_queries(
    query: str,
    expected_collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-collection ask plans should survive grouped model citations."""
    from app.services import runtime

    monkeypatch.setattr(
        runtime,
        "build_index_services",
        lambda settings, collection: (
            FixedEmbedder(),
            CollectionStore(collection or "general"),
        ),
    )
    monkeypatch.setattr(
        runtime, "build_llm_provider", lambda settings: GroupedCitationProvider()
    )

    result = build_planned_agent_service(Settings()).run(query)

    routed = result.execution.result
    assert result.selected_plan == "ask"
    assert routed.routing.collection == expected_collection
    assert routed.answer.collection == expected_collection
    assert routed.answer.has_relevant_context is True
    assert routed.answer.answer == "Grounded answer [Source 1] [Source 2]."


def test_existing_comparison_agent_query_still_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cross-collection comparison path must retain grounded answering."""
    from app.services import runtime

    query = (
        "Compare the authentication implementation with the security "
        "policy requirements."
    )
    selection = CollectionSelectionResult(
        collections=("technical", "policies"),
        strategy="deterministic",
        reason="Matched both collections.",
    )
    retrieval = CrossCollectionRetrievalResponse(
        query=query,
        results=(
            _retrieval_result("technical", 0, "Authentication implementation."),
            _retrieval_result("policies", 1, "Security policy requirement."),
        ),
        selection=selection,
        selected_collections=selection.collections,
        collections_searched=selection.collections,
        results_per_collection={"technical": 1, "policies": 1},
        total_candidates=2,
        deduplicated_candidates=2,
        returned_results=2,
        collection_failures={},
        latency_ms=1.0,
    )
    monkeypatch.setattr(runtime, "_cross_collection_retrieve", lambda *args: retrieval)
    monkeypatch.setattr(
        runtime, "build_llm_provider", lambda settings: GroupedCitationProvider()
    )

    result = build_planned_agent_service(
        Settings(rag_retrieval_strategy="cross_collection")
    ).run(query)

    routed = result.execution.result
    assert result.selected_plan == "ask"
    assert routed.answer.selected_collections == ("technical", "policies")
    assert routed.answer.answer == "Grounded answer [Source 1] [Source 2]."


def _retrieval_result(
    collection: str, index: int, text: str
) -> RetrievalResult:
    return to_retrieval_result(_match(collection, index, text), collection)
