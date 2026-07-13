"""Tests for grounded answer orchestration."""

from collections.abc import Sequence

import pytest

from app.services.answering import (
    GROUNDED_SYSTEM_PROMPT,
    AnswerGenerationError,
    answer_question,
)
from app.services.llm_providers import LLMRequestError
from app.services.vector_store import VectorSearchResult


class FakeEmbedder:
    """Return a deterministic query vector."""

    def embed_query(self, query: str) -> list[float]:
        return [float(len(query)), 1.0]


class FakeStore:
    """Return configurable vector matches."""

    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results

    def search(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchResult]:
        del query_embedding
        return self.results[:top_k]


class FakeProvider:
    """Record prompts and return controlled text."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        if self.fail:
            raise LLMRequestError("provider failed")
        return "The deadline is Friday [Source 1]."


def _match(distance: float = 0.2) -> VectorSearchResult:
    return VectorSearchResult(
        text="The project deadline is Friday.",
        metadata={
            "source_file": "plan.pdf",
            "file_type": "pdf",
            "page_number": 3,
            "chunk_index": 2,
            "document_hash": "abc123",
        },
        distance=distance,
    )


def test_answer_uses_retrieved_context_and_returns_sources() -> None:
    """Relevant chunks should be sent with grounding instructions."""
    provider = FakeProvider()

    result = answer_question(
        "When is the deadline?",
        4,
        1.2,
        4_000,
        FakeEmbedder(),
        FakeStore([_match()]),
        lambda: provider,
        "project",
    )

    assert result.answer == "The deadline is Friday [Source 1]."
    assert result.has_relevant_context is True
    assert result.sources[0].page_number == 3
    assert result.sources[0].document_hash == "abc123"
    assert result.collection == "project"
    assert result.sources[0].collection == "project"
    assert "only from the supplied context" in provider.system_prompt
    assert GROUNDED_SYSTEM_PROMPT == provider.system_prompt
    assert "The project deadline is Friday." in provider.user_prompt
    assert "[Source 1]" in provider.user_prompt


def test_no_context_does_not_build_or_call_provider() -> None:
    """No relevant match should produce a deterministic valid response."""
    provider = FakeProvider()
    factory_calls = 0

    def factory() -> FakeProvider:
        nonlocal factory_calls
        factory_calls += 1
        return provider

    result = answer_question(
        "Unknown topic?", 4, 0.1, 4_000, FakeEmbedder(), FakeStore([_match()]), factory
    )

    assert result.has_relevant_context is False
    assert result.sources == ()
    assert "could not find relevant information" in result.answer
    assert factory_calls == 0
    assert provider.calls == 0


def test_empty_question_is_rejected() -> None:
    """Empty questions should stop before retrieval or generation."""
    with pytest.raises(ValueError, match="must not be empty"):
        answer_question(" ", 4, 1.2, 4_000, FakeEmbedder(), FakeStore([]), FakeProvider)


def test_provider_failure_is_converted() -> None:
    """Provider failures should cross the answer boundary as domain errors."""
    provider = FakeProvider(fail=True)

    with pytest.raises(AnswerGenerationError, match="provider request failed"):
        answer_question(
            "Question?", 4, 1.2, 4_000, FakeEmbedder(), FakeStore([_match()]), lambda: provider
        )
