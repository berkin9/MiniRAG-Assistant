"""Tests for grounded context and citation construction."""

from app.services.context_builder import build_context
from app.services.retrieval import RetrievalResult


def _result(
    text: str,
    source: str,
    page: int | None,
    chunk: int,
    distance: float,
) -> RetrievalResult:
    return RetrievalResult(text, source, "pdf" if page else "txt", page, chunk, "hash", distance)


def test_context_preserves_rank_labels_pages_and_text() -> None:
    """Ranked results should become stable labeled source blocks."""
    context = build_context(
        (
            _result("First ranked text", "/docs/plan.pdf", 4, 7, 0.2),
            _result("Second ranked text", "/docs/notes.txt", 0, 2, 0.3),
        ),
        max_characters=2_000,
    )

    assert [source.label for source in context.sources] == ["Source 1", "Source 2"]
    assert context.text.index("First ranked text") < context.text.index("Second ranked text")
    assert "[Source 1]" in context.text
    assert "File: plan.pdf" in context.text
    assert "Page: 4" in context.text
    assert "Chunk: 7" in context.text
    assert "Page: 0" not in context.text


def test_context_deduplicates_locations_and_identical_text() -> None:
    """Duplicate locations or content should not consume prompt context."""
    context = build_context(
        (
            _result("same", "a.pdf", 1, 0, 0.1),
            _result("changed duplicate location", "a.pdf", 1, 0, 0.2),
            _result("same", "b.pdf", 2, 0, 0.3),
        ),
        max_characters=1_000,
    )

    assert len(context.sources) == 1
    assert context.text.count("Content:") == 1
