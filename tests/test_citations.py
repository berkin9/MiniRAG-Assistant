"""Tests for stable citation identity, validation, repair, and rendering."""

import re
from contextlib import contextmanager

import pytest

from app import main as cli
from app import ui
from app.services.answering import (
    GROUNDED_SYSTEM_PROMPT,
    AnswerGenerationError,
    answer_from_retrieval,
)
from app.services.citations import (
    build_citation_id,
    render_display_citations,
    validate_and_repair_citations,
)
from app.services.collection_selection import CollectionSelectionResult
from app.services.context_builder import build_context
from app.services.cross_collection import CrossCollectionRetrievalResponse
from app.services.multirag_answering import answer_cross_collection
from app.services.retrieval import RetrievalResponse, RetrievalResult


def _result(
    text: str,
    source: str,
    collection: str,
    chunk: int,
    rank: int,
) -> RetrievalResult:
    return RetrievalResult(
        text=text,
        source_file=source,
        file_type="pdf",
        page_number=2,
        chunk_index=chunk,
        document_hash=f"hash-{source}",
        distance=0.1 * rank,
        collection=collection,
        chunk_id=f"hash-{source}:{chunk}",
        normalized_score=1 / (1 + (0.1 * rank)),
        fusion_score=1 / (60 + rank),
        rank_within_collection=rank,
        global_rank=rank,
        matched_collections=(collection,),
        raw_score=0.1 * rank,
    )


def test_citation_ids_are_unique_deterministic_and_path_safe() -> None:
    first = _result(
        "First", "/private/users/alice/secret.pdf", "technical", 5, 1
    )
    second = _result("Second", "/other/secret.pdf", "technical", 5, 2)

    first_id = build_citation_id(first, 1)

    assert first_id == build_citation_id(first, 1)
    assert first_id != build_citation_id(second, 2)
    assert re.fullmatch(r"TECH-01-C5-[A-F0-9]{6}", first_id)
    assert "alice" not in first_id.lower()
    assert "/" not in first_id


def test_context_assigns_ids_after_final_order_with_one_to_one_mapping() -> None:
    evidence = (
        _result("First evidence", "first.pdf", "technical", 5, 1),
        _result("Second evidence", "second.pdf", "policies", 1, 2),
    )

    first = build_context(evidence, 4_000, include_collections=True)
    second = build_context(evidence, 4_000, include_collections=True)

    assert [source.citation_id for source in first.sources] == [
        source.citation_id for source in second.sources
    ]
    assert len(set(first.citation_id_to_display_label)) == 2
    assert first.citation_id_to_display_label == {
        first.sources[0].citation_id: "Source 1",
        first.sources[1].citation_id: "Source 2",
    }
    assert [source.global_rank for source in first.sources] == [1, 2]


def test_context_blocks_are_bounded_unambiguous_and_ordered() -> None:
    evidence = (
        _result("Access tokens remain valid for 15 minutes.", "auth.pdf", "technical", 5, 1),
        _result("Users authenticate with OpenID Connect.", "oidc.pdf", "technical", 1, 2),
    )

    context = build_context(evidence, 4_000, include_collections=True)

    assert context.text.count("<CITATION id=") == 2
    assert context.text.count("</CITATION>") == 2
    for source in context.sources:
        start = context.text.index(f'<CITATION id="{source.citation_id}">')
        content = context.text.index("Content:", start)
        end = context.text.index("</CITATION>", content)
        assert start < content < end
    assert context.text.index("15 minutes") < context.text.index("OpenID Connect")


def test_context_budget_removes_complete_blocks_only() -> None:
    evidence = (
        _result("A" * 200, "large.pdf", "technical", 1, 1),
        _result("small", "small.pdf", "technical", 2, 2),
    )

    context = build_context(evidence, 100)

    assert context.text == ""
    assert context.sources == ()
    assert context.citation_id_to_display_label == {}


def test_grounded_prompt_requires_exact_claim_level_attribution() -> None:
    prompt = GROUNDED_SYSTEM_PROMPT

    assert "exact citation ID" in prompt
    assert "Never infer a citation from evidence order" in prompt
    assert "Never change, renumber, shorten, or invent citation IDs" in prompt
    assert "loosely related evidence" in prompt
    assert "Good attribution:" in prompt
    assert "Bad attribution:" in prompt
    assert "chain-of-thought" not in prompt


def test_validator_accepts_valid_and_duplicate_ids() -> None:
    mapping = {"TECH-01-C5-A1B2C3": "Source 1"}

    answer, result = validate_and_repair_citations(
        "Fact [TECH-01-C5-A1B2C3]. Again [TECH-01-C5-A1B2C3].",
        mapping,
    )

    assert result.valid is True
    assert result.used_citation_ids == ("TECH-01-C5-A1B2C3",)
    assert result.repaired is False
    assert answer.count("[TECH-01-C5-A1B2C3]") == 2


def test_validator_detects_unknown_cross_request_and_malformed_ids() -> None:
    mapping = {"TECH-01-C5-A1B2C3": "Source 1"}

    _, unknown = validate_and_repair_citations(
        "Fact [POLI-02-C1-D4E5F6].", mapping
    )
    _, malformed = validate_and_repair_citations(
        "Fact [TECH-1-C5-A1B2C3].", mapping
    )
    _, unclosed = validate_and_repair_citations(
        "Fact [TECH-01-C5-A1B2C3 and [Source 1", mapping
    )

    assert unknown.valid is False
    assert unknown.unknown_citation_ids == ("POLI-02-C1-D4E5F6",)
    assert malformed.valid is False
    assert malformed.malformed_citations == ("[TECH-1-C5-A1B2C3]",)
    assert unclosed.valid is False
    assert "TECH-01-C5-A1B2C3" in unclosed.malformed_citations
    assert "[Source 1" in unclosed.malformed_citations


def test_validator_repairs_whitespace_and_unambiguous_display_label() -> None:
    mapping = {"TECH-01-C5-A1B2C3": "Source 1"}

    answer, result = validate_and_repair_citations(
        "One [ TECH-01-C5-A1B2C3 ]. Two [Source 1].",
        mapping,
    )

    assert result.valid is True
    assert result.repaired is True
    assert answer == (
        "One [TECH-01-C5-A1B2C3]. Two [TECH-01-C5-A1B2C3]."
    )


def test_validator_does_not_guess_unknown_source_number() -> None:
    mapping = {"TECH-01-C5-A1B2C3": "Source 1"}

    answer, result = validate_and_repair_citations("Fact [Source 9].", mapping)

    assert result.valid is False
    assert result.unknown_citation_ids == ("Source 9",)
    assert answer == "Fact [Source 9]."


def test_answers_without_citations_preserve_existing_policy() -> None:
    answer, result = validate_and_repair_citations(
        "The evidence is insufficient.",
        {"TECH-01-C5-A1B2C3": "Source 1"},
    )

    assert result.valid is True
    assert result.used_citation_ids == ()
    assert answer == "The evidence is insufficient."


def test_display_rendering_replaces_only_exact_validated_tokens() -> None:
    mapping = {"TECH-01-C5-A1B2C3": "Source 1"}

    rendered = render_display_citations(
        "[TECH-01-C5-A1B2C3] TECH-01-C5-A1B2C3 [TECH-01-C5-A1B2C3-extra]",
        mapping,
    )

    assert rendered == (
        "[Source 1] TECH-01-C5-A1B2C3 [TECH-01-C5-A1B2C3-extra]"
    )


@pytest.mark.parametrize(
    "grouped",
    (
        "[Source 1, Source 2]",
        "[Source 1; Source 2]",
        "[Source 1 and Source 2]",
    ),
)
def test_grouped_known_display_citations_are_safely_expanded(grouped: str) -> None:
    """Provider grouping should normalize only exact known source labels."""
    mapping = {
        "TECH-01-C1-A1B2C3": "Source 1",
        "POLI-02-C2-D4E5F6": "Source 2",
    }

    normalized, validation = validate_and_repair_citations(
        f"Grounded statement {grouped}.", mapping
    )

    assert normalized == (
        "Grounded statement [TECH-01-C1-A1B2C3] [POLI-02-C2-D4E5F6]."
    )
    assert validation.valid is True
    assert validation.repaired is True
    assert validation.used_citation_ids == tuple(mapping)


def test_grouped_citation_with_unknown_member_remains_invalid() -> None:
    """Grouping must not make an invented or unavailable source acceptable."""
    mapping = {"TECH-01-C1-A1B2C3": "Source 1"}

    _, validation = validate_and_repair_citations(
        "Claim [Source 1, Source 99].", mapping
    )

    assert validation.valid is False
    assert validation.malformed_citations == ("[Source 1, Source 99]",)


class ControlledProvider:
    """Return controlled answer text while recording exact call count."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        self.calls += 1
        return self.answer


def test_unrepairable_answer_fails_after_one_provider_call() -> None:
    provider = ControlledProvider("Claim [Source 99].")
    retrieval = RetrievalResponse(
        "question",
        (_result("Evidence", "source.pdf", "technical", 1, 1),),
        "technical",
    )

    with pytest.raises(AnswerGenerationError, match="invalid citation"):
        answer_from_retrieval(retrieval, 4_000, lambda: provider)

    assert provider.calls == 1


class RegressionProvider:
    """Cite the exact evidence IDs associated with the observed live facts."""

    def __init__(self) -> None:
        self.calls = 0
        self.mapping: dict[str, str] = {}

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        self.calls += 1
        blocks = re.findall(
            r'<CITATION id="([^"]+)">.*?Content:\n(.*?)\n</CITATION>',
            user_prompt,
            flags=re.DOTALL,
        )
        self.mapping = {content: citation_id for citation_id, content in blocks}
        auth_id = next(
            citation_id
            for content, citation_id in self.mapping.items()
            if "15 minutes" in content
        )
        oidc_id = next(
            citation_id
            for content, citation_id in self.mapping.items()
            if "OpenID Connect" in content
        )
        return (
            f"Access tokens expire after 15 minutes [{auth_id}]. "
            f"Refresh tokens use secure HTTP-only cookies [{auth_id}]. "
            f"Authentication uses OpenID Connect [{oidc_id}]. "
            f"Role-based access uses extracted roles [{oidc_id}]."
        )


def _regression_response() -> CrossCollectionRetrievalResponse:
    results = (
        _result("Unrelated overview.", "overview.pdf", "general", 0, 1),
        _result(
            "Access tokens remain valid for 15 minutes.\n"
            "Refresh tokens are stored in secure HTTP-only cookies.",
            "authentication.pdf",
            "technical",
            5,
            2,
        ),
        _result("Security policy overview.", "policy.pdf", "policies", 2, 3),
        _result(
            "Users authenticate through OpenID Connect.\n"
            "The backend validates signed access tokens and extracts roles.",
            "openid.pdf",
            "technical",
            1,
            4,
        ),
    )
    selection = CollectionSelectionResult(
        collections=("technical", "policies"), strategy="deterministic"
    )
    return CrossCollectionRetrievalResponse(
        query="Compare authentication with security policy requirements.",
        results=results,
        selection=selection,
        selected_collections=selection.collections,
        collections_searched=selection.collections,
        results_per_collection={"technical": 3, "policies": 1},
        total_candidates=4,
        deduplicated_candidates=4,
        returned_results=4,
        collection_failures={},
        latency_ms=1.0,
    )


def test_live_bug_claims_remain_bound_to_underlying_stable_ids() -> None:
    provider = RegressionProvider()

    result = answer_cross_collection(
        _regression_response(), 10_000, lambda: provider
    )

    assert provider.calls == 1
    assert "15 minutes [Source 2]" in result.answer
    assert "HTTP-only cookies [Source 2]" in result.answer
    assert "OpenID Connect [Source 4]" in result.answer
    assert "extracted roles [Source 4]" in result.answer
    assert "15 minutes" in result.sources[1].text_preview
    assert "OpenID Connect" in result.sources[3].text_preview
    assert result.citation_id_to_display_label == {
        source.citation_id: source.label for source in result.sources
    }


def test_cli_and_streamlit_render_the_same_stable_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = answer_cross_collection(
        _regression_response(), 10_000, RegressionProvider
    )
    markdown: list[str] = []

    @contextmanager
    def expander(label: str, expanded: bool = False):
        del label, expanded
        yield

    monkeypatch.setattr(ui.st, "subheader", lambda value: None)
    monkeypatch.setattr(ui.st, "write", lambda value: None)
    monkeypatch.setattr(ui.st, "expander", expander)
    monkeypatch.setattr(ui.st, "markdown", markdown.append)
    monkeypatch.setattr(ui.st, "caption", lambda value: None)

    cli._print_answer(result)
    ui._render_answer(result)

    cli_output = capsys.readouterr().out
    for source in result.sources:
        assert f"[{source.label}] [{source.citation_id}]" in cli_output
        assert any(
            f"[{source.label}] [{source.citation_id}]" in value
            for value in markdown
        )
