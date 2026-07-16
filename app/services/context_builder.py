"""Grounded prompt context construction from retrieval results."""

from dataclasses import dataclass
from pathlib import Path

from app.services.citations import build_citation_id
from app.services.retrieval import RetrievalResult


@dataclass(frozen=True)
class AnswerSource:
    """A source citation exposed by grounded answer generation."""

    label: str
    source_file: str
    file_type: str
    page_number: int | None
    chunk_index: int
    document_hash: str
    distance: float
    text_preview: str
    collection: str = "general"
    matched_collections: tuple[str, ...] = ()
    normalized_score: float | None = None
    global_rank: int | None = None
    citation_id: str = ""


@dataclass(frozen=True)
class GroundedContext:
    """Prompt context and its ordered public citations."""

    text: str
    sources: tuple[AnswerSource, ...]
    citation_id_to_display_label: dict[str, str]


def build_context(
    results: tuple[RetrievalResult, ...],
    max_characters: int,
    include_collections: bool = False,
) -> GroundedContext:
    """Build ranked, deduplicated source blocks within a character budget."""
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero")

    unique_results: list[tuple[RetrievalResult, str]] = []
    seen_locations: set[tuple[str, int | None, int]] = set()
    seen_text: set[str] = set()
    used_characters = 0
    for result in results:
        location = (result.source_file, result.page_number, result.chunk_index)
        normalized_text = result.text.strip()
        if location in seen_locations or normalized_text in seen_text:
            continue
        seen_locations.add(location)
        seen_text.add(normalized_text)
        unique_results.append((result, normalized_text))

    blocks: list[str] = []
    sources: list[AnswerSource] = []
    mapping: dict[str, str] = {}
    for result, normalized_text in unique_results:
        display_number = len(sources) + 1
        label = f"Source {display_number}"
        citation_id = build_citation_id(result, display_number)
        source = AnswerSource(
            label=label,
            source_file=result.source_file,
            file_type=result.file_type,
            page_number=result.page_number or None,
            chunk_index=result.chunk_index,
            document_hash=result.document_hash,
            distance=result.distance,
            text_preview=" ".join(normalized_text.split())[:240],
            collection=result.collection,
            matched_collections=result.matched_collections
            or (result.collection,),
            normalized_score=result.normalized_score,
            global_rank=result.global_rank,
            citation_id=citation_id,
        )
        block = _format_citation_block(
            source, normalized_text, include_collections
        )
        if used_characters + len(block) > max_characters:
            break
        blocks.append(block)
        sources.append(source)
        mapping[citation_id] = label
        used_characters += len(block)
    return GroundedContext(
        text="\n\n".join(blocks),
        sources=tuple(sources),
        citation_id_to_display_label=mapping,
    )


def _format_citation_block(
    source: AnswerSource,
    content: str,
    include_collections: bool = False,
) -> str:
    """Format one complete citation block with an immutable identity boundary."""
    lines = [
        f'<CITATION id="{source.citation_id}">',
        f"Document: {Path(source.source_file).name}",
    ]
    if include_collections:
        collections = source.matched_collections or (source.collection,)
        lines.append(f"Collections: {', '.join(collections)}")
    if source.page_number:
        lines.append(f"Page: {source.page_number}")
    lines.extend(
        [
            f"Chunk: {source.chunk_index}",
            "Content:",
            content,
            "</CITATION>",
        ]
    )
    return "\n".join(lines)
