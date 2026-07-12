"""Grounded prompt context construction from retrieval results."""

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class GroundedContext:
    """Prompt context and its ordered public citations."""

    text: str
    sources: tuple[AnswerSource, ...]


def build_context(
    results: tuple[RetrievalResult, ...], max_characters: int
) -> GroundedContext:
    """Build ranked, deduplicated source blocks within a character budget."""
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero")

    blocks: list[str] = []
    sources: list[AnswerSource] = []
    seen_locations: set[tuple[str, int | None, int]] = set()
    seen_text: set[str] = set()
    used_characters = 0
    for result in results:
        location = (result.source_file, result.page_number, result.chunk_index)
        normalized_text = result.text.strip()
        if location in seen_locations or normalized_text in seen_text:
            continue
        label = f"Source {len(sources) + 1}"
        block = _format_source_block(label, result, normalized_text)
        if blocks and used_characters + len(block) > max_characters:
            break
        if not blocks and len(block) > max_characters:
            block = block[:max_characters].rstrip()
        blocks.append(block)
        used_characters += len(block)
        seen_locations.add(location)
        seen_text.add(normalized_text)
        sources.append(
            AnswerSource(
                label=label,
                source_file=result.source_file,
                file_type=result.file_type,
                page_number=result.page_number or None,
                chunk_index=result.chunk_index,
                document_hash=result.document_hash,
                distance=result.distance,
                text_preview=" ".join(normalized_text.split())[:240],
            )
        )
    return GroundedContext(text="\n\n".join(blocks), sources=tuple(sources))


def _format_source_block(
    label: str, result: RetrievalResult, content: str
) -> str:
    """Format one stable prompt source block."""
    lines = [f"[{label}]", f"File: {Path(result.source_file).name}"]
    if result.page_number:
        lines.append(f"Page: {result.page_number}")
    lines.extend([f"Chunk: {result.chunk_index}", "Content:", content])
    return "\n".join(lines)
