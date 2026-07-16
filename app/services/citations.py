"""Stable citation identity, validation, local repair, and display mapping."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from app.services.retrieval import RetrievalResult

_STABLE_ID = re.compile(r"^[A-Z0-9]{2,8}-\d{2,4}-C\d+-[A-F0-9]{6}$")
_DISPLAY_LABEL = re.compile(r"^Source\s+(\d+)$", re.IGNORECASE)
_BRACKET_TOKEN = re.compile(r"\[([^\[\]\n]{1,80})\]")
_CITATION_LIKE = re.compile(
    r"^(?:source\b|[A-Za-z0-9]{2,8}[- ]\d|[A-Za-z0-9-]*-C\d+)",
    re.IGNORECASE,
)
_BARE_STABLE_ID = re.compile(
    r"\b[A-Z0-9]{2,8}-\d{1,4}-C\d+-[A-F0-9]{3,12}\b"
)
_UNCLOSED_SOURCE = re.compile(r"\[Source\s+\d+(?![\d\]])", re.IGNORECASE)


@dataclass(frozen=True)
class CitationValidationResult:
    """Structured identity validation without semantic entailment claims."""

    valid: bool
    used_citation_ids: tuple[str, ...]
    unknown_citation_ids: tuple[str, ...]
    malformed_citations: tuple[str, ...]
    repaired: bool = False


def build_citation_id(result: RetrievalResult, display_number: int) -> str:
    """Create one short deterministic ID from final evidence identity."""
    if display_number < 1:
        raise ValueError("display_number must be at least 1")
    prefix = _collection_prefix(result.collection)
    identity = "|".join(
        (
            result.chunk_id,
            result.document_hash,
            Path(result.source_file).name.casefold(),
            str(result.page_number or 0),
            str(result.chunk_index),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:6].upper()
    return f"{prefix}-{display_number:02d}-C{result.chunk_index}-{digest}"


def validate_and_repair_citations(
    answer: str,
    citation_id_to_display_label: dict[str, str],
) -> tuple[str, CitationValidationResult]:
    """Normalize unambiguous tokens and reject unknown or malformed citations."""
    _validate_mapping(citation_id_to_display_label)
    display_to_id = {
        label: citation_id
        for citation_id, label in citation_id_to_display_label.items()
    }
    used: list[str] = []
    unknown: list[str] = []
    malformed: list[str] = []
    repaired = False

    def normalize(match: re.Match[str]) -> str:
        nonlocal repaired
        original = match.group(0)
        token = match.group(1).strip()
        citation_id: str | None = None
        if token in citation_id_to_display_label:
            citation_id = token
        elif token in display_to_id:
            citation_id = display_to_id[token]
            repaired = True
        elif _STABLE_ID.fullmatch(token) or _DISPLAY_LABEL.fullmatch(token):
            unknown.append(token)
        elif _CITATION_LIKE.search(token):
            malformed.append(original)
        else:
            return original
        if citation_id is None:
            return original
        normalized = f"[{citation_id}]"
        if normalized != original:
            repaired = True
        used.append(citation_id)
        return normalized

    normalized_answer = _BRACKET_TOKEN.sub(normalize, answer)
    for match in _BARE_STABLE_ID.finditer(normalized_answer):
        if (
            match.start() == 0
            or match.end() == len(normalized_answer)
            or normalized_answer[match.start() - 1] != "["
            or normalized_answer[match.end()] != "]"
        ):
            malformed.append(match.group(0))
    malformed.extend(match.group(0) for match in _UNCLOSED_SOURCE.finditer(answer))
    result = CitationValidationResult(
        valid=not unknown and not malformed,
        used_citation_ids=_unique(used),
        unknown_citation_ids=_unique(unknown),
        malformed_citations=_unique(malformed),
        repaired=repaired,
    )
    return normalized_answer, result


def render_display_citations(
    answer: str,
    citation_id_to_display_label: dict[str, str],
) -> str:
    """Replace only exact validated stable-ID tokens with readable labels."""
    _validate_mapping(citation_id_to_display_label)
    if not citation_id_to_display_label:
        return answer
    alternatives = "|".join(
        re.escape(citation_id)
        for citation_id in sorted(
            citation_id_to_display_label, key=len, reverse=True
        )
    )
    pattern = re.compile(rf"\[({alternatives})\]")
    return pattern.sub(
        lambda match: f"[{citation_id_to_display_label[match.group(1)]}]",
        answer,
    )


def _collection_prefix(collection: str) -> str:
    """Create a safe readable prefix without filesystem or database identifiers."""
    alphanumeric = "".join(character for character in collection.upper() if character.isalnum())
    prefix = alphanumeric[:4]
    return prefix if len(prefix) >= 2 else (prefix + "SRC")[:3]


def _validate_mapping(mapping: dict[str, str]) -> None:
    """Require a one-to-one mapping of valid IDs to sequential display labels."""
    if len(mapping) != len(set(mapping.values())):
        raise ValueError("Citation display labels must map one-to-one")
    for citation_id, label in mapping.items():
        if not _STABLE_ID.fullmatch(citation_id):
            raise ValueError(f"Invalid citation ID: {citation_id}")
        if not _DISPLAY_LABEL.fullmatch(label):
            raise ValueError(f"Invalid citation display label: {label}")


def _unique(values: list[str]) -> tuple[str, ...]:
    """Deduplicate diagnostics while preserving answer order."""
    return tuple(dict.fromkeys(values))
