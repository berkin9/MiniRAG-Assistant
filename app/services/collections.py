"""Logical RAG collection validation and physical name resolution."""

import re
from dataclasses import dataclass

MAX_COLLECTION_NAME_LENGTH = 48
_VALID_COLLECTION_NAME = re.compile(r"^[a-z0-9_-]+$")


class CollectionNameError(ValueError):
    """Raised when a logical collection name is unsafe or invalid."""


def normalize_collection_name(name: str) -> str:
    """Normalize a safe logical collection name to a stable identifier."""
    raw_name = name.strip()
    if not raw_name:
        raise CollectionNameError("Collection name must not be empty")
    if "/" in raw_name or "\\" in raw_name or ".." in raw_name:
        raise CollectionNameError("Collection name must not contain path traversal")

    normalized = re.sub(r"\s+", "-", raw_name.lower())
    if len(normalized) > MAX_COLLECTION_NAME_LENGTH:
        raise CollectionNameError(
            f"Collection name must be at most {MAX_COLLECTION_NAME_LENGTH} characters"
        )
    if not _VALID_COLLECTION_NAME.fullmatch(normalized):
        raise CollectionNameError(
            "Collection name may contain only lowercase letters, numbers, "
            "hyphens, or underscores"
        )
    return normalized


@dataclass(frozen=True)
class CollectionRegistry:
    """Resolve configured logical collections to isolated Chroma collections."""

    base_collection_name: str
    default_collection: str = "general"
    configured_collections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize and validate collection configuration once."""
        if not self.base_collection_name.strip():
            raise CollectionNameError("Base Chroma collection name must not be empty")
        default = normalize_collection_name(self.default_collection)
        configured = _deduplicate(
            (
                default,
                *(
                    normalize_collection_name(name)
                    for name in self.configured_collections
                ),
            )
        )
        object.__setattr__(self, "default_collection", default)
        object.__setattr__(self, "configured_collections", configured)

    def resolve_logical_name(self, name: str | None = None) -> str:
        """Return the default or a validated user-supplied logical name."""
        return (
            self.default_collection
            if name is None
            else normalize_collection_name(name)
        )

    def physical_name(self, name: str | None = None) -> str:
        """Return the dedicated physical Chroma collection name."""
        logical_name = self.resolve_logical_name(name)
        if logical_name == "general":
            return self.base_collection_name
        return f"{self.base_collection_name}__{logical_name}"

    def list_collections(self) -> tuple[str, ...]:
        """Return configured collections in deterministic configuration order."""
        return self.configured_collections


def _deduplicate(names: tuple[str, ...]) -> tuple[str, ...]:
    """Deduplicate names while preserving their configured order."""
    return tuple(dict.fromkeys(names))
