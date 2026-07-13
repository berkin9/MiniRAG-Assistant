"""Tests for logical collection management and physical isolation."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.config import Settings
from app.services.collections import (
    CollectionNameError,
    CollectionRegistry,
    normalize_collection_name,
)
from app.services.hashing import hash_file
from app.services.indexing import index_document
from app.services.retrieval import retrieve
from app.services.vector_store import ChromaVectorStore


class FixedEmbedder:
    """Provide deterministic vectors without loading a model."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        del query
        return [1.0, 0.0]


def test_collection_normalization_and_physical_names() -> None:
    """Logical names should normalize once and resolve under the base name."""
    registry = CollectionRegistry(
        "minirag_documents",
        "General",
        ("project", "Technical Docs", "project"),
    )

    assert normalize_collection_name("  Technical Docs ") == "technical-docs"
    assert registry.default_collection == "general"
    assert registry.list_collections() == (
        "general",
        "project",
        "technical-docs",
    )
    assert registry.physical_name("Project") == "minirag_documents__project"
    assert registry.physical_name() == "minirag_documents"


@pytest.mark.parametrize(
    "name",
    ["", "../policies", "technical/docs", "technical\\docs", "project@home"],
)
def test_unsafe_collection_names_are_rejected(name: str) -> None:
    """Traversal and unsafe punctuation must never be silently normalized."""
    with pytest.raises(CollectionNameError):
        normalize_collection_name(name)


def test_default_collection_is_backward_compatible() -> None:
    """Omitted collection selection should resolve to configured general."""
    settings = Settings()
    registry = CollectionRegistry(
        settings.chroma_collection_name,
        settings.default_rag_collection,
        settings.rag_collections,
    )

    assert registry.resolve_logical_name(None) == "general"


def test_indexing_and_retrieval_are_isolated_by_collection(tmp_path: Path) -> None:
    """Each logical collection should own deduplication and retrieval state."""
    registry = CollectionRegistry("test_documents", "general", ("project", "technical"))
    project_store = ChromaVectorStore(
        tmp_path / "chroma", registry.physical_name("project")
    )
    technical_store = ChromaVectorStore(
        tmp_path / "chroma", registry.physical_name("technical")
    )
    source = tmp_path / "shared.txt"
    source.write_text("shared document", encoding="utf-8")
    embedder = FixedEmbedder()

    project_result = index_document(
        source, 100, 10, embedder, project_store, "project"
    )
    technical_result = index_document(
        source, 100, 10, embedder, technical_store, "technical"
    )
    repeated = index_document(
        source, 100, 10, embedder, technical_store, "technical"
    )

    assert project_result.status == "indexed"
    assert technical_result.status == "indexed"
    assert repeated.status == "already_indexed"
    assert project_store.count() == technical_store.count() == 1

    technical_only = tmp_path / "implementation.txt"
    technical_only.write_text("authentication implementation", encoding="utf-8")
    index_document(
        technical_only, 100, 10, embedder, technical_store, "technical"
    )
    assert technical_store.has_document(hash_file(technical_only))
    assert not project_store.has_document(hash_file(technical_only))

    project_search = retrieve(
        "authentication", 4, 1.2, embedder, project_store, "project"
    )
    technical_search = retrieve(
        "authentication", 4, 1.2, embedder, technical_store, "technical"
    )

    assert {result.text for result in project_search.results} == {"shared document"}
    assert {result.text for result in technical_search.results} == {
        "shared document",
        "authentication implementation",
    }
    assert all(result.collection == "project" for result in project_search.results)
    assert all(
        result.collection == "technical" for result in technical_search.results
    )
