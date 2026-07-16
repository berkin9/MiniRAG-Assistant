"""Tests for bounded cross-collection configuration and selection."""

import json

import pytest
from pydantic import ValidationError

from app.config import ConfigurationError, Settings, get_settings
from app.services.collection_selection import (
    CollectionSelectionResult,
    DeterministicMultiCollectionSelector,
    LLMMultiCollectionSelector,
    ManualCollectionSelector,
)
from app.services.collections import CollectionRegistry
from app.services.llm_providers import LLMRequestError


class FakeProvider:
    """Return one controlled selection response and count calls."""

    def __init__(self, response: str, fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls = 0
        self.system_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del user_prompt
        self.calls += 1
        self.system_prompt = system_prompt
        if self.fail:
            raise LLMRequestError("provider secret")
        return self.response


def _registry() -> CollectionRegistry:
    return CollectionRegistry(
        "documents", "general", ("general", "project", "technical", "policies")
    )


def test_retrieval_strategy_defaults_to_single_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing configuration should retain the original retrieval path."""
    monkeypatch.delenv("RAG_RETRIEVAL_STRATEGY", raising=False)

    settings = get_settings()

    assert settings.rag_retrieval_strategy == "single_collection"
    assert settings.multirag_max_collections == 3
    assert settings.multirag_top_k_per_collection == 3
    assert settings.multirag_global_top_k == 6
    assert settings.multirag_deduplication_enabled is True


def test_cross_collection_strategy_is_accepted() -> None:
    settings = Settings(rag_retrieval_strategy="cross_collection")

    assert settings.rag_retrieval_strategy == "cross_collection"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RAG_RETRIEVAL_STRATEGY", "wide", "RAG_RETRIEVAL_STRATEGY"),
        ("MULTIRAG_MAX_COLLECTIONS", "0", "must be at least 1"),
        ("MULTIRAG_MAX_COLLECTIONS", "5", "registered collections"),
        ("MULTIRAG_TOP_K_PER_COLLECTION", "0", "must be at least 1"),
        ("MULTIRAG_TOP_K_PER_COLLECTION", "21", "hard limit"),
        ("MULTIRAG_GLOBAL_TOP_K", "0", "must be at least 1"),
        ("MULTIRAG_GLOBAL_TOP_K", "51", "hard limit"),
        ("MULTIRAG_DEDUPLICATION_ENABLED", "maybe", "must be a boolean"),
    ],
)
def test_invalid_multirag_configuration(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        get_settings()


def test_selection_model_rejects_empty_duplicate_and_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        CollectionSelectionResult(collections=(), strategy="manual")
    with pytest.raises(ValidationError, match="duplicates"):
        CollectionSelectionResult(
            collections=("technical", "technical"), strategy="manual"
        )
    with pytest.raises(ValidationError):
        CollectionSelectionResult(
            collections=("technical",), strategy="llm", confidence=1.1
        )


def test_manual_selector_validates_registry_and_maximum_before_use() -> None:
    selector = ManualCollectionSelector(
        _registry(), 2, ("technical", "policies")
    )

    result = selector.select("Compare implementation and policy")

    assert result.collections == ("technical", "policies")
    with pytest.raises(ValueError, match="Unknown collection"):
        ManualCollectionSelector(_registry(), 2, ("secret",)).select("query")
    with pytest.raises(ValueError, match="maximum"):
        ManualCollectionSelector(
            _registry(), 1, ("technical", "policies")
        ).select("query")


def test_deterministic_selector_selects_multiple_routes_in_stable_order() -> None:
    selector = DeterministicMultiCollectionSelector(_registry(), 3)

    result = selector.select(
        "Compare the authentication implementation with the project security policy."
    )

    assert result.collections == ("policies", "technical", "project")
    assert result.strategy == "deterministic"


def test_deterministic_selector_uses_default_and_never_needs_provider() -> None:
    selector = DeterministicMultiCollectionSelector(_registry(), 2)

    result = selector.select("What color is the cover?")

    assert result.collections == ("general",)
    assert result.confidence == 0.0


def test_deterministic_selector_enforces_maximum_and_handles_one_match() -> None:
    selector = DeterministicMultiCollectionSelector(_registry(), 1)

    one = selector.select("How is authentication implemented?")
    multiple = selector.select("Compare authentication with security policy")

    assert one.collections == ("technical",)
    assert len(multiple.collections) == 1


def test_llm_selector_accepts_one_strict_bounded_json_response() -> None:
    provider = FakeProvider(
        json.dumps(
            {
                "collections": ["technical", "policies"],
                "reason": "Implementation and constraints are both needed.",
                "confidence": 0.91,
            }
        )
    )
    fallback = DeterministicMultiCollectionSelector(_registry(), 2)
    selector = LLMMultiCollectionSelector(
        _registry(), 2, 0.6, lambda: provider, fallback
    )

    result = selector.select("Compare authentication with security policy")

    assert result.collections == ("technical", "policies")
    assert result.strategy == "llm"
    assert provider.calls == 1
    assert "raw JSON only" in provider.system_prompt


@pytest.mark.parametrize(
    "response",
    [
        "",
        "not-json",
        "```json\n{}\n```",
        json.dumps(
            {
                "collections": ["secret"],
                "reason": "Unknown.",
                "confidence": 0.9,
            }
        ),
        json.dumps(
            {
                "collections": ["technical", "policies", "project"],
                "reason": "Too many.",
                "confidence": 0.9,
            }
        ),
        json.dumps(
            {
                "collections": ["technical"],
                "reason": "Low confidence.",
                "confidence": 0.2,
            }
        ),
    ],
)
def test_invalid_llm_selection_falls_back_once_without_raw_output(
    response: str,
) -> None:
    provider = FakeProvider(response)
    fallback = DeterministicMultiCollectionSelector(_registry(), 2)
    selector = LLMMultiCollectionSelector(
        _registry(), 2, 0.6, lambda: provider, fallback
    )

    result = selector.select("authentication policy")

    assert result.collections == ("technical", "policies")
    assert result.strategy == "deterministic_fallback"
    assert result.fallback_used is True
    assert provider.calls == 1
    if response:
        assert response not in (result.fallback_reason or "")


def test_llm_provider_failure_uses_one_safe_deterministic_fallback() -> None:
    provider = FakeProvider("", fail=True)
    fallback = DeterministicMultiCollectionSelector(_registry(), 2)
    selector = LLMMultiCollectionSelector(
        _registry(), 2, 0.6, lambda: provider, fallback
    )

    result = selector.select("authentication policy")

    assert result.strategy == "deterministic_fallback"
    assert result.fallback_reason == "LLM selection failed: LLMRequestError"
    assert provider.calls == 1
    assert "secret" not in result.fallback_reason
