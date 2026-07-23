"""Tests for environment configuration."""

import pytest

from app.config import ConfigurationError, get_settings


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CHUNK_SIZE", "0", "CHUNK_SIZE must be greater than zero"),
        ("CHUNK_SIZE", "nope", "CHUNK_SIZE must be an integer"),
        ("CHUNK_OVERLAP", "-1", "CHUNK_OVERLAP must be non-negative"),
        ("CHUNK_OVERLAP", "800", "CHUNK_OVERLAP must be smaller"),
    ],
)
def test_invalid_configuration_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    """Invalid environment values should report their setting names."""
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        get_settings()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("DEFAULT_TOP_K", "0", "DEFAULT_TOP_K must be greater than zero"),
        ("DEFAULT_TOP_K", "many", "DEFAULT_TOP_K must be an integer"),
        (
            "MAX_RETRIEVAL_DISTANCE",
            "-0.1",
            "MAX_RETRIEVAL_DISTANCE must be non-negative",
        ),
        (
            "MAX_RETRIEVAL_DISTANCE",
            "near",
            "MAX_RETRIEVAL_DISTANCE must be a number",
        ),
        (
            "MAX_RETRIEVAL_DISTANCE",
            "nan",
            "MAX_RETRIEVAL_DISTANCE must be a finite number",
        ),
    ],
)
def test_invalid_retrieval_configuration(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    """Invalid retrieval settings should produce clear errors."""
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        get_settings()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LLM_PROVIDER", "other", "LLM_PROVIDER"),
        ("LLM_MODEL", " ", "LLM_MODEL must not be empty"),
        ("ANSWER_TEMPERATURE", "2.1", "must be between 0 and 2"),
        ("MAX_ANSWER_TOKENS", "0", "must be greater than zero"),
        ("LLM_REQUEST_TIMEOUT", "0", "must be greater than zero"),
        ("MAX_CONTEXT_CHARACTERS", "0", "must be greater than zero"),
        ("MAX_UPLOAD_SIZE_MB", "0", "must be greater than zero"),
    ],
)
def test_invalid_answer_configuration(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    """Answer and upload settings should be validated without requiring API keys."""
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        get_settings()


def test_collection_configuration_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment collection choices should be normalized and deduplicated."""
    monkeypatch.setenv("DEFAULT_RAG_COLLECTION", "General")
    monkeypatch.setenv("RAG_COLLECTIONS", "project, Technical Docs,project")

    settings = get_settings()

    assert settings.default_rag_collection == "general"
    assert settings.rag_collections == ("general", "project", "technical-docs")


def test_demo_indexing_configuration_uses_defaults_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared demo settings should follow existing path and boolean parsing."""
    defaults = get_settings()
    assert defaults.demo_data_dir.as_posix() == "data/demo"
    assert defaults.auto_index_demo_documents is True

    monkeypatch.setenv("DEMO_DATA_DIR", "fixtures/shared-demo")
    monkeypatch.setenv("AUTO_INDEX_DEMO_DOCUMENTS", "false")
    configured = get_settings()

    assert configured.demo_data_dir.as_posix() == "fixtures/shared-demo"
    assert configured.auto_index_demo_documents is False


def test_invalid_demo_indexing_boolean_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_INDEX_DEMO_DOCUMENTS", "sometimes")

    with pytest.raises(ConfigurationError, match="AUTO_INDEX_DEMO_DOCUMENTS"):
        get_settings()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RAG_ROUTING_MODE", "agent", "RAG_ROUTING_MODE must be one of"),
        ("DEFAULT_QUERY_MODE", "sometimes", "DEFAULT_QUERY_MODE must be one of"),
    ],
)
def test_invalid_routing_configuration(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    """Routing strategy and interaction mode must be validated at startup."""
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        get_settings()


def test_agent_planning_defaults_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sprint 1 must not opt existing agent execution into LLM planning."""
    monkeypatch.delenv("AGENT_PLANNING_MODE", raising=False)

    settings = get_settings()

    assert settings.agent_planning_mode == "deterministic"
    assert settings.agent_planning_temperature == 0.0
    assert settings.agent_max_planning_tokens == 400
    assert settings.agent_min_planning_confidence == 0.60
    assert settings.agent_max_steps == 2
    assert settings.agent_planning_fallback_enabled is True


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("AGENT_PLANNING_MODE", "hybrid", "AGENT_PLANNING_MODE must be one of"),
        (
            "AGENT_PLANNING_TEMPERATURE",
            "2.1",
            "AGENT_PLANNING_TEMPERATURE must be between 0 and 2",
        ),
        (
            "AGENT_MAX_PLANNING_TOKENS",
            "0",
            "AGENT_MAX_PLANNING_TOKENS must be greater than zero",
        ),
        (
            "AGENT_MIN_PLANNING_CONFIDENCE",
            "-0.1",
            "AGENT_MIN_PLANNING_CONFIDENCE must be between 0 and 1",
        ),
        (
            "AGENT_MIN_PLANNING_CONFIDENCE",
            "1.1",
            "AGENT_MIN_PLANNING_CONFIDENCE must be between 0 and 1",
        ),
        ("AGENT_MAX_STEPS", "0", "AGENT_MAX_STEPS must be at least 1"),
        ("AGENT_MAX_STEPS", "3", "AGENT_MAX_STEPS cannot exceed"),
        (
            "AGENT_PLANNING_FALLBACK_ENABLED",
            "maybe",
            "AGENT_PLANNING_FALLBACK_ENABLED must be a boolean",
        ),
    ],
)
def test_invalid_agent_planning_configuration(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    """Planner configuration errors should name the invalid setting."""
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        get_settings()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("1", True), ("false", False), ("0", False)],
)
def test_agent_planning_fallback_boolean_values(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    """Common environment boolean forms should parse deterministically."""
    monkeypatch.setenv("AGENT_PLANNING_FALLBACK_ENABLED", value)

    assert get_settings().agent_planning_fallback_enabled is expected
