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
