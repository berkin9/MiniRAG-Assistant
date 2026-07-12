"""Tests for environment configuration."""

import pytest

from app.config import ConfigurationError, get_settings


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CHUNK_SIZE", "0", "CHUNK_SIZE must be greater than zero"),
        ("CHUNK_SIZE", "nope", "CHUNK_SIZE must be an integer"),
        ("CHUNK_OVERLAP", "-1", "CHUNK_OVERLAP must be non-negative"),
        ("CHUNK_OVERLAP", "500", "CHUNK_OVERLAP must be smaller"),
    ],
)
def test_invalid_configuration_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    """Invalid environment values should report their setting names."""
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        get_settings()
