"""Tests for provider selection and safe configuration errors."""

import pytest

from app.config import ConfigurationError, Settings
from app.services.llm_providers import (
    GeminiProvider,
    LLMConfigurationError,
    OpenAIProvider,
    build_llm_provider,
)


def test_builds_supported_openai_provider() -> None:
    """OpenAI settings should produce the isolated OpenAI adapter."""
    provider = build_llm_provider(
        Settings(llm_provider="openai", openai_api_key="secret-openai")
    )

    assert isinstance(provider, OpenAIProvider)


def test_builds_supported_gemini_provider() -> None:
    """Gemini settings should produce the isolated Gemini adapter."""
    provider = build_llm_provider(
        Settings(
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
            gemini_api_key="secret-gemini",
        )
    )

    assert isinstance(provider, GeminiProvider)


def test_unsupported_provider_is_rejected() -> None:
    """Unsupported provider names should fail configuration clearly."""
    with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
        Settings(llm_provider="unknown")


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_selected_provider_requires_key_without_leaking_it(provider: str) -> None:
    """Missing-key errors should name the setting but contain no secret."""
    settings = Settings(
        llm_provider=provider,
        llm_model="test-model",
        openai_api_key="",
        gemini_api_key="",
    )

    with pytest.raises(LLMConfigurationError) as captured:
        build_llm_provider(settings)

    message = str(captured.value)
    assert "API_KEY" in message
    assert "secret" not in message.lower()
