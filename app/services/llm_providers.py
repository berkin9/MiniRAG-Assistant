"""Provider-independent LLM interface and SDK adapters."""

from typing import Protocol

from app.config import Settings


class LLMProviderError(RuntimeError):
    """Base error for LLM provider failures."""


class LLMConfigurationError(LLMProviderError):
    """Raised when an LLM provider cannot be configured safely."""


class LLMRequestError(LLMProviderError):
    """Raised when an LLM provider request fails."""


class LLMProvider(Protocol):
    """Provider-independent text generation contract."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate answer text from system and user prompts."""


class OpenAIProvider:
    """Generate text with the OpenAI Responses API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI lazily and return provider-independent text."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key, timeout=self._timeout)
            response = client.responses.create(
                model=self._model,
                instructions=system_prompt,
                input=user_prompt,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            )
            text = response.output_text
        except Exception as error:
            raise LLMRequestError("OpenAI answer generation failed") from error
        if not text or not text.strip():
            raise LLMRequestError("OpenAI returned an empty response")
        return text.strip()


class GeminiProvider:
    """Generate text with the Google Gen AI SDK."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call Gemini lazily and return provider-independent text."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=int(self._timeout * 1_000)),
            )
            response = client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self._temperature,
                    max_output_tokens=self._max_tokens,
                ),
            )
            text = response.text
        except Exception as error:
            raise LLMRequestError("Gemini answer generation failed") from error
        if not text or not text.strip():
            raise LLMRequestError("Gemini returned an empty response")
        return text.strip()


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Build the configured provider without importing or creating an SDK client."""
    common = {
        "model": settings.llm_model,
        "temperature": settings.answer_temperature,
        "max_tokens": settings.max_answer_tokens,
        "timeout": settings.request_timeout,
    }
    if settings.llm_provider == "openai":
        if not settings.openai_api_key.strip():
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            )
        return OpenAIProvider(api_key=settings.openai_api_key, **common)
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key.strip():
            raise LLMConfigurationError(
                "GEMINI_API_KEY is required when LLM_PROVIDER=gemini"
            )
        return GeminiProvider(api_key=settings.gemini_api_key, **common)
    raise LLMConfigurationError(f"Unsupported LLM provider: {settings.llm_provider}")
