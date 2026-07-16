"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from dotenv import load_dotenv

from app.services.collections import CollectionRegistry

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "minirag_documents"
DEFAULT_TOP_K = 4
DEFAULT_MAX_RETRIEVAL_DISTANCE = 1.2
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_ANSWER_TEMPERATURE = 0.2
DEFAULT_MAX_ANSWER_TOKENS = 500
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_MAX_CONTEXT_CHARACTERS = 12_000
DEFAULT_MAX_UPLOAD_SIZE_MB = 10
DEFAULT_RAG_COLLECTION = "general"
DEFAULT_RAG_COLLECTIONS = ("general", "project", "technical", "policies")
DEFAULT_RAG_ROUTING_MODE = "deterministic"
DEFAULT_QUERY_MODE = "manual"
DEFAULT_AGENT_PLANNING_MODE = "deterministic"
DEFAULT_AGENT_PLANNING_TEMPERATURE = 0.0
DEFAULT_AGENT_MAX_PLANNING_TOKENS = 400
SUPPORTED_RAG_ROUTING_MODES = frozenset({"deterministic", "llm"})
SUPPORTED_QUERY_MODES = frozenset({"manual", "automatic"})
SUPPORTED_AGENT_PLANNING_MODES = frozenset({"deterministic", "llm"})
SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "gemini"})
SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".pdf"})


class ConfigurationError(ValueError):
    """Raised when an environment setting is invalid."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings for local document processing."""

    data_dir: Path = Path("data")
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chroma_persist_dir: Path = Path(".chroma")
    chroma_collection_name: str = DEFAULT_COLLECTION_NAME
    default_top_k: int = DEFAULT_TOP_K
    max_retrieval_distance: float = DEFAULT_MAX_RETRIEVAL_DISTANCE
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    openai_api_key: str = ""
    gemini_api_key: str = ""
    upload_dir: Path = Path("data/uploads")
    answer_temperature: float = DEFAULT_ANSWER_TEMPERATURE
    max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS
    max_upload_size_mb: int = DEFAULT_MAX_UPLOAD_SIZE_MB
    default_rag_collection: str = DEFAULT_RAG_COLLECTION
    rag_collections: tuple[str, ...] = DEFAULT_RAG_COLLECTIONS
    rag_routing_mode: str = DEFAULT_RAG_ROUTING_MODE
    default_query_mode: str = DEFAULT_QUERY_MODE
    agent_planning_mode: str = DEFAULT_AGENT_PLANNING_MODE
    agent_planning_temperature: float = DEFAULT_AGENT_PLANNING_TEMPERATURE
    agent_max_planning_tokens: int = DEFAULT_AGENT_MAX_PLANNING_TOKENS

    def __post_init__(self) -> None:
        """Validate chunk settings."""
        if self.chunk_size <= 0:
            raise ConfigurationError("CHUNK_SIZE must be greater than zero")
        if self.chunk_overlap < 0:
            raise ConfigurationError("CHUNK_OVERLAP must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
            )
        if not self.embedding_model.strip():
            raise ConfigurationError("EMBEDDING_MODEL must not be empty")
        if not self.chroma_collection_name.strip():
            raise ConfigurationError("CHROMA_COLLECTION_NAME must not be empty")
        if self.default_top_k <= 0:
            raise ConfigurationError("DEFAULT_TOP_K must be greater than zero")
        if not isfinite(self.max_retrieval_distance):
            raise ConfigurationError(
                "MAX_RETRIEVAL_DISTANCE must be a finite number"
            )
        if self.max_retrieval_distance < 0:
            raise ConfigurationError(
                "MAX_RETRIEVAL_DISTANCE must be non-negative"
            )
        if self.llm_provider not in SUPPORTED_LLM_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
            raise ConfigurationError(
                f"LLM_PROVIDER must be one of: {supported}"
            )
        if not self.llm_model.strip():
            raise ConfigurationError("LLM_MODEL must not be empty")
        if not 0 <= self.answer_temperature <= 2:
            raise ConfigurationError("ANSWER_TEMPERATURE must be between 0 and 2")
        if self.max_answer_tokens <= 0:
            raise ConfigurationError("MAX_ANSWER_TOKENS must be greater than zero")
        if self.request_timeout <= 0:
            raise ConfigurationError("LLM_REQUEST_TIMEOUT must be greater than zero")
        if self.max_context_characters <= 0:
            raise ConfigurationError(
                "MAX_CONTEXT_CHARACTERS must be greater than zero"
            )
        if self.max_upload_size_mb <= 0:
            raise ConfigurationError("MAX_UPLOAD_SIZE_MB must be greater than zero")
        try:
            registry = CollectionRegistry(
                self.chroma_collection_name,
                self.default_rag_collection,
                self.rag_collections,
            )
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
        object.__setattr__(
            self, "default_rag_collection", registry.default_collection
        )
        object.__setattr__(self, "rag_collections", registry.list_collections())
        if self.rag_routing_mode not in SUPPORTED_RAG_ROUTING_MODES:
            supported = ", ".join(sorted(SUPPORTED_RAG_ROUTING_MODES))
            raise ConfigurationError(
                f"RAG_ROUTING_MODE must be one of: {supported}"
            )
        if self.default_query_mode not in SUPPORTED_QUERY_MODES:
            supported = ", ".join(sorted(SUPPORTED_QUERY_MODES))
            raise ConfigurationError(
                f"DEFAULT_QUERY_MODE must be one of: {supported}"
            )
        if self.agent_planning_mode not in SUPPORTED_AGENT_PLANNING_MODES:
            supported = ", ".join(sorted(SUPPORTED_AGENT_PLANNING_MODES))
            raise ConfigurationError(
                f"AGENT_PLANNING_MODE must be one of: {supported}"
            )
        if not 0 <= self.agent_planning_temperature <= 2:
            raise ConfigurationError(
                "AGENT_PLANNING_TEMPERATURE must be between 0 and 2"
            )
        if self.agent_max_planning_tokens <= 0:
            raise ConfigurationError(
                "AGENT_MAX_PLANNING_TOKENS must be greater than zero"
            )


def _read_int(name: str, default: int) -> int:
    """Read an integer environment variable with a clear error."""
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}") from error


def _read_float(name: str, default: float) -> float:
    """Read a floating-point environment variable with a clear error."""
    value = os.getenv(name, str(default))
    try:
        return float(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number, got {value!r}") from error


def get_settings() -> Settings:
    """Load settings from a local .env file and the environment."""
    load_dotenv()
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        chunk_size=_read_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_read_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", ".chroma")),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME", DEFAULT_COLLECTION_NAME
        ),
        default_top_k=_read_int("DEFAULT_TOP_K", DEFAULT_TOP_K),
        max_retrieval_distance=_read_float(
            "MAX_RETRIEVAL_DISTANCE", DEFAULT_MAX_RETRIEVAL_DISTANCE
        ),
        llm_provider=os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower(),
        llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        upload_dir=Path(os.getenv("UPLOAD_DIR", "data/uploads")),
        answer_temperature=_read_float(
            "ANSWER_TEMPERATURE", DEFAULT_ANSWER_TEMPERATURE
        ),
        max_answer_tokens=_read_int(
            "MAX_ANSWER_TOKENS", DEFAULT_MAX_ANSWER_TOKENS
        ),
        request_timeout=_read_float(
            "LLM_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT
        ),
        max_context_characters=_read_int(
            "MAX_CONTEXT_CHARACTERS", DEFAULT_MAX_CONTEXT_CHARACTERS
        ),
        max_upload_size_mb=_read_int(
            "MAX_UPLOAD_SIZE_MB", DEFAULT_MAX_UPLOAD_SIZE_MB
        ),
        default_rag_collection=os.getenv(
            "DEFAULT_RAG_COLLECTION", DEFAULT_RAG_COLLECTION
        ),
        rag_collections=tuple(
            name.strip()
            for name in os.getenv(
                "RAG_COLLECTIONS", ",".join(DEFAULT_RAG_COLLECTIONS)
            ).split(",")
            if name.strip()
        ),
        rag_routing_mode=os.getenv(
            "RAG_ROUTING_MODE", DEFAULT_RAG_ROUTING_MODE
        ).lower(),
        default_query_mode=os.getenv(
            "DEFAULT_QUERY_MODE", DEFAULT_QUERY_MODE
        ).lower(),
        agent_planning_mode=os.getenv(
            "AGENT_PLANNING_MODE", DEFAULT_AGENT_PLANNING_MODE
        ).lower(),
        agent_planning_temperature=_read_float(
            "AGENT_PLANNING_TEMPERATURE", DEFAULT_AGENT_PLANNING_TEMPERATURE
        ),
        agent_max_planning_tokens=_read_int(
            "AGENT_MAX_PLANNING_TOKENS", DEFAULT_AGENT_MAX_PLANNING_TOKENS
        ),
    )
