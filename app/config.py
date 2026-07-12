"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "minirag_documents"
DEFAULT_TOP_K = 4
DEFAULT_MAX_RETRIEVAL_DISTANCE = 1.2
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
    )
