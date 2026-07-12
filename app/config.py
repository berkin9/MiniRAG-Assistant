"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when an environment setting is invalid."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings for local document processing."""

    data_dir: Path = Path("data")
    chunk_size: int = 500
    chunk_overlap: int = 50

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


def _read_int(name: str, default: int) -> int:
    """Read an integer environment variable with a clear error."""
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}") from error


def get_settings() -> Settings:
    """Load settings from a local .env file and the environment."""
    load_dotenv()
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        chunk_size=_read_int("CHUNK_SIZE", 500),
        chunk_overlap=_read_int("CHUNK_OVERLAP", 50),
    )
