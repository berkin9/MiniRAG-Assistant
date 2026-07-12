"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings for local document processing."""

    data_dir: str = "data"
    chunk_size: int = 500
    chunk_overlap: int = 50


def get_settings() -> Settings:
    """Load settings from a local .env file and the environment."""
    load_dotenv()
    return Settings(
        data_dir=os.getenv("DATA_DIR", "data"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
    )
