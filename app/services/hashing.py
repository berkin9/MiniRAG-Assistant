"""Stable content hashing utilities."""

import hashlib
from pathlib import Path


def hash_file(path: str | Path, block_size: int = 65_536) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    if block_size <= 0:
        raise ValueError("block_size must be greater than zero")
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
