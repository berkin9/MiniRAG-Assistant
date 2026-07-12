"""Command-line entry point for local document chunking."""

import argparse
from pathlib import Path

from app.config import get_settings
from app.services.document_loader import load_document
from app.services.text_splitter import split_text


def main() -> None:
    """Load a document and report the generated text chunks."""
    parser = argparse.ArgumentParser(description="Split a local document into chunks.")
    parser.add_argument("document", type=Path, help="Path to a UTF-8 text document")
    args = parser.parse_args()

    settings = get_settings()
    text = load_document(args.document)
    chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    print(f"Created {len(chunks)} chunk(s) from {args.document}.")


if __name__ == "__main__":
    main()
