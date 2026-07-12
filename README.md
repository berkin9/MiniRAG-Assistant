# MiniRAG Assistant

A small local pipeline for loading documents and splitting them into
retrieval-ready chunks.

## Requirements

- Python 3.11 or newer

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

The values in `.env` control the default data directory, chunk size, and chunk
overlap.

## Ingest documents

Supported document types are UTF-8 `.txt`, UTF-8 `.md`, and `.pdf`. Place files
in `data/` (nested directories are supported), then run:

```bash
python -m app.main ingest ./data
```

Example output:

```text
INFO: Loaded document data/example.txt
Documents loaded: 1
Chunks created: 3
```

If the directory argument is omitted, the command uses `DATA_DIR` from `.env`.
`CHUNK_SIZE` and `CHUNK_OVERLAP` control chunking.

## Architecture

- `app/models/` contains the loaded document model.
- `app/services/document_loader.py` extracts document text and metadata.
- `app/services/text_splitter.py` splits text at natural boundaries with overlap.
- `app/services/ingestion.py` discovers, loads, and chunks local documents.
- `app/main.py` provides the command-line interface.

The project intentionally has no embeddings, vector database, or question
answering yet.

## Test

```bash
pytest
```
