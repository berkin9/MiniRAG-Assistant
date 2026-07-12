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

### Ingestion flow

1. Recursively discover supported, non-hidden files.
2. Read TXT and Markdown as UTF-8, or extract PDF text page by page.
3. Reject unsupported, empty, and unreadable documents with clear errors.
4. Split text using paragraph, line, word, then character boundaries.
5. Attach the source file, file type, PDF page number, and chunk index to every
   chunk for future retrieval.

Chunks default to approximately 800 characters with 150 characters of overlap.
These values can be changed through `.env` without changing application code.

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
