# MiniRAG Assistant

A minimal local foundation for loading and splitting text documents before adding
retrieval or language-model integrations.

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

## Run

Place a UTF-8 text file in `data/`, then run:

```bash
python -m app.main data/example.txt
```

The command loads the document, splits it into overlapping character chunks,
and reports the number of chunks created. Vector storage and LLM calls are not
implemented yet.

## Test

```bash
pytest
```
