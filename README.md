# MiniRAG Assistant

A local retrieval pipeline that loads documents, creates semantic embeddings,
stores chunks persistently, and searches them by meaning.

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
overlap, embedding model, Chroma index, and retrieval behavior. The embedding
model is downloaded on first indexing or search and then runs locally; no API
key is required.

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

## Index documents

```bash
python -m app.main index ./data
```

One file can be indexed by passing its path instead. Each chunk receives a
local embedding: a numeric representation that places semantically similar text
nearby. MiniRAG uses `sentence-transformers/all-MiniLM-L6-v2` because it is a
small, well-established model with a useful quality/speed balance for a local
portfolio application.

ChromaDB stores chunk text, embeddings, and source metadata under `.chroma/` by
default. A SHA-256 hash of the document bytes identifies the content, and stable
chunk IDs combine that hash with each chunk index. Re-indexing unchanged content
therefore skips embedding and storage, even if the same bytes have another name.

Example:

```text
data/example.txt: indexed (3 chunk(s))
Documents processed: 1
Chunks stored: 3
```

Running the same command again reports `already_indexed` and stores zero chunks.

## Search documents

```bash
python -m app.main search "What is the project deadline?"
python -m app.main search "What is the project deadline?" --top-k 2
```

The query is embedded with the same model and compared with stored vectors.
The collection uses cosine distance: lower is more relevant, and `0` means the
vectors point in the same direction. Results whose distance exceeds
`MAX_RETRIEVAL_DISTANCE` are omitted. The default maximum of `1.2` is an initial
tunable value rather than a universal relevance guarantee.

Example:

```text
1. data/example.txt, chunk 0, distance 0.3152
   The project deadline is Friday at 5 PM.
```

If no stored chunk passes the threshold, the command reports
`No relevant results found.`

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_DIR` | `data` | Default ingestion/indexing directory |
| `CHUNK_SIZE` | `800` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `150` | Characters retained between chunks |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local model |
| `CHROMA_PERSIST_DIR` | `.chroma` | Persistent index directory |
| `CHROMA_COLLECTION_NAME` | `minirag_documents` | Chroma collection |
| `DEFAULT_TOP_K` | `4` | Maximum search matches |
| `MAX_RETRIEVAL_DISTANCE` | `1.2` | Largest accepted cosine distance |

## Data flow

```text
Document → Loader → Chunker → Embedding Model → ChromaDB
Question → Embedding Model → ChromaDB Search → Relevant Chunks
```

The implementation keeps these responsibilities in separate services:

- `hashing.py` calculates stable content hashes.
- `embeddings.py` lazily loads and reuses the local model.
- `vector_store.py` translates between domain data and ChromaDB.
- `indexing.py` orchestrates ingestion, deduplication, embedding, and storage.
- `retrieval.py` validates queries, applies distance thresholds, and returns
  structured matches.

To clear the local index, first stop any running MiniRAG process, verify
`CHROMA_PERSIST_DIR`, and remove that directory. With the default configuration:

```bash
rm -rf .chroma
```

This permanently removes only the generated vector index; source documents are
not modified.

## Current limitations

- Search returns relevant chunks rather than a generated answer.
- There is no LLM, web interface, or hybrid keyword search.
- Changed content is indexed as a new document; stale versions are not
  automatically deleted.

## Test

```bash
pytest
```
