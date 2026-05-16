# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) system that ingests PDFs, stores vector embeddings in ChromaDB, and answers questions via a FastAPI backend + HTML frontend. Uses Google Gemini for both embeddings and generation.

## Architecture

**Ingest pipeline** (`database.py`):
1. Loads all PDFs from `data/` (capped at first 50 pages via `load_documents()`)
2. Splits into chunks (800 chars, 80 overlap) using `RecursiveCharacterTextSplitter`
3. Assigns chunk IDs in format `{source_path}:{page}:{chunk_index}`
4. Adds new-only chunks to ChromaDB in batches of 100, sleeping 60s between batches to respect Gemini embedding API rate limits (1,500 daily requests, with a per-minute limit)

**Query pipeline** (`api.py`):
1. Embeds the query and retrieves top-5 similar chunks from ChromaDB
2. Builds a prompt with retrieved context and calls `gemini-2.5-flash-lite`
3. Returns the answer + source metadata (file, page, chunk index, similarity score)

**Shared embedding** (`embedding.py`): Single function `get_embedding_function()` used by both `database.py` and `api.py` — returns a `GoogleGenerativeAIEmbeddings` instance using `models/gemini-embedding-001`.

**Frontend** (`rag_query_ui.html`): Static HTML/JS served directly by FastAPI at `/`. Makes POST requests to `/query`.

## Common Commands

**Install dependencies** (inside the `rag_env` virtual environment):
```bash
source rag_env/Scripts/activate   # Windows Git Bash
pip install fastapi uvicorn langchain langchain-chroma langchain-community langchain-google-genai pypdf
```

**Populate / update the ChromaDB database:**
```bash
python database.py            # incremental — only adds new chunks
python database.py --reset    # wipe chroma/ and rebuild from scratch
```

**Run the API server:**
```bash
uvicorn api:app --reload --port 8000
```
Then open `http://localhost:8000` for the UI, or POST to `http://localhost:8000/query`.

## Configuration

- `CHROMA_PATH` — ChromaDB persistence directory (default: `chroma/`, gitignored)
- `DATA_PATH` — PDF source directory (default: `data/`)
- Google API key is currently hardcoded in both `api.py` and `embedding.py`. The `.gitignore` includes `.env` — move the key there and load it with `python-dotenv` or `os.environ`.

## Key Constraints

- `load_documents()` in `database.py` truncates to the first 50 document pages — increase this limit if you need full-document ingestion.
- The 60-second sleep between embedding batches exists to avoid Gemini's per-minute rate limit; do not remove it without switching to a higher-quota API key.
- ChromaDB deduplicates by chunk ID, so re-running `database.py` without `--reset` is safe and only adds genuinely new chunks.
