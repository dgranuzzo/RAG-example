# RAG Hybrid (OpenAI | FAISS | Chroma)

Hybrid Retrieval-Augmented Generation pipeline with three interchangeable backends:

- OpenAI File Search (vector store) with optional metadata filtering
- Local FAISS + BM25 → MMR → optional LLM re-ranking
- Local Chroma + BM25 → MMR → optional LLM re-ranking

It ingests plain text and PDFs, chunks them with overlap, fuses lexical and vector signals, optionally applies MMR and LLM reranking, and generates grounded answers strictly from retrieved context.

Main entry point: `rag_hybrid.py` (CLI).

## Features

- Text and PDF ingestion (`.txt`, `.pdf`)
- Optional per-file sidecar metadata via `<file>.<ext>.meta.json`
- Hybrid late fusion: vector + BM25 with tunable weight
- MMR (Maximal Marginal Relevance) for diversity-aware selection
- Optional LLM reranking to improve ordering of top passages
- Source and metadata included in the final answer context

## Requirements

- Python 3.11+
- OpenAI API key (Responses API + Embeddings + Vector Stores)
- OS packages: none special for CPU (FAISS uses `faiss-cpu`)

Install Python deps:

```bash
pip install -r requirements.txt
```

If you use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Put these in a `.env` file at the repo root or export them in your shell:

```
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4.1-mini          # optional, default
OPENAI_EMB_MODEL=text-embedding-3-small # optional, default
```

## Preparing Documents

Place input files under a folder (default glob `docs/*.*`).

- Supported: `.txt`, `.pdf` (PDF text extraction via `pypdf`)
- Optional sidecar metadata: add a JSON file next to each doc named `<file>.<ext>.meta.json`.

Example to enable category filtering on `docs/policy.pdf`:

`docs/policy.pdf.meta.json`

```json
{ "categoria": "legal" }
```

You can include any other keys; they’ll be attached to each chunk’s metadata and shown in the final context.

## CLI Usage

Run `rag_hybrid.py` and pick a backend:

```bash
python rag_hybrid.py --backend {openai|faiss|chroma} --q "Your question" [options]
```

Common options:

- `--docs_glob` Glob for files (default `docs/*.*`)
- `--q` The question (required)
- `--categoria` Optional metadata filter value for key `categoria`
- `--top_k` Passages to send to the generator (default 6)
- Fusion/MMR/rerank knobs:
  - `--hybrid_alpha` Vector vs lexical weight in fusion (0..1, default 0.6)
  - `--mmr_lambda` Relevance vs diversity (default 0.6)
  - `--mmr_keep` Max passages surviving MMR to next stage (default 24)
  - `--reranker {none|llm}` Enable LLM reranking (default `llm`)
  - `--rerank_top_m` Candidates sent to LLM reranker (default 24)

### Examples

OpenAI File Search (with optional metadata filter):

```bash
python rag_hybrid.py \
  --backend openai \
  --docs_glob "docs/*.*" \
  --q "What is our refund policy?" \
  --categoria legal \
  --top_k 6
```

Local FAISS + BM25 → MMR → optional LLM rerank:

```bash
python rag_hybrid.py \
  --backend faiss \
  --docs_glob "docs/*.*" \
  --q "How to install the agent?" \
  --top_k 6 \
  --hybrid_alpha 0.6 \
  --mmr_lambda 0.6 \
  --mmr_keep 24 \
  --reranker llm
```

Local Chroma + BM25 → MMR → optional LLM rerank (in-memory):

```bash
python rag_hybrid.py \
  --backend chroma \
  --docs_glob "docs/*.*" \
  --q "Architecture overview" \
  --top_k 6
```

Persist Chroma to a directory:

```bash
python rag_hybrid.py \
  --backend chroma \
  --docs_glob "docs/*.*" \
  --q "Architecture overview" \
  --top_k 6 \
  --persist_chroma ./.chroma
```

## How It Works

- Ingestion: loads text or extracts PDF text, chunks with overlap, attaches sidecar metadata.
- Retrieval:
  - OpenAI File Search: creates a temporary vector store, uploads files with metadata, queries with optional metadata filter.
  - FAISS (local): normalized embeddings + cosine similarity.
  - Chroma (local service): vector search via OpenAI embeddings inside Chroma.
- Fusion: combine vector and BM25 scores with `--hybrid_alpha`.
- MMR: select diverse, relevant subsets with `--mmr_lambda` and `--mmr_keep`.
- Optional LLM rerank: ask the model to score candidates 0–100 and reorder.
- Generation: concatenate top contexts (with file names and metadata) and answer strictly from them.

## Docker

This repo includes a `Dockerfile`, but it currently references `app_fastapi.py` and starts `uvicorn app_fastapi:app`. That file is not present here, so the Docker image will not run as-is.

Options:

- Update the `Dockerfile` to run the CLI (`python rag_hybrid.py ...`).
- Or add `app_fastapi.py` exposing an API that wraps the CLI.

## Notes & Limits

- OpenAI usage incurs cost; keep `--top_k` modest during testing.
- Metadata filter key is `categoria`. Additional keys are preserved and shown during generation.
- PDF text quality depends on `pypdf` extraction; scanned PDFs may need OCR first.

## License

MIT — see `LICENSE`.

