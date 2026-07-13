# U.S. Copyright Law RAG Agent — API

FastAPI deployment of the RAG system prototyped in `Copyright RAG Agent.ipynb`.

## Layout

- `rag_agent.py` — `CopyrightRAGAgent` class containing all RAG logic (hybrid
  retrieval, semantic cache, guardrails, agent).
- `app.py` — FastAPI application exposing the agent over HTTP.

## Prerequisites

The vector store and semantic cache must already be built (the notebook creates
`./chroma_rag_db` and `./chroma_cache_db`). Run the API from the project root so
the default relative paths resolve, or override them with env vars.

## Setup

```bash
pip install -r app/requirements.txt
export OPENAI_API_KEY=sk-...          # required
```

## Run

```bash
# from the project root (CopyrightAgent/)
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

## Run with Docker

Containerization files live in the project root: `Dockerfile`, `.dockerignore`,
`docker-compose.yml`, and `.env.example`. The build context is the project root.

The large Chroma databases are **not** baked into the image — they are mounted
as volumes at run time, so they must already exist on the host (the notebook
builds `./chroma_rag_db` and `./chroma_cache_db`).

```bash
# from the project root (CopyrightAgent/)
cp .env.example .env          # then edit .env and set OPENAI_API_KEY
docker compose up --build
```

The API is then available at http://localhost:8000 (Swagger UI at `/docs`).

To build/run with plain Docker instead of compose:

```bash
docker build -t copyright-rag-api .
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/chroma_rag_db:/code/chroma_rag_db:ro" \
  -v "$(pwd)/chroma_cache_db:/code/chroma_cache_db" \
  copyright-rag-api
```

Notes:
- The knowledge-base volume is mounted read-only (`:ro`); the semantic-cache
  volume is writable so newly generated answers persist across restarts.
- The container runs as a non-root user (uid 1000). On Linux hosts, ensure the
  `chroma_cache_db` directory is writable by that uid (e.g. `chmod -R a+w` it, or
  `chown -R 1000`); on Docker Desktop (macOS/Windows) bind-mount permissions are
  handled automatically.
- A container `HEALTHCHECK` polls `/health`; `docker ps` shows health status.

## Endpoints

| Method | Path      | Description                              |
|--------|-----------|------------------------------------------|
| GET    | `/health` | Liveness / readiness probe.              |
| POST   | `/ask`    | Ask a U.S. copyright-law question.       |

### Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the four factors of fair use under Section 107?"}'
```

## Configuration (environment variables)

| Variable                    | Default              | Description                                  |
|-----------------------------|----------------------|----------------------------------------------|
| `OPENAI_API_KEY`            | _(required)_         | OpenAI API key.                              |
| `RAG_CHAT_MODEL`            | `gpt-5.4`            | Chat model for the agent.                    |
| `RAG_PERSIST_PATH`          | `./chroma_rag_db`    | Path to the knowledge-base vector store.     |
| `RAG_CACHE_PERSIST_PATH`    | `./chroma_cache_db`  | Path to the semantic cache store.            |
| `RAG_CACHE_COLLECTION`      | `semantic_cache`     | Cache collection name.                       |
| `RAG_CACHE_EMBED_MODEL`     | `text-embedding-3-small` | Embedding model for the cache.           |
| `RAG_CACHE_THRESHOLD`       | `0.9`                | Min relevance score for a cache hit.         |
| `RAG_RETRIEVER_K`           | `8`                  | Top-k for each retriever in the hybrid set.  |
| `RAG_ENABLE_PROMPT_SCANNER` | `true`               | Enable the Sunglasses injection scanner.     |
| `RAG_RETRY_MAX_ATTEMPTS`    | `4`                  | Total attempts per external call (1 + retries). |
| `RAG_RETRY_BASE_WAIT`       | `1.0`                | Backoff multiplier in seconds (exponential). |
| `RAG_RETRY_MAX_WAIT`        | `30.0`               | Max backoff between attempts, in seconds.    |

## Fault tolerance

Every external call (the agent's LLM invocation, hybrid retrieval, and the
semantic-cache read/write) is wrapped with **exponential-backoff retries**
(`tenacity`). Only transient errors are retried — OpenAI rate limits, timeouts,
connection errors, and 5xx responses — while permanent errors (auth, bad
request) fail fast. Waits grow as `base * 2^(n-1)`, capped at `RAG_RETRY_MAX_WAIT`.
If retries are exhausted, `/ask` responds with **503** and a `Retry-After` header.

> **Note:** the vector store is queried with the default `OpenAIEmbeddings()`
> model because that is what built the index in the notebook. Do not change the
> knowledge-base embedding model without rebuilding the index.
