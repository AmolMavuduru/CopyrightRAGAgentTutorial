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
  -v "$(pwd)/chroma_rag_db:/code/chroma_rag_db" \
  -v "$(pwd)/chroma_cache_db:/code/chroma_cache_db" \
  copyright-rag-api
```

Notes:
- Both Chroma volumes are mounted read-write. ChromaDB is backed by SQLite,
  which opens its database file read-write (for `-wal`/`-shm` and lock files)
  even for read-only queries — a read-only (`:ro`) mount fails with
  "attempt to write a readonly database". Your queries still don't alter the
  knowledge-base contents.
- The container runs as a non-root user (uid 1000). On Linux hosts, ensure both
  `chroma_rag_db` and `chroma_cache_db` are writable by that uid (e.g.
  `chmod -R a+w` them, or `chown -R 1000`); on Docker Desktop (macOS/Windows)
  bind-mount permissions are handled automatically.
- A container `HEALTHCHECK` polls `/health`; `docker ps` shows health status.
- The container listens on **port 80** (the ECS Express Mode default). Locally,
  compose maps it to `localhost:8000`.

## Deploy to AWS ECS Express Mode (via ECR)

The `Dockerfile` is set up for an ECS Express Mode service using **default
settings**: container port `80`, health check path `/`, and X86_64 Fargate. The
Chroma vector store and semantic cache are baked into the image, so the service
is self-contained (Express default provides no volumes/EFS). The image is pinned
to `linux/amd64` so it runs on default Fargate even when built on Apple Silicon.

1. **Build** (from the project root). The `--platform` is already pinned in the
   Dockerfile, but pass it here too if your Docker defaults differ:

   ```bash
   docker build --platform linux/amd64 -t copyright-rag-api .
   ```

2. **Push to ECR:**

   ```bash
   AWS_ACCOUNT_ID=<your-account-id>
   AWS_REGION=<your-region>
   REPO=copyright-rag-api

   aws ecr create-repository --repository-name "$REPO" --region "$AWS_REGION" || true
   aws ecr get-login-password --region "$AWS_REGION" \
     | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

   docker tag copyright-rag-api:latest "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest"
   docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest"
   ```

3. **Create the Express Mode service** in the ECS console: choose **Express
   mode**, select this ECR image, and leave the defaults (port 80, health check
   `/`). Under the container's environment variables, add **`OPENAI_API_KEY`**
   (as a Secret via Secrets Manager is recommended). Click **Create** and use the
   generated Application URL once healthy.

Notes:
- Keep the default memory of **2 GB** or higher — startup loads the vector store
  and builds the BM25 index in memory.
- The `OPENAI_API_KEY` is **not** baked into the image; set it as an environment
  variable / secret on the ECS service.
- The semantic cache is writable but ephemeral on ECS (per-task, not shared or
  persisted across deployments). This is functionally fine — it just repopulates.

## Endpoints

| Method | Path      | Description                              |
|--------|-----------|------------------------------------------|
| GET    | `/`       | Cheap 200 for the ECS/ALB health check.  |
| GET    | `/health` | Liveness / readiness probe.              |
| POST   | `/ask`    | Ask a U.S. copyright-law question.       |

### Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"question": "What are the four factors of fair use under Section 107?"}'
```

## Authentication

The `/ask` endpoint requires an API key supplied in the **`X-API-Key`** request
header, checked against the `RAG_API_KEY` environment variable (constant-time
comparison). `/` and `/health` are intentionally left open so the ECS/ALB health
check succeeds.

- Set `RAG_API_KEY` on the ECS service (an environment variable, or a Secrets
  Manager secret — same as `OPENAI_API_KEY`). Use a long random value, e.g.
  `openssl rand -hex 32`.
- Comma-separate multiple keys (`RAG_API_KEY=oldkey,newkey`) to rotate without
  downtime.
- If `RAG_API_KEY` is **unset**, `/ask` is left open and a warning is logged —
  acceptable for local dev, but always set it in production.
- Missing or invalid keys get `401 Unauthorized`. The Swagger UI (`/docs`) has an
  **Authorize** button for supplying the key interactively.

No Dockerfile change is required — `RAG_API_KEY` is provided at run time like any
other environment variable.

## Configuration (environment variables)

| Variable                    | Default              | Description                                  |
|-----------------------------|----------------------|----------------------------------------------|
| `OPENAI_API_KEY`            | _(required)_         | OpenAI API key.                              |
| `RAG_API_KEY`               | _(unset = open)_     | Client API key(s) for `/ask`; comma-separate to rotate. |
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
