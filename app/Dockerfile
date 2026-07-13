# Container image for the U.S. Copyright Law RAG Agent API,
# targeted at an Amazon ECS Express Mode deployment with DEFAULT settings.
#
# ECS Express Mode defaults (see AWS docs):
#   * Container port    : 80
#   * Health check path : /   (this image serves 200 at "/")
#   * Compute           : Fargate, X86_64 (linux/amd64) architecture
#
# Because Express Mode (default) provides no bind mounts / EFS, the Chroma
# vector store and semantic cache are baked into the image so the service is
# fully self-contained. The build context is the project root (CopyrightAgent/).

# Pin linux/amd64 so the image runs on default X86_64 Fargate even when built
# on an Apple Silicon / arm64 host. Remove the platform flag if you deploy to
# ARM64 (Graviton) tasks instead.
FROM --platform=linux/amd64 python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# Install Python deps. build-essential covers any deps without prebuilt wheels;
# libcap2-bin provides setcap so a non-root user can bind the privileged port 80.
# Both build-time tools are removed afterwards to keep the image small; the
# capability set on the Python binary persists.
COPY app/requirements.txt ./app/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libcap2-bin \
    && pip install --no-cache-dir -r app/requirements.txt \
    && setcap 'cap_net_bind_service=+ep' "$(readlink -f "$(which python3)")" \
    && apt-get purge -y --auto-remove build-essential libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

# Application code.
COPY app/ ./app/

# Bake in the knowledge base and semantic cache (Express default has no volumes).
# NOTE: the semantic cache is writable at runtime, but on ECS those writes land
# on the task's ephemeral storage and are not shared across tasks or preserved
# across deployments. That is fine functionally (it just repopulates).
COPY chroma_rag_db/ ./chroma_rag_db/
COPY chroma_cache_db/ ./chroma_cache_db/

# Run as an unprivileged user (can still bind port 80 thanks to setcap above).
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /code
USER appuser

# ECS Express Mode routes to container port 80 by default.
EXPOSE 80

# Container-level health check (separate from the ECS/ALB target-group check).
# start-period is generous because startup builds the BM25 index from the
# vector store before the app is ready to serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:80/health').status==200 else 1)"

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]
