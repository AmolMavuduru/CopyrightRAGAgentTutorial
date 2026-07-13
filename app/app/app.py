"""FastAPI service that deploys the U.S. Copyright Law RAG agent.

Run locally with::

    export OPENAI_API_KEY=sk-...
    export RAG_API_KEY=some-long-random-secret   # required to call /ask
    uvicorn app.app:app --reload --port 8000

The heavy :class:`CopyrightRAGAgent` (vector store, BM25 index, agent) is built
once on start-up via the lifespan handler and reused across all requests.

The ``/ask`` endpoint is protected by an ``X-API-Key`` header (see
``require_api_key``); ``/`` and ``/health`` stay open so the ECS/ALB health
check works.
"""

from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import List, Optional, Set

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .rag_agent import CopyrightRAGAgent, _retryable_exceptions

# Transient upstream errors that survived the agent's retry/backoff. We surface
# these as 503 (Service Unavailable) so clients know to retry later.
RETRYABLE_EXCEPTIONS = _retryable_exceptions()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------- authentication

API_KEY_HEADER_NAME = "X-API-Key"
# auto_error=False so we can control the response and support an "unset" mode.
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def _configured_api_keys() -> Set[str]:
    """Accepted client API keys, parsed from the RAG_API_KEY env var.

    Supports a comma-separated list so keys can be rotated (old + new valid at
    once). Set this on the ECS service, the same way as OPENAI_API_KEY.
    """
    raw = os.getenv("RAG_API_KEY", "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    """FastAPI dependency enforcing the ``X-API-Key`` header on protected routes.

    If ``RAG_API_KEY`` is not configured, authentication is disabled (fail open)
    and a warning is logged — convenient for local dev, but you MUST set
    ``RAG_API_KEY`` on the ECS deployment. When configured, a valid key is
    required and compared in constant time to avoid timing attacks.
    """
    accepted = _configured_api_keys()
    if not accepted:
        logger.warning(
            "RAG_API_KEY is not set: the /ask endpoint is UNPROTECTED. "
            "Set RAG_API_KEY to require an X-API-Key header."
        )
        return

    if api_key and any(secrets.compare_digest(api_key, k) for k in accepted):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key.",
        headers={"WWW-Authenticate": API_KEY_HEADER_NAME},
    )


# --------------------------------------------------------------------- schemas


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="A question about U.S. copyright law.",
        examples=["What are the four factors of fair use under Section 107?"],
    )
    return_contexts: bool = Field(
        default=True,
        description="Whether to include the retrieved source contexts in the response.",
    )


class AskResponse(BaseModel):
    answer: str
    contexts: Optional[List[str]] = None
    cached: bool = Field(
        default=False,
        description="True if the answer was served from the semantic cache.",
    )


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool


# ------------------------------------------------------------------ app set-up

# Holds the singleton agent; populated by the lifespan handler.
state: dict = {"agent": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: building CopyrightRAGAgent...")
    state["agent"] = CopyrightRAGAgent()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")
    state["agent"] = None


app = FastAPI(
    title="U.S. Copyright Law RAG Agent",
    description="A retrieval-augmented-generation API for answering questions about U.S. copyright law.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_agent() -> CopyrightRAGAgent:
    agent = state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready yet.")
    return agent


# --------------------------------------------------------------------- routes


@app.get("/")
def root() -> dict:
    """Lightweight root endpoint.

    ECS Express Mode's load balancer health check defaults to the ``/`` path, so
    this must return a 2xx for the target group to be marked healthy. It is
    intentionally cheap and does not depend on the agent being ready.
    """
    return {"service": "copyright-rag-api", "status": "ok"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    agent_ready = state.get("agent") is not None
    return HealthResponse(status="ok" if agent_ready else "starting", agent_ready=agent_ready)


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(request: AskRequest) -> AskResponse:
    agent = get_agent()
    try:
        result = agent.answer(request.question, return_contexts=request.return_contexts)
    except RETRYABLE_EXCEPTIONS as exc:
        # Retries with exponential backoff were exhausted against the upstream.
        logger.exception("Upstream temporarily unavailable after retries.")
        return JSONResponse(
            status_code=503,
            content={"detail": f"Upstream temporarily unavailable: {exc}"},
            headers={"Retry-After": "30"},
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
        logger.exception("Error while answering question.")
        raise HTTPException(status_code=500, detail=f"Failed to answer: {exc}") from exc

    if request.return_contexts:
        answer, contexts = result
    else:
        answer, contexts = result, None

    cached = bool(contexts is not None and len(contexts) == 0)
    if answer == "Unsafe prompt detected":
        cached = False

    return AskResponse(answer=answer, contexts=contexts, cached=cached)
