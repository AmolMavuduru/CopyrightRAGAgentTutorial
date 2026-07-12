"""FastAPI service that deploys the U.S. Copyright Law RAG agent.

Run locally with::

    export OPENAI_API_KEY=sk-...
    uvicorn app.app:app --reload --port 8000

The heavy :class:`CopyrightRAGAgent` (vector store, BM25 index, agent) is built
once on start-up via the lifespan handler and reused across all requests.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .rag_agent import CopyrightRAGAgent, _retryable_exceptions

# Transient upstream errors that survived the agent's retry/backoff. We surface
# these as 503 (Service Unavailable) so clients know to retry later.
RETRYABLE_EXCEPTIONS = _retryable_exceptions()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@app.post("/ask", response_model=AskResponse)
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
