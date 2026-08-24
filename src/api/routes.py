"""Production FastAPI routes for multimodal RAG and ingestion.

Endpoints
---------
GET  /health            Liveness + Redis + KB configuration probe
POST /v1/rag/query      Buffered RetrieveAndGenerate / Retrieve+Converse
POST /v1/rag/stream     SSE token stream (`text/event-stream`)
POST /v1/ingest         Write Bedrock metadata sidecar and start an ingestion job
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.deps import AppContainer
from api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    StreamEvent,
    StreamEventType,
)
from services.circuit_breaker import CircuitOpenError

logger = structlog.get_logger(__name__)

router = APIRouter()


def _container(request: Request) -> AppContainer:
    return request.app.state.container


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    container = _container(request)
    redis_ok = await container.cache.ping()
    kb_ok = bool(container.settings.knowledge_base_id)
    return HealthResponse(
        status="ok" if redis_ok else "degraded",
        redis=redis_ok,
        knowledge_base_configured=kb_ok,
        timestamp=datetime.now(UTC),
    )


@router.post("/v1/rag/query", response_model=RagQueryResponse, tags=["rag"])
async def rag_query(payload: RagQueryRequest, request: Request) -> RagQueryResponse:
    """Synchronous RAG. Prefer `/v1/rag/stream` for interactive UIs (lower TTFT)."""

    container = _container(request)
    try:
        return await container.rag.query(payload)
    except CircuitOpenError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("rag_query_failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="upstream retrieval or generation failed",
        ) from None


@router.post("/v1/rag/stream", tags=["rag"])
async def rag_stream(payload: RagQueryRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events stream of tokens, citations, cache stats, and a terminal done event."""

    container = _container(request)

    async def events() -> AsyncIterator[bytes]:
        try:
            async for event in container.rag.stream(payload):
                yield _sse(event)
        except CircuitOpenError as exc:
            yield _sse(
                StreamEvent(
                    event=StreamEventType.ERROR,
                    data={"message": str(exc), "code": "circuit_open"},
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("rag_stream_failed")
            yield _sse(
                StreamEvent(
                    event=StreamEventType.ERROR,
                    data={"message": str(exc)[:300], "code": "stream_error"},
                )
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/v1/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    """Attach tenant metadata and optionally trigger `StartIngestionJob`.

    For high-volume object uploads, prefer the EventBridge → SQS FIFO → Lambda
    path; this endpoint exists for controlled backfills and operator tooling.
    """

    container = _container(request)
    try:
        return await container.ingest.ingest(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("ingest_failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="failed to start knowledge base ingestion",
        ) from None


def _sse(event: StreamEvent) -> bytes:
    payload = json.dumps({"event": event.event, "data": event.data}, separators=(",", ":"))
    return f"event: {event.event}\ndata: {payload}\n\n".encode()
