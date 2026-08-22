"""Bedrock Knowledge Base RAG orchestration.

Query path
----------
1. Two-tier cache (L1 exact hash, L2 Nova semantic similarity).
2. Origin:
   a. Text queries prefer `retrieve_and_generate_stream` (token-by-token SSE,
      lowest TTFT) against Claude 5 Sonnet / Haiku via the model router.
   b. Image / crossmodal queries use `Retrieve` with `multimodalInputList`
      then `ConverseStream`. Nova Multimodal Embeddings Knowledge Bases do
      not fully support RetrieveAndGenerate for image queries (Bedrock 4xx);
      this split is the production-safe unification.
3. Write-through cache of the completed answer.

Tenant isolation is enforced with a Knowledge Base metadata filter on
`tenant_id` (and optional `category`) so one tenant cannot retrieve another
tenant's S3 Vectors entries.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog

from api.schemas import (
    CacheTier,
    Citation,
    ContentType,
    RagQueryRequest,
    RagQueryResponse,
    StreamEvent,
    StreamEventType,
)
from config import Settings
from observability.metrics import MetricsEmitter
from services.aws_clients import AwsClients
from services.cache_service import CachedAnswer, CacheService
from services.circuit_breaker import CircuitBreaker, CircuitOpenError
from services.model_router import ModelRouter

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a grounded multimodal assistant. Answer ONLY from the retrieved "
    "knowledge-base context. If the context is insufficient, say so. Cite "
    "source URIs when available. Do not invent product facts or visual details."
)


class RagService:
    def __init__(
        self,
        settings: Settings,
        clients: AwsClients,
        cache: CacheService,
        router: ModelRouter,
        agent_breaker: CircuitBreaker,
        runtime_breaker: CircuitBreaker,
        metrics: MetricsEmitter,
    ) -> None:
        self._settings = settings
        self._clients = clients
        self._cache = cache
        self._router = router
        self._agent_breaker = agent_breaker
        self._runtime_breaker = runtime_breaker
        self._metrics = metrics

    async def query(self, request: RagQueryRequest) -> RagQueryResponse:
        """Buffered (non-streaming) query used by POST /v1/rag/query."""

        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        cached, stats = (None, None)
        if not request.bypass_cache:
            cached, stats = await self._cache.lookup(request)
            if cached is not None:
                cached.request_id = request_id
                cached.latency_ms = (time.perf_counter() - started) * 1000
                return cached
        assert stats is not None or request.bypass_cache

        answer_parts: list[str] = []
        citations: list[Citation] = []
        model_id = self._router.resolve(request)
        async for event in self._origin_stream(request, request_id, model_id):
            if event.event is StreamEventType.TOKEN:
                answer_parts.append(str(event.data.get("text", "")))
            elif event.event is StreamEventType.CITATION:
                citations.append(Citation.model_validate(event.data))

        answer = "".join(answer_parts).strip()
        latency_ms = (time.perf_counter() - started) * 1000
        response = RagQueryResponse(
            answer=answer,
            citations=citations,
            model_id=model_id,
            session_id=request.session_id,
            cache=stats
            if stats is not None
            else stats_miss(),
            request_id=request_id,
            latency_ms=latency_ms,
        )
        await self._store_cache(request, stats, response)
        self._metrics.record_rag_latency(
            latency_ms=latency_ms,
            model_id=model_id,
            modality=request.modality,
        )
        return response

    async def stream(self, request: RagQueryRequest) -> AsyncIterator[StreamEvent]:
        """Token-by-token generator mapped to SSE by the FastAPI route."""

        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        model_id = self._router.resolve(request)
        yield StreamEvent(
            event=StreamEventType.SESSION,
            data={"request_id": request_id, "model_id": model_id, "session_id": request.session_id},
        )

        if not request.bypass_cache:
            cached, stats = await self._cache.lookup(request)
            yield StreamEvent(event=StreamEventType.CACHE, data=stats.model_dump())
            if cached is not None:
                yield StreamEvent(event=StreamEventType.TOKEN, data={"text": cached.answer})
                for citation in cached.citations:
                    yield StreamEvent(event=StreamEventType.CITATION, data=citation.model_dump())
                yield StreamEvent(
                    event=StreamEventType.METRICS,
                    data={"latency_ms": (time.perf_counter() - started) * 1000, "cache_tier": stats.tier},
                )
                yield StreamEvent(event=StreamEventType.DONE, data={"request_id": request_id})
                return
        else:
            stats = stats_miss()
            yield StreamEvent(event=StreamEventType.CACHE, data=stats.model_dump())

        answer_parts: list[str] = []
        citations: list[Citation] = []
        try:
            async for event in self._origin_stream(request, request_id, model_id):
                if event.event is StreamEventType.TOKEN:
                    answer_parts.append(str(event.data.get("text", "")))
                elif event.event is StreamEventType.CITATION:
                    citations.append(Citation.model_validate(event.data))
                yield event
        except CircuitOpenError as exc:
            yield StreamEvent(event=StreamEventType.ERROR, data={"message": str(exc), "code": "circuit_open"})
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("origin_stream_failed")
            yield StreamEvent(event=StreamEventType.ERROR, data={"message": str(exc)[:500], "code": "origin_error"})
            return

        response = RagQueryResponse(
            answer="".join(answer_parts).strip(),
            citations=citations,
            model_id=model_id,
            session_id=request.session_id,
            cache=stats,
            request_id=request_id,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        await self._store_cache(request, stats, response)
        self._metrics.record_rag_latency(
            latency_ms=response.latency_ms,
            model_id=model_id,
            modality=request.modality,
        )
        yield StreamEvent(
            event=StreamEventType.METRICS,
            data={"latency_ms": response.latency_ms, "cache_tier": CacheTier.MISS},
        )
        yield StreamEvent(event=StreamEventType.DONE, data={"request_id": request_id})

    async def _origin_stream(
        self,
        request: RagQueryRequest,
        request_id: str,
        model_id: str,
    ) -> AsyncIterator[StreamEvent]:
        if request.modality == "text":
            try:
                async for event in self._retrieve_and_generate_stream(request, model_id):
                    yield event
                return
            except Exception as exc:
                # Image-only multimodal KBs and some Nova MME configurations reject RAG.
                logger.warning(
                    "retrieve_and_generate_unavailable_falling_back",
                    error=str(exc)[:300],
                    request_id=request_id,
                )
        async for event in self._retrieve_then_converse_stream(request, model_id):
            yield event

    async def _retrieve_and_generate_stream(
        self,
        request: RagQueryRequest,
        model_id: str,
    ) -> AsyncIterator[StreamEvent]:
        """Native Bedrock KB streaming generation for text queries."""

        async def _start() -> Any:
            return await self._clients.bedrock_agent_runtime.retrieve_and_generate_stream(
                input={"text": request.text or ""},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self._settings.knowledge_base_id,
                        "modelArn": model_id,
                        "retrievalConfiguration": self._retrieval_config(request),
                        "generationConfiguration": {
                            "inferenceConfig": {
                                "textInferenceConfig": {
                                    "maxTokens": self._settings.max_generation_tokens,
                                    "temperature": self._settings.generation_temperature,
                                }
                            },
                            "promptTemplate": {
                                "textPromptTemplate": (
                                    SYSTEM_PROMPT
                                    + "\n\n$search_results$\n\nUser question:\n$query$"
                                )
                            },
                        },
                    },
                },
            )

        response = await self._agent_breaker.call(_start)
        async for event in response["stream"]:
            if "output" in event and "text" in event["output"]:
                yield StreamEvent(event=StreamEventType.TOKEN, data={"text": event["output"]["text"]})
            if "citation" in event:
                yield StreamEvent(
                    event=StreamEventType.CITATION,
                    data=_citation_from_kb(event["citation"]),
                )
            if "error" in event:
                raise RuntimeError(json.dumps(event["error"]))

    async def _retrieve_then_converse_stream(
        self,
        request: RagQueryRequest,
        model_id: str,
    ) -> AsyncIterator[StreamEvent]:
        """Crossmodal path: Retrieve (text or image) then Claude 5 ConverseStream."""

        retrieval_query = self._retrieval_query(request)

        async def _retrieve() -> dict[str, Any]:
            return await self._clients.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self._settings.knowledge_base_id,
                retrievalQuery=retrieval_query,
                retrievalConfiguration=self._retrieval_config(request),
            )

        retrieved = await self._agent_breaker.call(_retrieve)
        results = retrieved.get("retrievalResults") or []
        context_blocks: list[str] = []
        for result in results:
            content = result.get("content") or {}
            text = content.get("text") or content.get("byteContent") or ""
            location = (result.get("location") or {}).get("s3Location", {}).get("uri")
            score = result.get("score")
            metadata = result.get("metadata") or {}
            context_blocks.append(f"[score={score}] {location or ''}\n{text}")
            yield StreamEvent(
                event=StreamEventType.CITATION,
                data=Citation(
                    uri=location,
                    score=score,
                    metadata=metadata,
                    snippet=str(text)[:500] if text else None,
                    content_type=_guess_content_type(metadata, location),
                ).model_dump(),
            )

        user_content: list[dict[str, Any]] = [
            {
                "text": (
                    f"{SYSTEM_PROMPT}\n\nRetrieved context:\n"
                    + "\n---\n".join(context_blocks)
                    + "\n\nUser query:\n"
                    + (request.text or "Identify and describe the visually similar catalog items.")
                )
            }
        ]
        if request.image and request.image.base64_data:
            user_content.append(
                {
                    "image": {
                        "format": request.image.format.value,
                        "source": {"bytes": request.image.base64_data},
                    }
                }
            )

        async def _converse() -> Any:
            return await self._clients.bedrock_runtime.converse_stream(
                modelId=model_id,
                messages=[{"role": "user", "content": user_content}],
                inferenceConfig={
                    "maxTokens": self._settings.max_generation_tokens,
                    "temperature": self._settings.generation_temperature,
                },
            )

        stream = await self._runtime_breaker.call(_converse)
        async for event in stream.get("stream", []):
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                yield StreamEvent(event=StreamEventType.TOKEN, data={"text": delta["text"]})

    def _retrieval_config(self, request: RagQueryRequest) -> dict[str, Any]:
        filters: list[dict[str, Any]] = [
            {"equals": {"key": "tenant_id", "value": request.tenant.tenant_id}}
        ]
        if request.tenant.category:
            filters.append({"equals": {"key": "category", "value": request.tenant.category}})
        for key, value in request.metadata_filter.items():
            filters.append({"equals": {"key": key, "value": value}})
        vector_search: dict[str, Any] = {
            "numberOfResults": request.top_k or self._settings.retrieval_top_k,
        }
        if len(filters) == 1:
            vector_search["filter"] = filters[0]
        else:
            vector_search["filter"] = {"andAll": filters}
        return {"vectorSearchConfiguration": vector_search}

    @staticmethod
    def _retrieval_query(request: RagQueryRequest) -> dict[str, Any]:
        if request.text:
            return {"text": request.text}
        assert request.image is not None
        if request.image.base64_data:
            return {
                "multimodalInputList": [
                    {
                        "content": {"byteContent": request.image.base64_data},
                        "modality": "IMAGE",
                    }
                ]
            }
        return {
            "multimodalInputList": [
                {
                    "content": {"s3Uri": request.image.s3_uri},
                    "modality": "IMAGE",
                }
            ]
        }

    async def _store_cache(
        self,
        request: RagQueryRequest,
        stats: Any,
        response: RagQueryResponse,
    ) -> None:
        if not response.answer:
            return
        try:
            await self._cache.store(
                request,
                stats or stats_miss(),
                CachedAnswer(
                    answer=response.answer,
                    citations=[c.model_dump() for c in response.citations],
                    model_id=response.model_id,
                    session_id=response.session_id,
                ),
            )
        except Exception:
            logger.warning("cache_store_failed", exc_info=True)


def stats_miss() -> Any:
    from api.schemas import CacheStats

    return CacheStats(tier=CacheTier.MISS)


def _citation_from_kb(payload: dict[str, Any]) -> dict[str, Any]:
    refs = payload.get("retrievedReferences") or payload.get("generatedResponsePart") or {}
    if isinstance(refs, list) and refs:
        ref = refs[0]
        loc = (ref.get("location") or {}).get("s3Location", {}).get("uri")
        return Citation(
            uri=loc,
            metadata=ref.get("metadata") or {},
            snippet=(ref.get("content") or {}).get("text"),
        ).model_dump()
    return Citation().model_dump()


def _guess_content_type(metadata: dict[str, Any], uri: str | None) -> ContentType | None:
    raw = str(metadata.get("content_type") or "")
    for ct in ContentType:
        if ct.value == raw:
            return ct
    if not uri:
        return None
    lowered = uri.lower()
    if lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return ContentType.IMAGE
    if lowered.endswith((".mp4", ".mov", ".mkv")):
        return ContentType.VIDEO
    if lowered.endswith((".mp3", ".wav", ".flac")):
        return ContentType.AUDIO
    return ContentType.DOCUMENT
