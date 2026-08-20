"""Two-tier Redis cache for multimodal RAG queries.

L1 Exact Hash Cache
    Canonical SHA-256 of (tenant_id, modality, payload). Sub-10ms GET on
    ElastiCache Serverless. Identical questions (including identical images)
    never reach Bedrock.

L2 Semantic Cache
    Nova Multimodal Embeddings of the query, truncated to a Matryoshka prefix
    (default 384-d), compared with cosine similarity against a tenant-scoped
    candidate set. A hit short-circuits Knowledge Base retrieve + Claude
    invocation when similarity >= configured threshold.

ElastiCache Serverless does not guarantee the RediSearch VECTOR module on every
engine version, so the implementation keeps a bounded Redis ZSET of recent
cache ids per tenant and scores cosine similarity in-process. The candidate
window is small (default 64) and already L2-normalized, so scoring is well
under 1ms — negligible next to a Bedrock round-trip.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import orjson
import redis.asyncio as redis
import structlog

from api.schemas import CacheStats, CacheTier, RagQueryRequest, RagQueryResponse
from config import Settings
from observability.metrics import MetricsEmitter
from services.embedding_service import NovaEmbeddingService, cosine_similarity

logger = structlog.get_logger(__name__)

L1_PREFIX = "mmrag:l1"
L2_PREFIX = "mmrag:l2"
L2_INDEX_PREFIX = "mmrag:l2idx"


@dataclass(slots=True)
class CachedAnswer:
    answer: str
    citations: list[dict[str, Any]]
    model_id: str
    session_id: str | None
    embedding: list[float] | None = None


class CacheService:
    """Redis-backed L1/L2 cache with CloudWatch hit-ratio instrumentation."""

    def __init__(
        self,
        settings: Settings,
        redis_client: redis.Redis,
        embeddings: NovaEmbeddingService,
        metrics: MetricsEmitter,
    ) -> None:
        self._settings = settings
        self._redis = redis_client
        self._embeddings = embeddings
        self._metrics = metrics

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except redis.RedisError:
            logger.warning("redis_ping_failed", exc_info=True)
            return False

    @staticmethod
    def exact_hash(request: RagQueryRequest) -> str:
        """Stable identity for L1. Images hash the Base64 or S3 URI, not pixels."""

        payload = {
            "tenant": request.tenant.tenant_id,
            "modality": request.modality,
            "text": request.text,
            "image": request.image.model_dump() if request.image else None,
            "top_k": request.top_k,
            "filter": request.metadata_filter,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def lookup(self, request: RagQueryRequest) -> tuple[RagQueryResponse | None, CacheStats]:
        """L1 then L2. Returns (response, stats). Response is None on a full miss."""

        started = time.perf_counter()
        content_hash = self.exact_hash(request)

        l1 = await self._get_l1(request.tenant.tenant_id, content_hash)
        if l1 is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            stats = CacheStats(
                tier=CacheTier.L1_EXACT,
                similarity=1.0,
                saved_latency_ms=max(0.0, 800.0 - elapsed_ms),
                estimated_cost_saved_usd=self._settings.estimated_kb_query_cost_usd
                + self._settings.estimated_llm_cost_per_1k_tokens_usd,
            )
            self._metrics.record_cache_lookup(
                tenant_id=request.tenant.tenant_id,
                tier="L1",
                hit=True,
                saved_latency_ms=stats.saved_latency_ms or 0.0,
                estimated_tokens_saved=800,
            )
            return self._to_response(l1, stats, request), stats

        embedding: list[float] | None = None
        try:
            embedding = await self._embed_request(request)
            l2, similarity = await self._get_l2(request.tenant.tenant_id, embedding)
        except Exception:
            logger.warning("l2_lookup_failed", exc_info=True)
            l2, similarity = None, None

        if l2 is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            stats = CacheStats(
                tier=CacheTier.L2_SEMANTIC,
                similarity=similarity,
                saved_latency_ms=max(0.0, 700.0 - elapsed_ms),
                estimated_cost_saved_usd=self._settings.estimated_llm_cost_per_1k_tokens_usd,
            )
            self._metrics.record_cache_lookup(
                tenant_id=request.tenant.tenant_id,
                tier="L2",
                hit=True,
                saved_latency_ms=stats.saved_latency_ms or 0.0,
                estimated_tokens_saved=700,
            )
            # Promote to L1 so the next identical hash is a sub-10ms hit.
            await self._put_l1(request.tenant.tenant_id, content_hash, l2)
            return self._to_response(l2, stats, request), stats

        self._metrics.record_cache_lookup(
            tenant_id=request.tenant.tenant_id,
            tier="MISS",
            hit=False,
        )
        miss = CacheStats(tier=CacheTier.MISS)
        # Stash embedding on the stats object via a private attr used by store().
        miss.__dict__["_embedding"] = embedding
        miss.__dict__["_content_hash"] = content_hash
        return None, miss

    async def store(self, request: RagQueryRequest, stats: CacheStats, cached: CachedAnswer) -> None:
        """Write-through to L1 and L2 after a successful origin generation."""

        content_hash = stats.__dict__.get("_content_hash") or self.exact_hash(request)
        embedding = cached.embedding or stats.__dict__.get("_embedding")
        if embedding is None:
            try:
                embedding = await self._embed_request(request)
            except Exception:
                logger.warning("cache_store_embed_failed", exc_info=True)
                embedding = None
        cached.embedding = embedding
        await self._put_l1(request.tenant.tenant_id, content_hash, cached)
        if embedding:
            await self._put_l2(request.tenant.tenant_id, cached)

    async def _embed_request(self, request: RagQueryRequest) -> list[float]:
        if request.text:
            return await self._embeddings.embed_for_cache(text=request.text)
        assert request.image is not None
        return await self._embeddings.embed_for_cache(
            image_b64=request.image.base64_data,
            image_s3_uri=request.image.s3_uri,
            image_format=request.image.format.value,
        )

    def _l1_key(self, tenant_id: str, content_hash: str) -> str:
        return f"{L1_PREFIX}:{tenant_id}:{content_hash}"

    def _l2_doc_key(self, tenant_id: str, cache_id: str) -> str:
        return f"{L2_PREFIX}:{tenant_id}:{cache_id}"

    def _l2_index_key(self, tenant_id: str) -> str:
        return f"{L2_INDEX_PREFIX}:{tenant_id}"

    async def _get_l1(self, tenant_id: str, content_hash: str) -> CachedAnswer | None:
        raw = await self._redis.get(self._l1_key(tenant_id, content_hash))
        if not raw:
            return None
        return self._loads(raw)

    async def _put_l1(self, tenant_id: str, content_hash: str, cached: CachedAnswer) -> None:
        await self._redis.set(
            self._l1_key(tenant_id, content_hash),
            self._dumps(cached),
            ex=self._settings.l1_ttl_seconds,
        )

    async def _get_l2(
        self,
        tenant_id: str,
        query_embedding: list[float],
    ) -> tuple[CachedAnswer | None, float | None]:
        index_key = self._l2_index_key(tenant_id)
        # Newest-first bounded window.
        cache_ids = await self._redis.zrevrange(
            index_key,
            0,
            self._settings.semantic_cache_candidate_limit - 1,
        )
        if not cache_ids:
            return None, None

        keys = [self._l2_doc_key(tenant_id, cid.decode() if isinstance(cid, bytes) else cid) for cid in cache_ids]
        blobs = await self._redis.mget(keys)
        best: CachedAnswer | None = None
        best_sim = -1.0
        for blob in blobs:
            if not blob:
                continue
            candidate = self._loads(blob)
            if not candidate.embedding:
                continue
            sim = cosine_similarity(query_embedding, candidate.embedding)
            if sim > best_sim:
                best_sim = sim
                best = candidate
        if best is None or best_sim < self._settings.semantic_cache_threshold:
            return None, best_sim if best is not None else None
        logger.info("l2_hit", tenant_id=tenant_id, similarity=round(best_sim, 4))
        return best, best_sim

    async def _put_l2(self, tenant_id: str, cached: CachedAnswer) -> None:
        cache_id = uuid.uuid4().hex
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(
            self._l2_doc_key(tenant_id, cache_id),
            self._dumps(cached),
            ex=self._settings.l2_ttl_seconds,
        )
        index_key = self._l2_index_key(tenant_id)
        pipe.zadd(index_key, {cache_id: time.time()})
        pipe.zremrangebyrank(index_key, 0, -self._settings.semantic_cache_candidate_limit - 1)
        pipe.expire(index_key, self._settings.l2_ttl_seconds)
        await pipe.execute()

    @staticmethod
    def _dumps(cached: CachedAnswer) -> bytes:
        return orjson.dumps(
            {
                "answer": cached.answer,
                "citations": cached.citations,
                "model_id": cached.model_id,
                "session_id": cached.session_id,
                "embedding": cached.embedding,
            }
        )

    @staticmethod
    def _loads(raw: bytes | str) -> CachedAnswer:
        data = orjson.loads(raw)
        return CachedAnswer(
            answer=data["answer"],
            citations=data.get("citations") or [],
            model_id=data["model_id"],
            session_id=data.get("session_id"),
            embedding=data.get("embedding"),
        )

    @staticmethod
    def _to_response(
        cached: CachedAnswer,
        stats: CacheStats,
        request: RagQueryRequest,
    ) -> RagQueryResponse:
        from api.schemas import Citation

        return RagQueryResponse(
            answer=cached.answer,
            citations=[Citation.model_validate(c) for c in cached.citations],
            model_id=cached.model_id,
            session_id=cached.session_id or request.session_id,
            cache=stats,
            request_id="cache",
            latency_ms=stats.saved_latency_ms or 0.0,
        )
