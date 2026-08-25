"""Async cache lookup tests with a fake Redis."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.schemas import CacheTier, RagQueryRequest, TenantContext
from config import Settings
from observability.metrics import MetricsEmitter
from services.cache_service import CacheService, CachedAnswer


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[bytes, bytes] = {}
        self.zsets: dict[bytes, dict[str, float]] = {}

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> bytes | None:
        return self.kv.get(key.encode() if isinstance(key, str) else key)

    async def set(self, key: str, value: bytes, ex: int | None = None) -> None:  # noqa: ARG002
        self.kv[key.encode() if isinstance(key, str) else key] = value

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        return [await self.get(k) for k in keys]

    async def zrevrange(self, key: str, start: int, end: int) -> list[bytes]:
        z = self.zsets.get(key.encode() if isinstance(key, str) else key, {})
        ordered = sorted(z.items(), key=lambda kv: kv[1], reverse=True)
        sliced = ordered[start : end + 1 if end >= 0 else None]
        return [k.encode() if isinstance(k, str) else k for k, _ in sliced]

    def pipeline(self, transaction: bool = True) -> Any:  # noqa: ARG002
        pipe = MagicMock()
        pipe.set = lambda *a, **k: None
        pipe.zadd = lambda key, mapping: self.zsets.setdefault(
            key.encode() if isinstance(key, str) else key, {}
        ).update(mapping)
        pipe.zremrangebyrank = lambda *a, **k: None
        pipe.expire = lambda *a, **k: None
        pipe.execute = AsyncMock(return_value=[])
        return pipe


@pytest.mark.asyncio
async def test_l1_hit_skips_embeddings() -> None:
    settings = Settings(knowledge_base_id="kb", cache_embedding_dimension=384, embedding_dimension=1024)
    embeddings = MagicMock()
    embeddings.embed_for_cache = AsyncMock(side_effect=AssertionError("L1 must not embed"))
    metrics = MetricsEmitter(settings)
    cache = CacheService(settings, FakeRedis(), embeddings, metrics)  # type: ignore[arg-type]
    request = RagQueryRequest(tenant=TenantContext(tenant_id="t1"), text="return policy")
    cached = CachedAnswer(answer="30 days", citations=[], model_id="claude", session_id=None)
    await cache._put_l1("t1", cache.exact_hash(request), cached)

    response, stats = await cache.lookup(request)
    assert response is not None
    assert response.answer == "30 days"
    assert stats.tier is CacheTier.L1_EXACT
    embeddings.embed_for_cache.assert_not_called()
