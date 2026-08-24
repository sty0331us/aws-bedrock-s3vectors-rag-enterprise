"""FastAPI composition root. Instantiates long-lived AWS and Redis clients."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import redis.asyncio as redis
import structlog
from aiobotocore.session import AioSession, get_session
from fastapi import FastAPI

from config import Settings, get_settings
from observability.metrics import MetricsEmitter
from services.aws_clients import AwsClients
from services.cache_service import CacheService
from services.circuit_breaker import CircuitBreaker
from services.embedding_service import NovaEmbeddingService
from services.ingest_service import IngestService
from services.model_router import ModelRouter
from services.rag_service import RagService

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    redis: redis.Redis
    session: AioSession
    aws: AwsClients
    rag: RagService
    ingest: IngestService
    cache: CacheService


def _breaker(name: str, settings: Settings, metrics: MetricsEmitter) -> CircuitBreaker:
    return CircuitBreaker(
        name=name,
        failure_threshold=settings.circuit_failure_threshold,
        recovery_seconds=settings.circuit_recovery_seconds,
        max_attempts=settings.bedrock_max_attempts,
        base_backoff=settings.bedrock_base_backoff_seconds,
        max_backoff=settings.bedrock_max_backoff_seconds,
        on_throttle=lambda: metrics.record_throttle(operation=name),
    )


async def build_container(settings: Settings) -> AppContainer:
    session = get_session()
    metrics = MetricsEmitter(settings)
    redis_client = redis.Redis.from_url(
        settings.redis_url,
        socket_connect_timeout=settings.redis_socket_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
        decode_responses=False,
        health_check_interval=15,
    )
    embed_breaker = _breaker("bedrock-runtime-embeddings", settings, metrics)
    agent_breaker = _breaker("bedrock-agent-runtime", settings, metrics)
    runtime_breaker = _breaker("bedrock-runtime-converse", settings, metrics)
    ingest_breaker = _breaker("bedrock-agent-ingest", settings, metrics)

    aws = await AwsClients.connect(session, settings.aws_region)
    embeddings = NovaEmbeddingService(settings, aws, embed_breaker)
    cache = CacheService(settings, redis_client, embeddings, metrics)
    rag = RagService(
        settings=settings,
        clients=aws,
        cache=cache,
        router=ModelRouter(settings),
        agent_breaker=agent_breaker,
        runtime_breaker=runtime_breaker,
        metrics=metrics,
    )
    ingest = IngestService(settings, aws, ingest_breaker)
    return AppContainer(
        settings=settings,
        redis=redis_client,
        session=session,
        aws=aws,
        rag=rag,
        ingest=ingest,
        cache=cache,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.enable_xray:
        from aws_xray_sdk.core import patch_all, xray_recorder

        xray_recorder.configure(service=settings.service_name, context_missing="LOG_ERROR")
        patch_all()

    container = await build_container(settings)
    app.state.container = container
    logger.info(
        "api_started",
        environment=settings.environment,
        kb_configured=bool(settings.knowledge_base_id),
    )
    try:
        yield
    finally:
        await container.aws.close()
        await container.redis.aclose()


def get_container(app: FastAPI) -> AppContainer:
    return app.state.container
