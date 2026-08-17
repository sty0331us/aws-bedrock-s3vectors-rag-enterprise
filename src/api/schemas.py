"""Pydantic v2 request/response contracts for the multimodal RAG API.

The schemas are intentionally explicit about multimodal payloads: clients may
send a text prompt, inline Base64 media, an S3 URI, or a combination. Image +
text in a single Knowledge Base Retrieve call is not yet supported by Bedrock;
the service rejects that combination at validation time rather than failing
downstream with a 4xx from the Agent Runtime API.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class ContentType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class CacheTier(StrEnum):
    L1_EXACT = "l1_exact"
    L2_SEMANTIC = "l2_semantic"
    MISS = "miss"


class ImageFormat(StrEnum):
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"


class TenantContext(BaseModel):
    """Multi-tenant isolation key propagated into metadata filters and cache keys."""

    tenant_id: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-:]+$")]
    category: str | None = Field(default=None, max_length=128)


class MediaPayload(BaseModel):
    """Exactly one of `base64_data` or `s3_uri` must be provided."""

    mime_type: str = Field(default="image/jpeg", examples=["image/jpeg", "image/png"])
    format: ImageFormat = ImageFormat.JPEG
    base64_data: str | None = Field(default=None, min_length=8, description="Raw Base64, no data: URI prefix")
    s3_uri: str | None = Field(default=None, pattern=r"^s3://[a-z0-9.\-]+/.+")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        if bool(self.base64_data) == bool(self.s3_uri):
            raise ValueError("provide exactly one of base64_data or s3_uri")
        return self


class RagQueryRequest(BaseModel):
    """Unified crossmodal query. Text and image are mutually exclusive at retrieve time."""

    tenant: TenantContext
    text: str | None = Field(default=None, min_length=1, max_length=32_000)
    image: MediaPayload | None = None
    session_id: str | None = Field(default=None, max_length=256)
    top_k: int | None = Field(default=None, ge=1, le=100)
    metadata_filter: dict[str, str] = Field(default_factory=dict)
    bypass_cache: bool = False
    stream: bool = False

    @model_validator(mode="after")
    def _require_modality(self) -> Self:
        if bool(self.text) == bool(self.image):
            raise ValueError("provide exactly one of text or image (Bedrock does not accept mixed query modalities)")
        return self

    @property
    def modality(self) -> Literal["text", "image"]:
        return "image" if self.image is not None else "text"


class Citation(BaseModel):
    uri: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    snippet: str | None = None
    content_type: ContentType | None = None


class CacheStats(BaseModel):
    tier: CacheTier
    similarity: float | None = None
    saved_latency_ms: float | None = None
    estimated_cost_saved_usd: float | None = None


class RagQueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model_id: str
    session_id: str | None = None
    cache: CacheStats
    request_id: str
    latency_ms: float


class StreamEventType(StrEnum):
    SESSION = "session"
    CACHE = "cache"
    TOKEN = "token"
    CITATION = "citation"
    METRICS = "metrics"
    ERROR = "error"
    DONE = "done"


class StreamEvent(BaseModel):
    """Server-Sent Event payload. `event` maps to the SSE event name."""

    event: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Kick off ingestion for an object already in the source bucket, or register an upload."""

    tenant: TenantContext
    s3_uri: str = Field(pattern=r"^s3://[a-z0-9.\-]+/.+")
    content_type: ContentType = ContentType.DOCUMENT
    category: str | None = Field(default=None, max_length=128)
    metadata: dict[str, str] = Field(default_factory=dict)
    start_ingestion_job: bool = True

    @field_validator("metadata")
    @classmethod
    def _limit_metadata_keys(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 20:
            raise ValueError("metadata may contain at most 20 keys")
        return value


class IngestResponse(BaseModel):
    ingestion_job_id: str | None = None
    data_source_id: str
    knowledge_base_id: str
    s3_uri: str
    metadata_object_uri: str | None = None
    status: Literal["queued", "started", "accepted"] = "accepted"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    redis: bool
    knowledge_base_configured: bool
    timestamp: datetime
