"""Unit tests for Matryoshka truncation, hashing, backoff, and metadata sidecars."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from api.schemas import MediaPayload, RagQueryRequest, TenantContext
from services.circuit_breaker import full_jitter
from services.embedding_service import cosine_similarity, truncate_matryoshka
from services.ingest_service import build_bedrock_metadata, parse_s3_uri
from services.cache_service import CacheService
from api.schemas import ContentType, IngestRequest


def test_matryoshka_truncation_renormalizes() -> None:
    full = [0.5] * 1024
    truncated = truncate_matryoshka(full, 384)
    assert len(truncated) == 384
    norm = math.sqrt(sum(x * x for x in truncated))
    assert abs(norm - 1.0) < 1e-5


def test_matryoshka_rejects_unknown_dimension() -> None:
    with pytest.raises(ValueError):
        truncate_matryoshka([0.1] * 1024, 512)


def test_cosine_identical_is_one() -> None:
    vec = [0.1, 0.2, 0.3, 0.4]
    assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-5


def test_full_jitter_is_bounded() -> None:
    for attempt in range(6):
        delay = full_jitter(attempt, base=0.25, cap=8.0)
        assert 0.0 <= delay <= 8.0


def test_query_rejects_mixed_modalities() -> None:
    with pytest.raises(ValidationError):
        RagQueryRequest(
            tenant=TenantContext(tenant_id="acme"),
            text="find similar shoes",
            image=MediaPayload(base64_data="aGVsbG8gd29ybGQ="),
        )


def test_query_requires_one_modality() -> None:
    with pytest.raises(ValidationError):
        RagQueryRequest(tenant=TenantContext(tenant_id="acme"))


def test_l1_hash_is_stable() -> None:
    req = RagQueryRequest(tenant=TenantContext(tenant_id="acme"), text="warranty policy")
    # Hash helper is instance-bound but does not need Redis.
    h1 = CacheService.exact_hash(req)
    h2 = CacheService.exact_hash(req)
    assert h1 == h2
    assert len(h1) == 64


def test_metadata_sidecar_contains_tenant_isolation_fields() -> None:
    request = IngestRequest(
        tenant=TenantContext(tenant_id="acme", category="footwear"),
        s3_uri="s3://bucket/acme/catalog/shoe.png",
        content_type=ContentType.IMAGE,
        metadata={"brand": "nova"},
    )
    sidecar = build_bedrock_metadata(request)
    attrs = sidecar["metadataAttributes"]
    assert attrs["tenant_id"]["value"]["stringValue"] == "acme"
    assert attrs["content_type"]["value"]["stringValue"] == "image"
    assert attrs["category"]["value"]["stringValue"] == "footwear"
    assert attrs["brand"]["value"]["stringValue"] == "nova"


def test_parse_s3_uri() -> None:
    bucket, key = parse_s3_uri("s3://my-bucket/path/to/object.pdf")
    assert bucket == "my-bucket"
    assert key == "path/to/object.pdf"
