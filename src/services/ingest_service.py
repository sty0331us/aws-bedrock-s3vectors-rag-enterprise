"""S3 sidecar metadata + Bedrock StartIngestionJob orchestration.

Bedrock Knowledge Bases pick up `{object}.metadata.json` next to source objects.
We write tenant isolation fields there (`tenant_id`, `content_type`, `category`,
`created_at`) so Retrieve filters map onto S3 Vectors filterable metadata.

The worker is invoked from SQS FIFO (MessageGroupId = tenant_id) so a single
tenant's objects are processed in order while tenants interleave.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import structlog

from api.schemas import IngestRequest, IngestResponse
from config import Settings
from services.aws_clients import AwsClients
from services.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

RESERVED_METADATA_KEYS = ("tenant_id", "content_type", "category", "created_at")


class IngestService:
    def __init__(
        self,
        settings: Settings,
        clients: AwsClients,
        breaker: CircuitBreaker,
    ) -> None:
        self._settings = settings
        self._clients = clients
        self._breaker = breaker

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        bucket, key = parse_s3_uri(request.s3_uri)
        metadata = build_bedrock_metadata(request)
        sidecar_key = f"{key}.metadata.json"
        await self._put_sidecar(bucket, sidecar_key, metadata)

        job_id: str | None = None
        status: str = "queued"
        if request.start_ingestion_job:
            job_id = await self.start_ingestion_job()
            status = "started"

        return IngestResponse(
            ingestion_job_id=job_id,
            data_source_id=self._settings.data_source_id,
            knowledge_base_id=self._settings.knowledge_base_id,
            s3_uri=request.s3_uri,
            metadata_object_uri=f"s3://{bucket}/{sidecar_key}",
            status=status,  # type: ignore[arg-type]
        )

    async def handle_s3_event(self, record: dict[str, Any]) -> str | None:
        """Lambda entry for EventBridge/SQS-wrapped Object Created events."""

        bucket = record.get("s3", {}).get("bucket", {}).get("name") or record.get("bucket")
        key = record.get("s3", {}).get("object", {}).get("key") or record.get("key")
        if not bucket or not key:
            raise ValueError(f"unrecognized S3 event shape: {list(record.keys())}")
        if key.endswith(".metadata.json"):
            logger.info("skip_sidecar", key=key)
            return None
        logger.info("ingest_from_event", bucket=bucket, key=key)
        return await self.start_ingestion_job()

    async def start_ingestion_job(self) -> str:
        if not self._settings.knowledge_base_id or not self._settings.data_source_id:
            raise RuntimeError("KNOWLEDGE_BASE_ID and DATA_SOURCE_ID must be configured")

        async def _start() -> dict[str, Any]:
            client = self._clients.bedrock_agent
            existing = await client.list_ingestion_jobs(
                knowledgeBaseId=self._settings.knowledge_base_id,
                dataSourceId=self._settings.data_source_id,
                filters=[{"attribute": "STATUS", "operator": "EQ", "values": ["STARTING", "IN_PROGRESS"]}],
                maxResults=1,
            )
            jobs = existing.get("ingestionJobSummaries") or []
            if jobs:
                job_id = jobs[0]["ingestionJobId"]
                logger.info("ingestion_job_already_running", job_id=job_id)
                return {"ingestionJob": {"ingestionJobId": job_id}}
            return await client.start_ingestion_job(
                knowledgeBaseId=self._settings.knowledge_base_id,
                dataSourceId=self._settings.data_source_id,
                description=f"event-driven sync {datetime.now(UTC).isoformat()}",
            )

        response = await self._breaker.call(_start)
        job_id = response["ingestionJob"]["ingestionJobId"]
        logger.info("ingestion_job_started", job_id=job_id)
        return job_id

    async def _put_sidecar(self, bucket: str, key: str, metadata: dict[str, Any]) -> None:
        async def _put() -> None:
            await self._clients.s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(metadata).encode("utf-8"),
                ContentType="application/json",
            )

        await self._breaker.call(_put)


def build_bedrock_metadata(request: IngestRequest) -> dict[str, Any]:
    """Bedrock S3 data-source sidecar schema.

    Filterable attributes become S3 Vectors metadata used at query time.
    """

    attributes: dict[str, dict[str, str]] = {
        "tenant_id": {"value": {"type": "STRING", "stringValue": request.tenant.tenant_id}},
        "content_type": {"value": {"type": "STRING", "stringValue": request.content_type.value}},
        "created_at": {
            "value": {"type": "STRING", "stringValue": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
        },
    }
    category = request.category or request.tenant.category
    if category:
        attributes["category"] = {"value": {"type": "STRING", "stringValue": category}}
    for key, value in request.metadata.items():
        if key in RESERVED_METADATA_KEYS:
            continue
        attributes[key] = {"value": {"type": "STRING", "stringValue": value}}
    return {"metadataAttributes": attributes}


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")
