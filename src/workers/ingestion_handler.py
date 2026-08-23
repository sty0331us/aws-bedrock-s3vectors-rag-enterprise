"""SQS-triggered ingestion worker.

Event path
    S3 Object Created → EventBridge → SQS FIFO (+ DLQ) → this handler
    → Bedrock Agent `StartIngestionJob`.

FIFO MessageGroupId is the tenant prefix of the object key
(`s3://bucket/{tenant_id}/...`) so one tenant cannot head-of-line block others.
Partial batch failure is reported back to Lambda so only poison records retry.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import structlog
from aiobotocore.session import get_session

from config import get_settings
from services.aws_clients import AwsClients
from services.circuit_breaker import CircuitBreaker
from services.ingest_service import IngestService

logger = structlog.get_logger(__name__)


def _breaker() -> CircuitBreaker:
    settings = get_settings()
    return CircuitBreaker(
        name="ingestion-worker",
        failure_threshold=settings.circuit_failure_threshold,
        recovery_seconds=settings.circuit_recovery_seconds,
        max_attempts=settings.bedrock_max_attempts,
        base_backoff=settings.bedrock_base_backoff_seconds,
        max_backoff=settings.bedrock_max_backoff_seconds,
    )


async def _handle_async(event: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    aws = await AwsClients.connect(get_session(), settings.aws_region)
    service = IngestService(settings, aws, _breaker())
    failures: list[dict[str, str]] = []
    try:
        for record in event.get("Records", []):
            message_id = record.get("messageId", "unknown")
            try:
                body = _parse_body(record.get("body", "{}"))
                s3_records = _extract_s3_records(body)
                if not s3_records:
                    await service.start_ingestion_job()
                    continue
                for s3_record in s3_records:
                    await service.handle_s3_event(s3_record)
            except Exception:
                logger.exception("record_failed", message_id=message_id)
                failures.append({"itemIdentifier": message_id})
    finally:
        await aws.close()
    return {"batchItemFailures": failures}


def _parse_body(body: str) -> dict[str, Any]:
    parsed = json.loads(body) if isinstance(body, str) else body
    # SNS/EventBridge wrapping.
    if isinstance(parsed, dict) and "Message" in parsed:
        return json.loads(parsed["Message"])
    return parsed


def _extract_s3_records(body: dict[str, Any]) -> list[dict[str, Any]]:
    if "Records" in body:
        return [r for r in body["Records"] if "s3" in r]
    if body.get("source") == "aws.s3" or "detail" in body:
        detail = body.get("detail") or body
        bucket = (detail.get("bucket") or {}).get("name")
        obj = detail.get("object") or {}
        if bucket and obj.get("key"):
            return [{"s3": {"bucket": {"name": bucket}, "object": obj}}]
    return []


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ARG001
    """Lambda entrypoint (sync wrapper around the async worker)."""

    logger.info("worker_invoke", records=len(event.get("Records", [])), request_id=getattr(context, "aws_request_id", None))
    return asyncio.run(_handle_async(event))


if __name__ == "__main__":
    sample = {
        "Records": [
            {
                "messageId": "local",
                "body": json.dumps(
                    {
                        "detail": {
                            "bucket": {"name": os.environ.get("SOURCE_BUCKET", "demo")},
                            "object": {"key": "tenant-a/docs/sample.pdf"},
                        }
                    }
                ),
            }
        ]
    }
    print(handler(sample, None))
