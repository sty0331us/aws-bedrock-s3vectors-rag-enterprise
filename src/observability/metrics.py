"""CloudWatch custom metrics for cache effectiveness and estimated cost reduction.

Metric emission is fire-and-forget: a metrics failure must never fail a user query.
At 100M MAU we rely on EMF (Embedded Metric Format) so the application does not
open a PutMetricData hot path; CloudWatch ingests the JSON log line automatically.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import structlog

from config import Settings

logger = structlog.get_logger(__name__)


class MetricsEmitter:
    """Emits EMF documents to stdout (captured by the CloudWatch Logs agent / FireLens)."""

    def __init__(self, settings: Settings) -> None:
        self._namespace = settings.cloudwatch_namespace
        self._environment = settings.environment
        self._llm_cost_per_1k = settings.estimated_llm_cost_per_1k_tokens_usd
        self._kb_query_cost = settings.estimated_kb_query_cost_usd

    def record_cache_lookup(
        self,
        *,
        tenant_id: str,
        tier: str,
        hit: bool,
        saved_latency_ms: float = 0.0,
        estimated_tokens_saved: int = 0,
    ) -> None:
        cost_saved = (estimated_tokens_saved / 1000.0) * self._llm_cost_per_1k
        if hit:
            cost_saved += self._kb_query_cost
        self._emit(
            metrics={
                "CacheHit": 1 if hit else 0,
                "CacheMiss": 0 if hit else 1,
                "SavedLatencyMs": saved_latency_ms,
                "EstimatedCostSavedUsd": round(cost_saved, 6),
            },
            dimensions={"Tier": tier, "Environment": self._environment},
            properties={"TenantId": tenant_id},
        )

    def record_rag_latency(self, *, latency_ms: float, model_id: str, modality: str) -> None:
        self._emit(
            metrics={"RagLatencyMs": latency_ms},
            dimensions={
                "ModelId": model_id.split("/")[-1][:64],
                "Modality": modality,
                "Environment": self._environment,
            },
        )

    def record_throttle(self, *, operation: str) -> None:
        self._emit(
            metrics={"BedrockThrottle": 1},
            dimensions={"Operation": operation, "Environment": self._environment},
        )

    def _emit(
        self,
        *,
        metrics: dict[str, float | int],
        dimensions: dict[str, str],
        properties: dict[str, Any] | None = None,
    ) -> None:
        try:
            emf: dict[str, Any] = {
                "_aws": {
                    "Timestamp": int(time.time() * 1000),
                    "CloudWatchMetrics": [
                        {
                            "Namespace": self._namespace,
                            "Dimensions": [list(dimensions.keys())],
                            "Metrics": [{"Name": name, "Unit": self._unit(name)} for name in metrics],
                        }
                    ],
                }
            }
            emf.update(dimensions)
            emf.update(metrics)
            if properties:
                emf.update(properties)
            sys.stdout.write(json.dumps(emf, separators=(",", ":")) + "\n")
            sys.stdout.flush()
        except Exception:
            logger.warning("metrics_emit_failed", exc_info=True)

    @staticmethod
    def _unit(name: str) -> str:
        if name.endswith("Ms"):
            return "Milliseconds"
        if name.endswith("Usd"):
            return "None"
        return "Count"
