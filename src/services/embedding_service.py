"""Amazon Nova Multimodal Embeddings client.

Uses `amazon.nova-2-multimodal-embeddings-v1:0` via Bedrock Runtime InvokeModel.
Embeddings for distinct modalities land in one vector space, which is what makes
crossmodal retrieval (text→image, image→image) possible.

Matryoshka Representation Learning stores the most salient signal in the leading
dimensions. We request the S3 Vectors index dimension (default 1024) from Nova and
truncate + L2-normalize a 384-d prefix for the Redis L2 semantic cache.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import structlog

from config import NOVA_MATRYOSHKA_DIMENSIONS, Settings
from services.aws_clients import AwsClients
from services.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

EmbeddingPurpose = Literal["GENERIC_INDEX", "GENERIC_RETRIEVAL", "CLASSIFICATION", "CLUSTERING"]


def truncate_matryoshka(embedding: list[float], dimension: int) -> list[float]:
    """Keep the leading `dimension` values and L2-normalize (MRL best practice)."""

    if dimension not in NOVA_MATRYOSHKA_DIMENSIONS:
        raise ValueError(f"unsupported Matryoshka dimension: {dimension}")
    if len(embedding) < dimension:
        raise ValueError(f"cannot expand embedding from {len(embedding)} to {dimension}")
    vec = np.asarray(embedding[:dimension], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.tolist()
    return (vec / norm).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity on already-normalized (or raw) vectors."""

    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class NovaEmbeddingService:
    """Async Nova embedding facade used by the semantic cache and optional direct search."""

    def __init__(
        self,
        settings: Settings,
        clients: AwsClients,
        breaker: CircuitBreaker,
    ) -> None:
        self._settings = settings
        self._clients = clients
        self._breaker = breaker
        self._model_id = settings.bedrock_embedding_model_id

    async def embed_text(
        self,
        text: str,
        *,
        dimension: int | None = None,
        purpose: EmbeddingPurpose = "GENERIC_RETRIEVAL",
    ) -> list[float]:
        body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": purpose,
                "embeddingDimension": dimension or self._settings.embedding_dimension,
                "text": {"truncationMode": "END", "value": text},
            },
        }
        return await self._invoke(body)

    async def embed_image(
        self,
        *,
        base64_data: str | None = None,
        s3_uri: str | None = None,
        image_format: str = "jpeg",
        dimension: int | None = None,
        purpose: EmbeddingPurpose = "GENERIC_RETRIEVAL",
    ) -> list[float]:
        image: dict[str, Any] = {"format": image_format}
        if base64_data:
            image["source"] = {"bytes": base64_data}
        elif s3_uri:
            image["source"] = {"s3Location": {"uri": s3_uri}}
        else:
            raise ValueError("embed_image requires base64_data or s3_uri")
        body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": purpose,
                "embeddingDimension": dimension or self._settings.embedding_dimension,
                "image": image,
            },
        }
        return await self._invoke(body)

    async def embed_for_cache(
        self,
        *,
        text: str | None = None,
        image_b64: str | None = None,
        image_s3_uri: str | None = None,
        image_format: str = "jpeg",
    ) -> list[float]:
        """Produce the truncated vector stored in Redis L2."""

        dim = self._settings.cache_embedding_dimension
        if text:
            full = await self.embed_text(text, dimension=self._settings.embedding_dimension)
        else:
            full = await self.embed_image(
                base64_data=image_b64,
                s3_uri=image_s3_uri,
                image_format=image_format,
                dimension=self._settings.embedding_dimension,
            )
        return truncate_matryoshka(full, dim)

    async def _invoke(self, body: dict[str, Any]) -> list[float]:
        async def _call() -> list[float]:
            response = await self._clients.bedrock_runtime.invoke_model(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            raw = await response["body"].read()
            parsed = json.loads(raw)
            embeddings = parsed.get("embeddings") or []
            if not embeddings:
                raise RuntimeError(f"Nova returned no embeddings: {parsed.keys()}")
            vector = embeddings[0]["embedding"]
            logger.debug("nova_embedding", dim=len(vector), type=embeddings[0].get("embeddingType"))
            return vector

        return await self._breaker.call(_call)
