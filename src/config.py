"""Runtime configuration for the multimodal RAG platform.

Values are sourced from environment variables so the same artifact can run
on ECS Fargate, Lambda, or a local developer workstation. CDK stacks emit
the production values as task/Lambda environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Matryoshka dimensions supported by amazon.nova-2-multimodal-embeddings-v1:0.
NOVA_MATRYOSHKA_DIMENSIONS: frozenset[int] = frozenset({256, 384, 1024, 3072})


class Settings(BaseSettings):
    """Strongly typed process configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    environment: Literal["dev", "staging", "prod"] = "dev"
    service_name: str = "mmrag-api"

    knowledge_base_id: str = Field(default="", alias="KNOWLEDGE_BASE_ID")
    data_source_id: str = Field(default="", alias="DATA_SOURCE_ID")
    bedrock_embedding_model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0"
    bedrock_prompt_router_arn: str = Field(default="", alias="BEDROCK_PROMPT_ROUTER_ARN")

    # Claude 5 Sonnet geo inference profile (deep reasoning / synthesis).
    claude_sonnet_model_id: str = "us.anthropic.claude-sonnet-5"
    # Fast / cost-efficient companion in the Claude 5 stack (Haiku 4.5 on Bedrock).
    claude_haiku_model_id: str = "us.anthropic.claude-haiku-4-5"

    # S3 Vectors index dimension. Must match infra/lib/constructs/vector-store.ts.
    embedding_dimension: int = 1024
    # L2 semantic cache uses a truncated Matryoshka prefix to cut Redis RAM.
    cache_embedding_dimension: int = 384
    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    semantic_cache_candidate_limit: int = Field(default=64, ge=8, le=512)
    l1_ttl_seconds: int = 86_400
    l2_ttl_seconds: int = 43_200

    redis_endpoint: str = "localhost"
    redis_port: int = 6379
    redis_tls: bool = False
    redis_auth_token: str = ""
    redis_socket_timeout_seconds: float = 1.5

    source_bucket: str = ""
    multimodal_bucket: str = ""
    ingest_queue_url: str = ""

    retrieval_top_k: int = Field(default=8, ge=1, le=100)
    max_generation_tokens: int = Field(default=2048, ge=64, le=8192)
    generation_temperature: float = Field(default=0.2, ge=0.0, le=1.0)

    bedrock_max_attempts: int = Field(default=5, ge=1, le=10)
    bedrock_base_backoff_seconds: float = 0.25
    bedrock_max_backoff_seconds: float = 8.0
    circuit_failure_threshold: int = 8
    circuit_recovery_seconds: float = 15.0

    enable_xray: bool = False
    cloudwatch_namespace: str = "MMRAG"
    estimated_llm_cost_per_1k_tokens_usd: float = 0.003
    estimated_kb_query_cost_usd: float = 0.0025 / 1_000  # S3 Vectors query API $2.50 / million

    @field_validator("embedding_dimension", "cache_embedding_dimension")
    @classmethod
    def _validate_matryoshka(cls, value: int) -> int:
        if value not in NOVA_MATRYOSHKA_DIMENSIONS:
            raise ValueError(
                f"dimension {value} is not a Nova Matryoshka size; "
                f"allowed={sorted(NOVA_MATRYOSHKA_DIMENSIONS)}"
            )
        return value

    @property
    def redis_url(self) -> str:
        scheme = "rediss" if self.redis_tls else "redis"
        auth = f":{self.redis_auth_token}@" if self.redis_auth_token else ""
        return f"{scheme}://{auth}{self.redis_endpoint}:{self.redis_port}/0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Tests should call get_settings.cache_clear()."""

    return Settings()
