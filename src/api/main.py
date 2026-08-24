"""ASGI entrypoint for ECS Fargate (uvicorn) and Lambda (Mangum)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import lifespan
from api.routes import router

app = FastAPI(
    title="Multimodal RAG API",
    version="1.0.0",
    description=(
        "Production multimodal RAG over Amazon Bedrock Knowledge Bases with "
        "Amazon S3 Vectors, Nova Multimodal Embeddings, and two-tier Redis caching."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
    expose_headers=["X-Request-Id"],
)
app.include_router(router)

# Optional Lambda adapter. Unused on ECS; Mangum is a no-op import cost.
try:
    from mangum import Mangum

    handler = Mangum(app, lifespan="off")
except Exception:  # pragma: no cover
    handler = None
