"""Async circuit breaker with exponential backoff and full jitter.

Bedrock on-demand quotas return HTTP 429 / ThrottlingException under burst.
Full jitter (AWS Architecture Blog: Exponential Backoff And Jitter) prevents
synchronized retry storms across the ECS task fleet at peak MAU concurrency.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import ParamSpec, TypeVar

import structlog
from botocore.exceptions import ClientError

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

RETRYABLE_CODES = frozenset(
    {
        "ThrottlingException",
        "Throttling",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "InternalServerException",
        "RequestTimeout",
        "ModelErrorException",
    }
)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when the breaker is open and fail-fast is required."""


def full_jitter(attempt: int, *, base: float, cap: float) -> float:
    """AWS full-jitter delay: `random_between(0, min(cap, base * 2**attempt))`."""

    exp = min(cap, base * (2**attempt))
    return random.random() * exp


def is_retryable(error: BaseException) -> bool:
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        http = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in RETRYABLE_CODES or http == 429
    message = str(error).lower()
    return "throttl" in message or "too many requests" in message or "429" in message


class CircuitBreaker:
    """Per-dependency breaker. One instance should wrap Bedrock Runtime, another Agent Runtime."""

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int,
        recovery_seconds: float,
        max_attempts: int,
        base_backoff: float,
        max_backoff: float,
        on_throttle: Callable[[], None] | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self._on_throttle = on_throttle
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, fn: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        await self._before_call()
        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                result = await fn(*args, **kwargs)
                await self._on_success()
                return result
            except Exception as exc:  # noqa: BLE001 — classified immediately below
                last_error = exc
                if not is_retryable(exc):
                    await self._on_failure()
                    raise
                if self._on_throttle:
                    self._on_throttle()
                delay = full_jitter(attempt, base=self.base_backoff, cap=self.max_backoff)
                logger.warning(
                    "retryable_error",
                    dependency=self.name,
                    attempt=attempt + 1,
                    delay_s=round(delay, 3),
                    error=str(exc)[:300],
                )
                await asyncio.sleep(delay)
        await self._on_failure()
        assert last_error is not None
        raise last_error

    async def _before_call(self) -> None:
        async with self._lock:
            if self._state is CircuitState.OPEN:
                if time.monotonic() - self._opened_at >= self.recovery_seconds:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("circuit_half_open", dependency=self.name)
                else:
                    raise CircuitOpenError(f"circuit open for {self.name}")

    async def _on_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._state = CircuitState.CLOSED

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.error("circuit_open", dependency=self.name, failures=self._failures)
