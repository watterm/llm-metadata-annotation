from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from typing import Protocol, TypeVar

from httpx import HTTPStatusError

T = TypeVar("T")
AsyncCallable = Callable[..., Awaitable[T]]

logger = logging.getLogger("RateLimiter")


def is_rate_limited_error(exc: Exception) -> bool:
    """Return True when the exception represents an HTTP 429 response."""

    if isinstance(exc, HTTPStatusError):
        try:
            return exc.response.status_code == 429
        except Exception:
            return False

    status = getattr(exc, "status_code", None)
    if status == 429:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    return False


@dataclass(slots=True)
class RateLimiterStats:
    """Statistics snapshot for a rate limiter."""

    current_rps: float
    min_rps: float
    max_rps: float
    in_flight: int
    queue_size: int


@dataclass(slots=True)
class RateLimiterConfig:
    """Configuration for local rate limiter with AIMD control.

    AIMD (Additive Increase Multiplicative Decrease) gradually adapts the rate:
    - Increases linearly after consecutive successes (additive)
    - Decreases multiplicatively on 429 errors (multiplicative)
    - Includes cooldown protection to prevent rate thrashing
    """

    # Identity
    name: str = "default"

    # Rate bounds - RPS is always clamped to [min_rps, max_rps]
    min_rps: float = 1.0
    max_rps: float = 3.0
    initial_rps: float = 3.0

    # Retry and timeout behavior
    max_retries: int = 3  # Total attempts per task (initial + retries)
    task_timeout: float | None = None  # Optional per-task timeout (seconds)
    max_concurrency: int | None = None  # Max concurrent tasks (None=unlimited)

    # Rate limit penalty (enforced delay after 429 response)
    rate_limit_penalty_duration: float = 1.0  # Seconds to pause after 429

    # AIMD decrease parameters (multiplicative decrease on 429)
    decrease_factor: float = 0.5  # Multiply current RPS by this on 429
    decrease_cooldown: float = 5.0  # Min seconds between rate decreases

    # AIMD increase parameters (additive increase on success)
    increase_threshold: int = 10  # Consecutive successes needed to increase
    increase_step: float = 0.1  # RPS added on each increase


class AsyncRateLimiter(Protocol):
    """Protocol defining the interface for rate limiters."""

    async def enqueue(
        self, api_call: AsyncCallable[T], *args: object, **kwargs: object
    ) -> T: ...

    async def close(self) -> None: ...
    def stats(self) -> RateLimiterStats: ...
    def update_config(self, **kwargs: object) -> None: ...
