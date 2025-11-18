from __future__ import annotations

from .base import AsyncRateLimiter, RateLimiterConfig
from .local import LocalRateLimiter

try:  # optional redis import
    from .redis import RedisRateLimiter, RedisRateLimiterConfig
except Exception:  # pragma: no cover
    RedisRateLimiter = None  # type: ignore
    RedisRateLimiterConfig = None  # type: ignore


def create_rate_limiter(
    config: RateLimiterConfig,
) -> AsyncRateLimiter:
    """Create a rate limiter instance from a config object.

    Kind is inferred by the concrete config type (RedisRateLimiterConfig -> redis).
    """
    if RedisRateLimiterConfig is not None and isinstance(
        config, RedisRateLimiterConfig
    ):
        if RedisRateLimiter is None:
            raise RuntimeError(
                "Redis support unavailable. Install with "
                "'pip install llm-metadata-annotation[redis]'"
            )
        return RedisRateLimiter(config)
    # Fallback: local adaptive limiter
    return LocalRateLimiter(config)  # type: ignore[arg-type]
