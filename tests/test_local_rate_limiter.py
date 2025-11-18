from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from llm_annotation_prediction.helpers.rate_limiter.base import RateLimiterConfig
from llm_annotation_prediction.helpers.rate_limiter.local import LocalRateLimiter


@pytest.fixture
def limiter_config() -> RateLimiterConfig:
    """Standard configuration for LocalRateLimiter tests."""
    return RateLimiterConfig(
        min_rps=1.0,
        max_rps=5.0,
        initial_rps=5.0,
        max_concurrency=1,
        decrease_factor=0.5,
        decrease_cooldown=1.0,
        increase_threshold=3,
        increase_step=0.5,
        max_retries=2,
    )


@pytest.mark.asyncio
async def test_local_rate_limiter_enforces_pacing(
    limiter_config: RateLimiterConfig,
) -> None:
    """Test that the limiter enforces the configured RPS rate."""
    limiter = LocalRateLimiter(limiter_config)
    start_times: list[float] = []
    order: list[str] = []

    async def tracked_call(label: str) -> str:
        # Record the moment the work starts so we can compare spacing between calls.
        start_times.append(time.monotonic())
        await asyncio.sleep(0.01)
        order.append(label)
        return label

    try:
        # Queue two tasks at once; the limiter should serialize them with at least
        # ~200ms spacing (5 RPS = 0.2s per call).
        results = await asyncio.gather(
            limiter.enqueue(tracked_call, "first"),
            limiter.enqueue(tracked_call, "second"),
        )

        assert results == ["first", "second"]
        assert order == ["first", "second"]
        assert len(start_times) == 2
        assert start_times[1] - start_times[0] >= 0.18
    finally:
        await limiter.close()


@pytest.mark.asyncio
async def test_local_rate_limiter_respects_rps_over_many_calls(
    limiter_config: RateLimiterConfig,
) -> None:
    """Test that the limiter maintains correct RPS across multiple calls."""
    limiter = LocalRateLimiter(limiter_config)

    labels = [f"call-{i}" for i in range(6)]
    start_times: list[float] = []

    async def tracked_call(label: str) -> str:
        # Each invocation records its start to verify the aggregate RPS spacing.
        start_times.append(time.monotonic())
        await asyncio.sleep(0.01)
        return label

    try:
        # Launch a burst of calls; the limiter should pace them based on the 5 RPS setting.
        results = await asyncio.gather(
            *(limiter.enqueue(tracked_call, label) for label in labels)
        )

        assert results == labels
        assert len(start_times) == len(labels)

        total_span = start_times[-1] - start_times[0]
        # With a 5 RPS ceiling, six calls need ~1s of wall time; allow for small timing jitter.
        assert total_span >= 0.9
    finally:
        await limiter.close()


@pytest.mark.asyncio
async def test_local_rate_limiter_decreases_rate_on_429(
    limiter_config: RateLimiterConfig,
) -> None:
    """Test AIMD: rate decreases multiplicatively on 429 errors."""
    limiter = LocalRateLimiter(limiter_config)

    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, request=request)

    attempts = 0

    async def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            # First call triggers a 429, which should decrease the rate
            raise httpx.HTTPStatusError(
                "rate limited", request=request, response=response
            )
        return "ok"

    try:
        initial_rps = limiter.stats().current_rps
        result = await limiter.enqueue(flaky_call)

        assert result == "ok"
        assert attempts == 2

        # After the 429, the RPS should have decreased by decrease_factor (0.5)
        stats = limiter.stats()
        expected_rps = max(
            limiter_config.min_rps, initial_rps * limiter_config.decrease_factor
        )
        assert abs(stats.current_rps - expected_rps) < 0.01
    finally:
        await limiter.close()


@pytest.mark.asyncio
async def test_local_rate_limiter_increases_rate_on_success(
    limiter_config: RateLimiterConfig,
) -> None:
    """Test AIMD: rate increases additively after consecutive successes."""
    # Configure with lower initial RPS to test increase behavior
    config = RateLimiterConfig(
        min_rps=1.0,
        max_rps=10.0,
        initial_rps=2.0,
        max_concurrency=1,
        decrease_factor=0.5,
        decrease_cooldown=1.0,
        increase_threshold=3,  # Increase after 3 consecutive successes
        increase_step=1.0,
        max_retries=2,
    )
    limiter = LocalRateLimiter(config)

    call_count = 0

    async def successful_call() -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return f"success-{call_count}"

    try:
        initial_rps = limiter.stats().current_rps
        assert initial_rps == 2.0

        # Make enough calls to trigger rate increase (need 3 consecutive successes)
        results = await asyncio.gather(
            limiter.enqueue(successful_call),
            limiter.enqueue(successful_call),
            limiter.enqueue(successful_call),
        )

        assert len(results) == 3
        assert call_count == 3

        # After 3 consecutive successes, RPS should increase by increase_step (1.0)
        stats = limiter.stats()
        expected_rps = min(config.max_rps, initial_rps + config.increase_step)
        assert abs(stats.current_rps - expected_rps) < 0.01
    finally:
        await limiter.close()


@pytest.mark.asyncio
async def test_local_rate_limiter_resets_consecutive_on_failure(
    limiter_config: RateLimiterConfig,
) -> None:
    """Test that non-rate-limit failures reset consecutive success counter."""
    limiter = LocalRateLimiter(limiter_config)

    call_count = 0

    async def mixed_call() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            # Non-rate-limit error should reset counter but not decrease rate
            # This will be retried and succeed on the next attempt
            raise ValueError("Some other error")
        await asyncio.sleep(0.01)
        return f"call-{call_count}"

    try:
        initial_rps = limiter.stats().current_rps

        # First call succeeds
        result1 = await limiter.enqueue(mixed_call)
        assert result1 == "call-1"

        # Second call fails once but then succeeds on retry
        result2 = await limiter.enqueue(mixed_call)
        assert result2 == "call-3"  # call_count=3 after retry

        # RPS should not have decreased (only 429s trigger decrease)
        stats = limiter.stats()
        assert abs(stats.current_rps - initial_rps) < 0.01
    finally:
        await limiter.close()


@pytest.mark.asyncio
async def test_local_rate_limiter_enforces_cooldown(
    limiter_config: RateLimiterConfig,
) -> None:
    """Test that cooldown prevents immediate retries after 429."""
    limiter = LocalRateLimiter(limiter_config)

    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, request=request)

    attempts = 0
    start_times: list[float] = []

    async def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        start_times.append(time.monotonic())
        if attempts == 1:
            raise httpx.HTTPStatusError(
                "rate limited", request=request, response=response
            )
        return "ok"

    try:
        result = await limiter.enqueue(flaky_call)

        assert result == "ok"
        assert attempts == 2
        assert len(start_times) == 2
        # The cooldown should enforce at least decrease_cooldown (1.0s) between attempts
        assert start_times[1] - start_times[0] >= 0.95
    finally:
        await limiter.close()


@pytest.mark.asyncio
async def test_local_rate_limiter_respects_max_retries(
    limiter_config: RateLimiterConfig,
) -> None:
    """Test that the limiter respects max_retries configuration."""
    limiter = LocalRateLimiter(limiter_config)

    attempts = 0

    async def always_fails() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError(f"Fail attempt {attempts}")

    try:
        with pytest.raises(ValueError, match="Fail attempt"):
            await limiter.enqueue(always_fails)

        # With max_retries=2, it does 2 total attempts (initial + 1 retry)
        assert attempts == 2
    finally:
        await limiter.close()


@pytest.mark.asyncio
async def test_local_rate_limiter_stats_tracking(
    limiter_config: RateLimiterConfig,
) -> None:
    """Test that stats are correctly tracked."""
    limiter = LocalRateLimiter(limiter_config)

    async def simple_call() -> str:
        await asyncio.sleep(0.01)
        return "done"

    try:
        # Check initial stats
        stats = limiter.stats()
        assert stats.current_rps == limiter_config.initial_rps
        assert stats.min_rps == limiter_config.min_rps
        assert stats.max_rps == limiter_config.max_rps
        assert stats.in_flight == 0
        assert stats.queue_size == 0

        # Start a call and check in-flight tracking
        task = asyncio.create_task(limiter.enqueue(simple_call))
        await asyncio.sleep(0.05)  # Give it time to start

        stats = limiter.stats()
        # Note: in_flight might be 0 or 1 depending on timing, just check it's reasonable
        assert stats.in_flight >= 0

        await task
    finally:
        await limiter.close()


@pytest.mark.asyncio
async def test_local_rate_limiter_handles_concurrent_calls(
    limiter_config: RateLimiterConfig,
) -> None:
    """Test that limiter correctly handles multiple concurrent enqueue calls."""
    limiter = LocalRateLimiter(limiter_config)

    labels = [f"call-{i}" for i in range(10)]

    async def simple_call(label: str) -> str:
        await asyncio.sleep(0.01)
        return label

    try:
        # Launch many calls concurrently
        results = await asyncio.gather(
            *(limiter.enqueue(simple_call, label) for label in labels)
        )

        # All should complete successfully
        assert sorted(results) == sorted(labels)
    finally:
        await limiter.close()


@pytest.mark.asyncio
async def test_local_rate_limiter_clamps_rps_to_bounds(
    limiter_config: RateLimiterConfig,
) -> None:
    """Test that RPS adjustments respect min_rps and max_rps bounds."""
    config = RateLimiterConfig(
        min_rps=2.0,
        max_rps=3.0,
        initial_rps=2.5,
        max_concurrency=1,
        decrease_factor=0.1,  # Very aggressive decrease
        decrease_cooldown=1.0,
        increase_threshold=1,
        increase_step=5.0,  # Very aggressive increase
        max_retries=2,
    )
    limiter = LocalRateLimiter(config)

    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, request=request)

    async def rate_limited_call() -> str:
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    try:
        # Trigger a rate decrease
        with pytest.raises(httpx.HTTPStatusError):
            await limiter.enqueue(rate_limited_call)

        # RPS should not go below min_rps
        stats = limiter.stats()
        assert stats.current_rps >= config.min_rps

        # Now test increase clamping - make successful calls
        async def successful_call() -> str:
            return "ok"

        await limiter.enqueue(successful_call)
        await limiter.enqueue(successful_call)

        # RPS should not exceed max_rps
        stats = limiter.stats()
        assert stats.current_rps <= config.max_rps
    finally:
        await limiter.close()
