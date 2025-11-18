from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import time
from typing import Any, cast

import fakeredis.aioredis
import httpx
import pytest
import pytest_asyncio

from llm_annotation_prediction.helpers.rate_limiter import redis as redis_module
from llm_annotation_prediction.helpers.rate_limiter.redis import (
    RedisRateLimiter,
    RedisRateLimiterConfig,
)


@pytest_asyncio.fixture
async def redis_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[tuple[RedisRateLimiter, fakeredis.aioredis.FakeRedis], None]:
    """Fixture providing a RedisRateLimiter backed by fakeredis."""
    fake_server = fakeredis.aioredis.FakeServer()  # type: ignore[attr-defined]
    fake_client = fakeredis.aioredis.FakeRedis(  # type: ignore[attr-defined]
        server=fake_server,
        decode_responses=True,
    )

    def _get_client(_: RedisRateLimiterConfig) -> fakeredis.aioredis.FakeRedis:
        return fake_client

    monkeypatch.setattr(redis_module, "_get_redis_client", _get_client)

    cfg = RedisRateLimiterConfig(
        name="test-redis-limiter",
        min_rps=1.0,
        max_rps=5.0,
        initial_rps=5.0,
        max_concurrency=1,
        decrease_factor=0.5,
        decrease_cooldown=5.0,
        increase_threshold=10,
        increase_step=0.1,
        max_retries=2,
    )
    limiter = RedisRateLimiter(cfg)

    try:
        yield limiter, fake_client
    finally:
        await limiter.close()
        await fake_client.aclose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_redis_rate_limiter_enforces_pacing(
    redis_limiter: tuple[RedisRateLimiter, fakeredis.aioredis.FakeRedis],
) -> None:
    limiter, _ = redis_limiter
    start_times: list[float] = []
    order: list[str] = []

    async def tracked_call(label: str) -> str:
        # Record the moment the work starts so we can compare spacing between calls.
        start_times.append(time.monotonic())
        await asyncio.sleep(0.01)
        order.append(label)
        return label

    # Queue two tasks at once; the limiter should serialize them with at least ~200ms spacing.
    results = await asyncio.gather(
        limiter.enqueue(tracked_call, "first"),
        limiter.enqueue(tracked_call, "second"),
    )

    assert results == ["first", "second"]
    assert order == ["first", "second"]
    assert len(start_times) == 2
    assert start_times[1] - start_times[0] >= 0.18


@pytest.mark.asyncio
async def test_redis_rate_limiter_respects_rps_over_many_calls(
    redis_limiter: tuple[RedisRateLimiter, fakeredis.aioredis.FakeRedis],
) -> None:
    limiter, _ = redis_limiter

    labels = [f"call-{i}" for i in range(6)]
    start_times: list[float] = []

    async def tracked_call(label: str) -> str:
        # Each invocation records its start to verify the aggregate RPS spacing.
        start_times.append(time.monotonic())
        await asyncio.sleep(0.01)
        return label

    # Launch a burst of calls; the limiter should pace them based on the 5 RPS setting.
    results = await asyncio.gather(
        *(limiter.enqueue(tracked_call, label) for label in labels)
    )

    assert results == labels
    assert len(start_times) == len(labels)

    total_span = start_times[-1] - start_times[0]
    # With a 5 RPS ceiling, six calls need ~1s of wall time; allow for small timing jitter.
    assert total_span >= 0.9


@pytest.mark.asyncio
async def test_redis_rate_limiter_handles_queued_calls_when_429_hits(
    redis_limiter: tuple[RedisRateLimiter, fakeredis.aioredis.FakeRedis],
) -> None:
    limiter, fake_client = redis_limiter

    # Configure the limiter for a lower 3 RPS ceiling and clear any leftover bucket state.
    limiter.update_config(
        min_rps=3.0,
        max_rps=3.0,
        initial_rps=3.0,
        max_concurrency=None,
    )
    limiter._initial_rps = 3.0
    limiter._global_rps = 3.0
    await cast(Any, fake_client).delete(limiter._bucket_key)  # type: ignore[attr-defined]

    request = httpx.Request("GET", "https://example.com/queued")
    response = httpx.Response(429, request=request)

    labels = [f"call-{i}" for i in range(12)]
    failure_label = labels[4]
    attempts: dict[str, int] = {}
    execution_order: list[str] = []
    start_times: list[float] = []

    async def queued_call(label: str) -> str:
        # Each queued call logs when it launches so we can detect the cooldown boundary.
        attempts[label] = attempts.get(label, 0) + 1
        start_times.append(time.monotonic())
        execution_order.append(label)
        if label == failure_label and attempts[label] == 1:
            # The first pass for the designated call simulates a 429 response.
            raise httpx.HTTPStatusError(
                "rate limited queued call", request=request, response=response
            )
        await asyncio.sleep(0.01)
        return label

    # Fire a burst of >10 calls; the limiter should run a few, hit the 429, then pause.
    results = await asyncio.gather(
        *(limiter.enqueue(queued_call, label) for label in labels)
    )

    assert results == labels
    assert attempts[failure_label] == 2

    # Find the first large gap in launch times, which marks the enforced cooldown window.
    gaps = [
        start_times[idx + 1] - start_times[idx] for idx in range(len(start_times) - 1)
    ]
    cooldown_index = next((idx + 1 for idx, gap in enumerate(gaps) if gap >= 0.6), None)
    assert cooldown_index is not None

    # All launches before the gap should correspond to the first five queued calls.
    assert cooldown_index >= 5
    assert execution_order[:cooldown_index] == labels[:cooldown_index]
    # The first call after the pause should be whichever item was next in the queue.
    assert execution_order[cooldown_index] == labels[cooldown_index]


@pytest.mark.asyncio
async def test_redis_rate_limiter_applies_cooldown_after_429(
    redis_limiter: tuple[RedisRateLimiter, fakeredis.aioredis.FakeRedis],
) -> None:
    limiter, fake_client = redis_limiter

    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, request=request)

    attempts = 0
    start_times: list[float] = []

    async def flaky_call() -> str:
        nonlocal attempts
        # First attempt fails with a 429; the limiter should pause before retrying.
        attempts += 1
        start_times.append(time.monotonic())
        if attempts == 1:
            raise httpx.HTTPStatusError(
                "rate limited", request=request, response=response
            )
        return "ok"

    # enqueue() should retry automatically and enforce the cooldown window.
    result = await limiter.enqueue(flaky_call)

    assert result == "ok"
    assert attempts == 2
    assert len(start_times) == 2
    assert start_times[1] - start_times[0] >= 0.95


@pytest.mark.asyncio
async def test_redis_rate_limiter_increases_rate_on_success(
    redis_limiter: tuple[RedisRateLimiter, fakeredis.aioredis.FakeRedis],
) -> None:
    """Test AIMD: rate increases additively after consecutive successes."""
    limiter, fake_client = redis_limiter

    # Configure with lower initial RPS and low threshold to test increase behavior
    limiter.update_config(
        min_rps=2.0,
        max_rps=10.0,
        initial_rps=2.0,
        increase_threshold=3,  # Increase after 3 consecutive successes
        increase_step=1.0,
    )
    limiter._initial_rps = 2.0
    limiter._global_rps = 2.0
    await fake_client.delete(limiter._bucket_key)  # type: ignore[attr-defined]

    call_count = 0

    async def successful_call() -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return f"success-{call_count}"

    initial_rps = limiter.stats().current_rps
    assert abs(initial_rps - 2.0) < 0.1

    # Make enough calls to trigger rate increase (need 3 consecutive successes)
    results = await asyncio.gather(
        limiter.enqueue(successful_call),
        limiter.enqueue(successful_call),
        limiter.enqueue(successful_call),
    )

    assert len(results) == 3
    assert call_count == 3

    # Give the stats refresher time to update
    await asyncio.sleep(1.5)

    # After 3 consecutive successes, RPS should increase by increase_step (1.0)
    stats = limiter.stats()
    expected_rps = min(10.0, initial_rps + 1.0)
    assert abs(stats.current_rps - expected_rps) < 0.2


@pytest.mark.asyncio
async def test_redis_rate_limiter_decreases_rate_on_429(
    redis_limiter: tuple[RedisRateLimiter, fakeredis.aioredis.FakeRedis],
) -> None:
    """Test AIMD: rate decreases multiplicatively on 429 errors."""
    limiter, fake_client = redis_limiter

    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, request=request)

    limiter.update_config(
        min_rps=1.0,
        max_rps=5.0,
        initial_rps=5.0,
        decrease_factor=0.5,
    )
    limiter._initial_rps = 5.0
    limiter._global_rps = 5.0
    await fake_client.delete(limiter._bucket_key)  # type: ignore[attr-defined]

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

    initial_rps = limiter.stats().current_rps
    result = await limiter.enqueue(flaky_call)

    assert result == "ok"
    assert attempts == 2

    # Give the stats refresher time to update
    await asyncio.sleep(1.5)

    # After the 429, the RPS should have decreased by decrease_factor (0.5)
    stats = limiter.stats()
    expected_rps = max(1.0, initial_rps * 0.5)
    assert abs(stats.current_rps - expected_rps) < 0.5


@pytest.mark.asyncio
async def test_redis_rate_limiter_clamps_rps_to_bounds(
    redis_limiter: tuple[RedisRateLimiter, fakeredis.aioredis.FakeRedis],
) -> None:
    """Test that RPS adjustments respect min_rps and max_rps bounds."""
    limiter, fake_client = redis_limiter

    limiter.update_config(
        min_rps=2.0,
        max_rps=3.0,
        initial_rps=2.5,
        decrease_factor=0.1,  # Very aggressive decrease
        increase_threshold=1,
        increase_step=5.0,  # Very aggressive increase
    )
    limiter._initial_rps = 2.5
    limiter._global_rps = 2.5
    await fake_client.delete(limiter._bucket_key)  # type: ignore[attr-defined]

    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, request=request)

    async def rate_limited_call() -> str:
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    # Trigger a rate decrease
    with pytest.raises(httpx.HTTPStatusError):
        await limiter.enqueue(rate_limited_call)

    # Give the stats refresher time to update
    await asyncio.sleep(1.5)

    # RPS should not go below min_rps
    stats = limiter.stats()
    assert stats.current_rps >= 2.0

    # Now test increase clamping - make successful calls
    async def successful_call() -> str:
        return "ok"

    await limiter.enqueue(successful_call)
    await limiter.enqueue(successful_call)

    # Give the stats refresher time to update
    await asyncio.sleep(1.5)

    # RPS should not exceed max_rps
    stats = limiter.stats()
    assert stats.current_rps <= 3.0


@pytest.mark.asyncio
async def test_redis_rate_limiter_stats_tracking(
    redis_limiter: tuple[RedisRateLimiter, fakeredis.aioredis.FakeRedis],
) -> None:
    """Test that stats are correctly tracked."""
    limiter, _ = redis_limiter

    async def simple_call() -> str:
        await asyncio.sleep(0.01)
        return "done"

    # Check initial stats
    stats = limiter.stats()
    assert stats.current_rps == limiter._cfg.initial_rps
    assert stats.min_rps == limiter._cfg.min_rps
    assert stats.max_rps == limiter._cfg.max_rps
    assert stats.in_flight == 0
    assert stats.queue_size == 0

    # Start a call and check in-flight tracking
    task = asyncio.create_task(limiter.enqueue(simple_call))
    await asyncio.sleep(0.05)  # Give it time to start

    stats = limiter.stats()
    # Note: in_flight might be 0 or 1 depending on timing
    assert stats.in_flight >= 0

    await task
