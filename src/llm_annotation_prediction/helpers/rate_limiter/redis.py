from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from .base import (
    AsyncRateLimiter,
    RateLimiterConfig,
    RateLimiterStats,
    T,
    is_rate_limited_error,
)

# -- Optional redis dependency -------------------------------------------------
# We isolate imports so type checkers know what we rely on without forcing the
# runtime dependency unless the feature is used.
try:  # Attempt real imports (runtime path)
    import redis.asyncio as _redis_async
except Exception as _e:
    _redis_async = None  # type: ignore[assignment]

    class AsyncScript:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # minimal fallback
            raise RuntimeError("redis extra not installed")

    _import_error: Exception | None = _e
else:
    _import_error = None


if TYPE_CHECKING:  # Protocols only needed for static analysis

    @runtime_checkable
    class _AsyncPipeline(Protocol):  # pragma: no cover - typing aid only
        def hget(self, name: str, key: str) -> _AsyncPipeline: ...
        def get(self, name: str) -> _AsyncPipeline: ...
        def execute(self) -> Awaitable[list[Any]]: ...

    @runtime_checkable
    class _AsyncRedis(Protocol):  # pragma: no cover - typing aid only
        def register_script(self, script: str) -> AsyncScript: ...
        async def incr(self, name: str) -> int: ...
        async def set(
            self,
            name: str,
            value: str,
            *,
            ex: int | None = None,
            px: int | None = None,
            nx: bool | None = None,
            xx: bool | None = None,
        ) -> bool | None: ...
        async def hset(
            self, name: str, mapping: dict[str, Any], **kwargs: Any
        ) -> int: ...
        async def hget(self, name: str, key: str) -> str | None: ...
        def pipeline(self) -> _AsyncPipeline: ...
        async def aclose(self) -> None: ...

    # At type-check time we can alias for clarity
    aioredis = _redis_async
else:  # Runtime alias used below
    aioredis = _redis_async


@dataclass(slots=True)
class RedisRateLimiterConfig(RateLimiterConfig):
    """Configuration for Redis-based distributed rate limiter with AIMD control.

    Inherits all base rate limiter configuration and adds Redis-specific settings
    for distributed coordination across multiple processes/instances.
    """

    # Redis connection (only additional field beyond RateLimiterConfig)
    redis_url: str = "redis://localhost:6379/0"


def _get_redis_client(cfg: RedisRateLimiterConfig) -> _AsyncRedis:
    """Create and return a typed async redis client.

    Separated so the casting & optional dependency check stay in one place.
    """
    if aioredis is None:
        raise RuntimeError(
            "RedisRateLimiter requires the 'redis' extra to be installed"
        ) from _import_error
    client = aioredis.from_url(  # type: ignore[no-untyped-call]
        url=cfg.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    return cast("_AsyncRedis", client)


logger = logging.getLogger("RedisRateLimiter")


class RedisRateLimiter(AsyncRateLimiter):
    """Distributed adaptive rate limiter backed by Redis.

    * Global token bucket stored as a Redis hash (fields: tokens, last_refill, rps,
    succ_total, fail_total, last_adjust).
      * A Lua script performs atomic refill + token consume and returns pacing hints.
      * Each process runs:
        - A launcher coroutine draining a local queue (respecting per-process concurrency).
        - An adjuster coroutine attempting periodic adaptive RPS recalculation guarded by a
          Redis-based lock (SET NX w/ TTL) so only one process adjusts at a time.
      * Success/failure counts are aggregated globally via simple Redis counters.
      * Adaptation logic mirrors the local limiter thresholds, but applied globally; local
      instance does not maintain its own moving window to avoid double adaptation.

    Notes:
      * Concurrency limits are per-process only.
      * Each retry attempt consumes a token (same semantics as local implementation).
      * `update_config` is retained for parity; dynamic changes to thresholds and bounds influence
      subsequent global adjustments (already stored configs are *not* persisted globally).
    """

    # Lua script performing token acquisition & refill.
    # Implements a token bucket algorithm with configurable RPS.
    # Returns: [allowed (0/1), remaining_tokens, current_rps, wait_time]
    _ACQUIRE_SCRIPT = r"""
    local bucket = KEYS[1]
    local default_rps = tonumber(ARGV[1])

    -- Use Redis server time to avoid clock skew in distributed systems
    local time_info = redis.call('TIME')
    local now = tonumber(time_info[1]) + (tonumber(time_info[2]) / 1000000)

    local data = redis.call('HMGET', bucket, 'tokens','last_refill','rps')
    local tokens = tonumber(data[1])
    local last_refill = tonumber(data[2])
    local rps = tonumber(data[3])

    -- Initialize bucket on first access
    if not tokens or not last_refill or not rps then
        tokens = 1
        last_refill = now
        rps = default_rps
        redis.call('HMSET', bucket,'tokens',tokens,'last_refill',last_refill,'rps',rps)
    end

    -- Refill tokens based on elapsed time and current RPS
    local elapsed = now - last_refill
    if elapsed < 0 then elapsed = 0 end

    tokens = math.min(1, tokens + elapsed * rps)
    last_refill = now

    -- Try to consume a token
    local allowed = 0
    local wait = 0
    if tokens >= 1 then
        tokens = tokens - 1
        allowed = 1
    else
        -- Calculate wait time until next token available
        if rps > 0 then
            wait = (1 - tokens) / rps
        else
            wait = 1
        end
    end
    redis.call('HMSET', bucket,'tokens',tokens,'last_refill',last_refill,'rps',rps)
    return {allowed, tostring(tokens), tostring(rps), tostring(wait)}
    """

    # Lua script for AIMD rate decrease (on 429 errors).
    # Multiplicatively decreases RPS with cooldown protection.
    # Returns: [decreased (0/1), new_rps]
    _DECREASE_RATE_SCRIPT = r"""
    local bucket = KEYS[1]
    local decrease_factor = tonumber(ARGV[1])
    local min_rps = tonumber(ARGV[2])
    local decrease_cooldown = tonumber(ARGV[3])

    -- Get current time from Redis server
    local time_info = redis.call('TIME')
    local now = tonumber(time_info[1]) + (tonumber(time_info[2]) / 1000000)

    local data = redis.call('HMGET', bucket, 'rps', 'last_decrease', 'consecutive_successes')
    local current_rps = tonumber(data[1])
    local last_decrease = tonumber(data[2])
    local consecutive = tonumber(data[3])

    -- Initialize if missing
    if not current_rps then current_rps = min_rps end
    if not last_decrease then last_decrease = 0 end
    if not consecutive then consecutive = 0 end

    local decreased = 0
    local new_rps = current_rps

    -- Apply multiplicative decrease if cooldown period has passed
    if (now - last_decrease) >= decrease_cooldown then
        new_rps = math.max(min_rps, current_rps * decrease_factor)
        if new_rps ~= current_rps then
            decreased = 1
            last_decrease = now
        end
    end

    -- Always reset consecutive successes on rate limit error
    redis.call(
        'HMSET', bucket, 'rps', new_rps, 'last_decrease', last_decrease,
        'consecutive_successes', 0
    )
    return {decreased, tostring(new_rps)}
    """

    # Lua script for AIMD rate increase (on consecutive successes).
    # Additively increases RPS after threshold consecutive successes.
    # Returns: [increased (0/1), new_rps, consecutive_count]
    _INCREASE_RATE_SCRIPT = r"""
    local bucket = KEYS[1]
    local increase_step = tonumber(ARGV[1])
    local max_rps = tonumber(ARGV[2])
    local increase_threshold = tonumber(ARGV[3])

    local data = redis.call('HMGET', bucket, 'rps', 'consecutive_successes')
    local current_rps = tonumber(data[1])
    local consecutive = tonumber(data[2])

    -- Initialize if missing
    if not current_rps then current_rps = max_rps end
    if not consecutive then consecutive = 0 end

    -- Increment consecutive successes
    consecutive = consecutive + 1

    local increased = 0
    local new_rps = current_rps

    -- Check if we've hit the threshold
    if consecutive >= increase_threshold then
        new_rps = math.min(max_rps, current_rps + increase_step)
        if new_rps ~= current_rps then
            increased = 1
        end
        consecutive = 0  -- Reset after increase
    end

    redis.call('HMSET', bucket, 'rps', new_rps, 'consecutive_successes', consecutive)
    return {increased, tostring(new_rps), consecutive}
    """

    def __init__(self, cfg: RedisRateLimiterConfig):
        self._cfg: RedisRateLimiterConfig = cfg
        self._logger: logging.Logger = logging.getLogger(cfg.name or "RedisRateLimiter")
        # Build typed redis client via helper (centralizes optional dependency logic)
        self._redis: _AsyncRedis = _get_redis_client(cfg)

        # Local scheduling state
        self._queue: asyncio.Queue[
            tuple[
                Callable[..., Awaitable[Any]],
                tuple[Any, ...],
                dict[str, Any],
                asyncio.Future[Any],
                int,
            ]
        ] = asyncio.Queue()
        self._in_flight: set[asyncio.Task[Any]] = set()
        self._launcher_task: asyncio.Task[None] | None = None
        self._stats_refresh_task: asyncio.Task[None] | None = None
        self._closed = False

        # Derived Redis key namespace
        base: str = f"rl:{cfg.name}"
        self._bucket_key: str = f"{base}:bucket"

        # Initial RPS used for token refill defaults
        self._initial_rps = max(min(cfg.initial_rps, cfg.max_rps), cfg.min_rps)

        # Cached stats (refreshed periodically from Redis)
        self._global_rps = self._initial_rps
        self._stats_refresh_interval = 1.0  # seconds
        self._rate_limit_until = 0.0

        # Script registration (one-time per connection)
        self._acquire_script: AsyncScript = self._redis.register_script(
            script=self._ACQUIRE_SCRIPT
        )
        self._decrease_rate_script: AsyncScript = self._redis.register_script(
            script=self._DECREASE_RATE_SCRIPT
        )
        self._increase_rate_script: AsyncScript = self._redis.register_script(
            script=self._INCREASE_RATE_SCRIPT
        )

    # Public API -----------------------------------------------------------
    async def enqueue(
        self, api_call: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        if self._closed:
            raise RuntimeError("Rate limiter closed")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[T] = loop.create_future()
        await self._queue.put((api_call, args, kwargs, fut, 0))
        if not self._launcher_task:
            self._launcher_task = asyncio.create_task(self._launcher())
            self._stats_refresh_task = asyncio.create_task(self._stats_refresher())
        return await fut

    async def close(self) -> None:
        self._logger.info("Closing RedisRateLimiter")
        self._closed = True

        # Wait for launcher to drain
        if self._launcher_task:
            await self._launcher_task
        if self._stats_refresh_task:
            self._stats_refresh_task.cancel()
            try:
                await self._stats_refresh_task
            except Exception:
                pass
        # Cancel remaining tasks
        for t in list(self._in_flight):
            t.cancel()
        self._in_flight.clear()

        await self._redis.aclose()

    def stats(self) -> RateLimiterStats:
        """Return a cached snapshot; refresh occurs asynchronously.

        This keeps a synchronous API while avoiding illegal nested event loop usage.
        """
        return RateLimiterStats(
            current_rps=self._global_rps,
            min_rps=self._cfg.min_rps,
            max_rps=self._cfg.max_rps,
            in_flight=len(self._in_flight),
            queue_size=self._queue.qsize(),
        )

    def update_config(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            if hasattr(self._cfg, k):
                setattr(self._cfg, k, v)

    # Internal routines ----------------------------------------------------
    async def _launcher(self) -> None:
        while not self._closed or not self._queue.empty():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.2)
            except TimeoutError:
                if self._closed and not self._in_flight and self._queue.empty():
                    self._logger.info(
                        "Auto-shutdown: Closing RedisRateLimiter launcher"
                    )
                    break
                continue

            api_call, args, kwargs, fut, attempt = item
            if fut.done():
                continue

            # Per-process concurrency gate
            if (
                self._cfg.max_concurrency is not None
                and len(self._in_flight) >= self._cfg.max_concurrency
            ):
                await asyncio.sleep(0.05)
                await self._queue.put(item)
                continue

            penalty_delay = self._pending_rate_limit_delay()
            if penalty_delay > 0:
                await asyncio.sleep(penalty_delay)

            # Acquire distributed token (pacing)
            while True:
                allowed, wait = await self._acquire_token()
                if allowed:
                    break
                await asyncio.sleep(min(wait, 0.5))

            task = asyncio.create_task(
                self._run_task(api_call, args, kwargs, fut, attempt)
            )
            self._in_flight.add(task)
            task.add_done_callback(self._in_flight.discard)

    async def _run_task(
        self,
        api_call: Callable[..., Awaitable[T]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        fut: asyncio.Future[T],
        attempt: int,
    ) -> None:
        """Execute a single task with retry logic."""
        if fut.done():
            return

        try:
            result = await self._execute_call(api_call, args, kwargs)
        except Exception as e:
            await self._handle_task_failure(e, api_call, args, kwargs, fut, attempt)
            return

        # Success - increase rate and record
        await self._handle_task_success(result, fut)

    async def _execute_call(
        self,
        api_call: Callable[..., Awaitable[T]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        """Execute the API call with optional timeout."""
        call_coro = api_call(*args, **kwargs)
        if self._cfg.task_timeout:
            return await asyncio.wait_for(call_coro, timeout=self._cfg.task_timeout)
        return await call_coro

    async def _handle_task_success(self, result: T, fut: asyncio.Future[T]) -> None:
        """Handle successful task execution - increase rate and record stats."""
        try:
            res: list[str] = cast(
                list[str],
                await self._increase_rate_script(
                    keys=[self._bucket_key],
                    args=[
                        str(self._cfg.increase_step),
                        str(self._cfg.max_rps),
                        str(self._cfg.increase_threshold),
                    ],
                ),
            )
            increased = bool(int(res[0]))
            new_rps = float(res[1])
            if increased:
                self._global_rps = new_rps
                self._logger.info(f"[AIMD] Increased RPS to {new_rps:.2f}")
        except Exception as script_err:
            self._logger.debug(f"Rate increase script failed: {script_err}")

        if not fut.done():
            fut.set_result(result)

    async def _handle_task_failure(
        self,
        exc: Exception,
        api_call: Callable[..., Awaitable[T]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        fut: asyncio.Future[T],
        attempt: int,
    ) -> None:
        """Handle task failure - apply penalties, adjust rate, and retry if needed."""
        self._logger.warning(
            f"API call failed on attempt {attempt + 1}: {exc} ({type(exc).__name__})"
        )

        is_rate_limit = is_rate_limited_error(exc)
        if is_rate_limit:
            await self._handle_rate_limit_error()
        else:
            await self._reset_consecutive_successes()

        # Retry logic
        if attempt + 1 >= self._cfg.max_retries:
            if not fut.done():
                fut.set_exception(exc)
            return

        await asyncio.sleep(min(1.0, 0.2 * (attempt + 1)))
        await self._queue.put((api_call, args, kwargs, fut, attempt + 1))

    async def _handle_rate_limit_error(self) -> None:
        """Apply penalty and decrease rate on 429 error."""
        deadline = self._mark_rate_limit_penalty(self._cfg.rate_limit_penalty_duration)
        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 0:
            self._logger.info(
                "Rate limited (429). Pausing launches for %.2fs", remaining
            )

        try:
            res: list[str] = cast(
                list[str],
                await self._decrease_rate_script(
                    keys=[self._bucket_key],
                    args=[
                        str(self._cfg.decrease_factor),
                        str(self._cfg.min_rps),
                        str(self._cfg.decrease_cooldown),
                    ],
                ),
            )
            decreased = bool(int(res[0]))
            new_rps = float(res[1])
            if decreased:
                self._global_rps = new_rps
                self._logger.info(f"[AIMD] Decreased RPS to {new_rps:.2f}")
        except Exception as script_err:
            self._logger.debug(f"Rate decrease script failed: {script_err}")

    async def _reset_consecutive_successes(self) -> None:
        """Reset consecutive success counter on non-rate-limit failures."""
        try:
            await self._decrease_rate_script(
                keys=[self._bucket_key],
                args=["1.0", str(self._cfg.min_rps), str(self._cfg.decrease_cooldown)],
            )
        except Exception as script_err:
            self._logger.debug(f"Counter reset script failed: {script_err}")

    async def _acquire_token(self) -> tuple[bool, float]:
        try:
            # res: [allowed, tokens, rps, wait]
            res: list[str] = cast(
                list[str],
                await self._acquire_script(
                    keys=[self._bucket_key],
                    args=[str(self._initial_rps)],
                ),
            )
            allowed = bool(int(res[0]))
            self._global_rps = float(res[2])
            wait = float(res[3])
            return allowed, wait
        except Exception as e:
            self._logger.warning(f"Token acquire failed (fallback wait): {e}")
            return False, 0.5

    def _mark_rate_limit_penalty(self, duration: float) -> float:
        deadline = time.monotonic() + duration
        if deadline > self._rate_limit_until:
            self._rate_limit_until = deadline
        return self._rate_limit_until

    def _pending_rate_limit_delay(self) -> float:
        remaining = self._rate_limit_until - time.monotonic()
        return remaining if remaining > 0 else 0.0

    async def _stats_refresher(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self._stats_refresh_interval)
                pipe = self._redis.pipeline()
                pipe.hget(self._bucket_key, "rps")
                rps_raw = (await pipe.execute())[0]
                if rps_raw:
                    self._global_rps = float(rps_raw)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.debug(f"Stats refresh failed: {e}")
                continue


__all__ = ["RedisRateLimiter", "RedisRateLimiterConfig"]
