from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .base import (
    AsyncCallable,
    AsyncRateLimiter,
    RateLimiterConfig,
    RateLimiterStats,
    T,
    is_rate_limited_error,
)

logger = logging.getLogger("LocalRateLimiter")


class LocalRateLimiter(AsyncRateLimiter):
    """Local adaptive rate limiter with AIMD control.

    Uses Additive Increase Multiplicative Decrease (AIMD) for rate adaptation:
    - Gradually increases rate after consecutive successes
    - Multiplicatively decreases rate on 429 errors (with cooldown protection)
    - All state is local to this process
    """

    def __init__(self, cfg: RateLimiterConfig):
        self._cfg = cfg
        self._logger = logging.getLogger(cfg.name or "LocalRateLimiter")

        # AIMD state
        self._current_rps = max(min(cfg.initial_rps, cfg.max_rps), cfg.min_rps)
        self._consecutive_successes = 0
        self._last_rate_decrease: float = 0.0
        self._rate_limit_until: float = 0.0

        # Queue and task management
        self._queue: asyncio.Queue[
            tuple[
                AsyncCallable[Any],
                tuple[Any, ...],
                dict[str, Any],
                asyncio.Future[Any],
                int,
            ]
        ] = asyncio.Queue()
        self._in_flight: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._launcher_task: asyncio.Task[None] | None = None
        self._next_start_time = time.monotonic()

    async def enqueue(self, api_call: AsyncCallable[T], *args: Any, **kwargs: Any) -> T:
        if self._closed:
            raise RuntimeError("Rate limiter closed")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[T] = loop.create_future()
        await self._queue.put((api_call, args, kwargs, fut, 0))
        if not self._launcher_task:
            self._launcher_task = asyncio.create_task(self._launcher())
        return await fut

    async def close(self) -> None:
        self._logger.info("Closing LocalRateLimiter")
        self._closed = True
        if self._launcher_task:
            await self._launcher_task
        # Cancel remaining
        for task in list(self._in_flight):
            task.cancel()
        self._in_flight.clear()

    async def _launcher(self) -> None:
        while not self._closed or not self._queue.empty():
            try:
                api_call, args, kwargs, fut, attempt = await asyncio.wait_for(
                    self._queue.get(), timeout=0.2
                )
            except TimeoutError:
                # Auto-shutdown: if closed requested or no in-flight & empty queue for a while
                if self._closed and not self._in_flight and self._queue.empty():
                    self._logger.info("Auto-shutdown: Closing LocalRateLimiter")
                    break
                continue

            if fut.done():  # Might have been cancelled upstream
                continue

            # Concurrency gate
            if (
                self._cfg.max_concurrency is not None
                and len(self._in_flight) >= self._cfg.max_concurrency
            ):
                # Put back and wait a bit
                await asyncio.sleep(0.05)
                await self._queue.put((api_call, args, kwargs, fut, attempt))
                continue

            # Wait for any active rate limit penalty to expire
            penalty_delay = self._get_penalty_delay()
            if penalty_delay > 0:
                await asyncio.sleep(penalty_delay)

            # Pacing
            now = time.monotonic()
            if now < self._next_start_time:
                await asyncio.sleep(self._next_start_time - now)
            interval = 1 / self._current_rps if self._current_rps > 0 else 0.0
            self._next_start_time = max(now, self._next_start_time) + interval

            task = asyncio.create_task(
                self._run_task(api_call, args, kwargs, fut, attempt),
            )
            self._in_flight.add(task)
            task.add_done_callback(self._in_flight.discard)

    async def _run_task(
        self,
        api_call: AsyncCallable[T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        fut: asyncio.Future[T],
        attempt: int,
    ) -> None:
        """Execute a single API call attempt with retry logic."""
        if fut.done():
            return

        try:
            result = await self._execute_call(api_call, args, kwargs)
            self._handle_success(result, fut)
        except Exception as exc:
            await self._handle_failure(exc, api_call, args, kwargs, fut, attempt)

    async def _execute_call(
        self,
        api_call: AsyncCallable[T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        """Execute the API call with optional timeout."""
        call_coro = api_call(*args, **kwargs)
        if self._cfg.task_timeout:
            return await asyncio.wait_for(call_coro, timeout=self._cfg.task_timeout)
        return await call_coro

    def _handle_success(self, result: T, fut: asyncio.Future[T]) -> None:
        """Handle successful API call completion."""
        self._record_success()
        if not fut.done():
            fut.set_result(result)

    async def _handle_failure(
        self,
        exc: Exception,
        api_call: AsyncCallable[T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        fut: asyncio.Future[T],
        attempt: int,
    ) -> None:
        """Handle failed API call with retry logic."""
        self._logger.warning(
            f"API call failed on attempt {attempt + 1}: {exc} ({type(exc).__name__})"
        )

        # Apply rate limit penalty and AIMD decrease if this is a 429
        is_rate_limit = is_rate_limited_error(exc)
        if is_rate_limit:
            self._apply_rate_limit_penalty()

        # Record failure with AIMD logic
        self._record_failure(is_rate_limit=is_rate_limit)

        # Check if we should retry
        if attempt + 1 >= self._cfg.max_retries:
            if not fut.done():
                fut.set_exception(exc)
            return

        # Schedule retry with exponential backoff
        await asyncio.sleep(min(1.0, 0.2 * (attempt + 1)))
        await self._queue.put((api_call, args, kwargs, fut, attempt + 1))

    def _apply_rate_limit_penalty(self) -> None:
        """Apply penalty delay after receiving a 429 response."""
        penalty_deadline = self._mark_rate_limit_penalty(
            self._cfg.rate_limit_penalty_duration
        )
        self._next_start_time = max(self._next_start_time, penalty_deadline)

    def stats(self) -> RateLimiterStats:
        """Return current rate limiter statistics."""
        return RateLimiterStats(
            current_rps=self._current_rps,
            min_rps=self._cfg.min_rps,
            max_rps=self._cfg.max_rps,
            in_flight=len(self._in_flight),
            queue_size=self._queue.qsize(),
        )

    def update_config(self, **kwargs: object) -> None:
        """Update configuration parameters dynamically."""
        for k, v in kwargs.items():
            if hasattr(self._cfg, k):
                setattr(self._cfg, k, v)

    # AIMD logic
    def _record_success(self) -> None:
        """Record a successful API call and possibly increase RPS."""
        self._consecutive_successes += 1

        if self._consecutive_successes >= self._cfg.increase_threshold:
            old_rps = self._current_rps
            new_rps = min(
                self._cfg.max_rps, self._current_rps + self._cfg.increase_step
            )
            if new_rps != old_rps:
                self._current_rps = new_rps
                self._logger.info(f"Increasing RPS: {old_rps:.2f} -> {new_rps:.2f}")
            self._consecutive_successes = 0

    def _record_failure(self, is_rate_limit: bool) -> None:
        """Record a failed API call and possibly decrease RPS.

        Args:
            is_rate_limit: True if this is a 429 error, False for other failures.
        """
        self._consecutive_successes = 0  # Always reset counter on any failure

        if is_rate_limit:
            now = time.monotonic()
            if (now - self._last_rate_decrease) >= self._cfg.decrease_cooldown:
                old_rps = self._current_rps
                new_rps = max(
                    self._cfg.min_rps, self._current_rps * self._cfg.decrease_factor
                )
                if new_rps != old_rps:
                    self._current_rps = new_rps
                    self._last_rate_decrease = now
                    self._logger.info(f"Decreasing RPS: {old_rps:.2f} -> {new_rps:.2f}")

    # Penalty tracking
    def _mark_rate_limit_penalty(self, duration: float) -> float:
        """Record a pause duration caused by a rate-limit response.

        Returns the absolute monotonic deadline until which launches should wait.
        """
        deadline = time.monotonic() + duration
        if deadline > self._rate_limit_until:
            self._rate_limit_until = deadline
        return self._rate_limit_until

    def _get_penalty_delay(self) -> float:
        """Return remaining seconds for any active rate-limit penalty."""
        remaining = self._rate_limit_until - time.monotonic()
        return remaining if remaining > 0 else 0.0


__all__ = ["LocalRateLimiter"]
