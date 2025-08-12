import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Deque, Generic, List, TypeVar
from uuid import uuid4

import aiometer

T = TypeVar("T")
AsyncCallable = Callable[..., Awaitable[T]]


@dataclass
class QueuedTask(Generic[T]):
    """Represents a single API call task with retry tracking."""

    coroutine: AsyncCallable[T]
    future: asyncio.Future[T]
    retry_count: int = 0


class RateLimitedQueue(Generic[T]):
    """
    A dynamic rate limiter that adjusts request rates based on success/failure patterns.

    The rate limiter tracks API call outcomes in a sliding window and adjusts the
    requests per second (RPS) based on the success/failure ratio. When the failure
    rate exceeds a threshold, the RPS is reduced. When the success rate is high
    and stable, the RPS is gradually increased.

    Args:
        name: Name for logging purposes
        min_rps: Minimum requests per second allowed
        max_rps: Maximum requests per second allowed
        initial_rps: Starting requests per second
        max_retries: Maximum number of retries per task
        window_size: Number of recent calls to consider for rate adjustment
        success_threshold: Success rate required to increase RPS (0.0-1.0)
        failure_threshold: Failure rate that triggers RPS reduction (0.0-1.0)
        adjustment_size: Additive adjustment for rate changes (> 0)
        adjustment_cooldown: Seconds to wait between rate adjustments
    """

    def __init__(
        self,
        name: str,
        min_rps: float = 1.0,
        max_rps: float = 3.0,
        initial_rps: float = 3.0,
        max_retries: int = 3,
        window_size: int = 10,
        success_threshold: float = 0.9,
        failure_threshold: float = 0.05,
        adjustment_size: float = 0.5,
        adjustment_cooldown: int = 5,
    ):
        self._name = name
        self._logger = logging.getLogger(name)
        self._max_retries = max_retries

        # Rate limiting state
        self._min_rps = min_rps
        self._max_rps = max_rps
        self._current_rps = max(min_rps, min(max_rps, initial_rps))
        self._window_size = window_size
        self._success_threshold = success_threshold
        self._failure_threshold = failure_threshold
        self._adjustment_size = adjustment_size
        self._adjustment_cooldown = adjustment_cooldown
        self._history: Deque[bool] = deque(maxlen=window_size)
        self._last_adjustment = time.monotonic()

        # Queue state
        self._queue: asyncio.Queue[QueuedTask[T | Exception]] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    async def _worker(self) -> None:
        """Continuously process API call tasks from the queue in batches."""
        while True:
            tasks: List[QueuedTask[T | Exception]] = []

            # Wait briefly for at least one task
            try:
                task_item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                tasks.append(task_item)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                self._logger.debug("Worker task cancelled")
                return

            # Drain any other tasks that are immediately available
            while not self._queue.empty():
                tasks.append(self._queue.get_nowait())

            if tasks:
                await self._run_tasks(tasks)
            else:
                # If no tasks, sleep briefly to avoid busy waiting
                await asyncio.sleep(0.1)

    async def _run_tasks(self, tasks: List[QueuedTask[T | Exception]]) -> None:
        """Run a list of tasks in the queue. Assumes that the coroutines in tasks do
        not throw exceptions.
        This is important because we don't want to stop the whole batch if one of the
        tasks fails. Instead, we want to handle the failure in the task itself and
        continue with the rest of the tasks.
        """
        coros = [task.coroutine for task in tasks]
        self._logger.debug(f"Processing {len(coros)} calls at {self._current_rps} RPS")

        results = await aiometer.run_all(coros, max_per_second=self._current_rps)
        # Set results and adjust rate limit based on success/failure
        for task, result in zip(tasks, results, strict=False):
            if not task.future.done():
                if isinstance(result, Exception):
                    task.future.set_exception(result)
                    self._record_result(False)
                else:
                    task.future.set_result(result)
                    self._record_result(True)

    def _record_result(self, success: bool) -> None:
        """Record an API call result and potentially adjust the rate."""
        self._history.append(success)
        self._maybe_adjust_rate()

    def _maybe_adjust_rate(self) -> None:
        """Adjust RPS if conditions warrant a change."""
        if len(self._history) < self._window_size:
            return

        now = time.monotonic()
        if (now - self._last_adjustment) < self._adjustment_cooldown:
            return

        success_rate = sum(1 for x in self._history if x) / len(self._history)
        failure_rate = 1 - success_rate

        if failure_rate >= self._failure_threshold:
            self._decrease_rate(failure_rate, now)
        elif success_rate >= self._success_threshold:
            self._increase_rate(success_rate, now)

    def _decrease_rate(self, failure_rate: float, now: float) -> None:
        """Decreases rate limit by the specified adjustment size"""
        new_rps = max(self._min_rps, self._current_rps - self._adjustment_size)
        if new_rps != self._current_rps:
            self._current_rps = new_rps
            self._logger.warning(
                f"High failure rate ({failure_rate:.1%}), reducing RPS to {new_rps:.1f}"
            )
            self._last_adjustment = now

    def _increase_rate(self, success_rate: float, now: float) -> None:
        """Increases rate limit by the specified adjustment size"""
        new_rps = min(self._max_rps, self._current_rps + self._adjustment_size)
        if new_rps != self._current_rps:
            self._current_rps = new_rps
            self._logger.info(
                f"High success rate ({success_rate:.1%}), increasing RPS to {new_rps:.1f}"
            )
            self._last_adjustment = now

    async def enqueue(
        self,
        api_call: AsyncCallable[T],
        *args: Any,
    ) -> T:
        """
        Enqueue an API call and return its result when complete. The call will be retried
        if it fails.

        Args:
            api_call: Async function to execute
            *args: Arguments to pass to the api_call

        Returns:
            The result of the API call
        """
        # Start worker task if not running
        if not self._worker_task:
            self._logger.debug("Starting worker task")
            self._worker_task = asyncio.create_task(self._worker())

        # We don't raise exceptions, so aiometer doesn't receive them. Otherwise, it
        # would stop a complete task batch.
        async def task_wrapper() -> T | Exception:
            try:
                return await api_call(*args)
            except Exception as e:
                return e

        loop = asyncio.get_running_loop()
        task = QueuedTask(coroutine=task_wrapper, future=loop.create_future())

        # This is just for tracking in logs
        id = str(uuid4())[:8]

        # Try the task until we get a result or hit the retry limit
        result: T | Exception | None = None
        while result is None and task.retry_count < self._max_retries:
            self._logger.debug(f"{id}: Enqueuing task for attempt {task.retry_count}")
            await self._queue.put(task)
            try:
                result = await task.future
                self._logger.debug(f"{id}: Task result: {result}")
            except Exception as error:
                self._logger.warning(f"{id}: Exception in task: {error}")
                task.retry_count += 1

                if task.retry_count >= self._max_retries:
                    self._logger.error(
                        f"{id}: All retries failed. Final error: {error}"
                    )
                    raise error

                self._logger.info(f"{id}: Retrying. Attempt {task.retry_count}")
                task.future = loop.create_future()

        # This actually should not occur here and is a safety measure (and fixes typing)
        if isinstance(result, Exception) or result is None:
            raise ValueError(f"{id}: Could not finish task")

        self._logger.debug(f"{id}: Task completed successfully")

        return result
