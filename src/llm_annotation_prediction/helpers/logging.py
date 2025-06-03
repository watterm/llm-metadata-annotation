import logging
from logging.handlers import MemoryHandler
from pathlib import Path
from typing import Optional

LOG_FILENAME = "log.txt"
_LOG_BASE = "%(asctime)s | %(levelname)-7s | %(name)-16.16s"
LOG_FORMAT = f"{_LOG_BASE} | %(message)s"
LOG_FORMAT_DEBUG = f"{_LOG_BASE} | %(filename)-16.16s:%(lineno)-4s | %(message)s"
logger = logging.getLogger("Scaffold")


def set_external_baseline_log_levels(log_level: str) -> None:
    """
    Prevent external loggers from showing debug information
    """
    level = max(getattr(logging, log_level.upper()), logging.INFO)

    logging.getLogger("httpx").setLevel(level)
    logging.getLogger("httpcore").setLevel(level)
    logging.getLogger("asyncio").setLevel(level)


def setup_memory_logging() -> None:
    """
    Setup logging to a buffer in memory.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        handlers=[MemoryHandler(10000)],
    )


def setup_logging(folder: Path, no_save: bool, silent: bool, log_level: str) -> None:
    """
    Set up the logging module according to the config options.
    """

    root_logger = logging.getLogger()

    # Find out if there is a memory handler set up
    buffer: Optional[MemoryHandler] = None
    if root_logger.hasHandlers() and isinstance(root_logger.handlers[0], MemoryHandler):
        buffer = root_logger.handlers[0]

    # Set up logging according to options
    if not no_save:
        root_logger.handlers.append(logging.FileHandler(folder / LOG_FILENAME))
    if not silent:
        root_logger.handlers.append(logging.StreamHandler())

    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    for handler in root_logger.handlers:
        handler.setFormatter(
            logging.Formatter(LOG_FORMAT if log_level != "DEBUG" else LOG_FORMAT_DEBUG)
        )

    # If we used a memory buffer, flush it to the new main handler and remove it
    if buffer:
        if len(root_logger.handlers) > 1:
            buffer.setTarget(root_logger.handlers[1])
        buffer.flush()
        root_logger.removeHandler(buffer)

    set_external_baseline_log_levels(log_level)
