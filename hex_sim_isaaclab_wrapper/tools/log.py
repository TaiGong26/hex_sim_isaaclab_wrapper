"""Logging utilities — formatter and logger factory."""

from __future__ import annotations

import logging
from typing import Any, Optional


class LevelFormatter(logging.Formatter):
    """Formatter that switches format based on log level.

    - DEBUG: verbose with file+line info, full ISO-8601 date.
    - INFO/WARNING/ERROR/CRITICAL: compact with short time.
    """

    def __init__(self) -> None:
        """Initialize the level-aware formatter; format is chosen in `format()`."""
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format the record with a level-specific format string.

        DEBUG records get a verbose format (full date, file/line); all other
        levels get a compact format with short time.

        Args:
            record: Log record to format.

        Returns:
            The formatted log line.
        """
        if record.levelno == logging.DEBUG:
            fmt = (
                "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] "
                "%(message)s"
            )
            datefmt = "%Y-%m-%d %H:%M:%S"
        else:
            fmt = (
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            )
            datefmt = "%H:%M:%S"

        formatter = logging.Formatter(fmt, datefmt)
        return formatter.format(record)


def setup_logger(name: str = "", log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger with console (and optional file) output.

    Args:
        name:     Logger name (device / subsystem identifier).
        log_file: Optional path to a log file.

    Returns:
        Configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates on re-init
    if logger.handlers:
        logger.handlers.clear()

    # 1. Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(LevelFormatter())
    logger.addHandler(console)

    # 2. File handler (if requested)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(LevelFormatter())
        logger.addHandler(file_handler)

    return logger
