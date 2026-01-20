"""
Logging configuration for MindForge using loguru.
"""

import sys
from typing import Optional

from loguru import logger

from mindforge_dankoe.config.settings import settings


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
    """
    # Remove default handler
    logger.remove()

    # Get log level from settings or parameter
    log_level = level or settings.log_level

    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # File handler if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation="10 MB",
            retention="1 week",
            compression="zip",
        )

    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str = "mindforge"):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance bound with the given name
    """
    return logger.bind(name=name)


# Initialize logging on import
setup_logging()
