"""
Logging configuration for MCP Proxy Server.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    stream: Optional[sys.stderr.__class__] = None,
) -> logging.Logger:
    """
    Set up structured logging for the MCP Proxy Server.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string. If None, uses a structured format.
        stream: Output stream. Defaults to stderr.

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "[%(levelname)s] %(name)s: %(message)s"

    if stream is None:
        stream = sys.stderr

    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        stream=stream,
        force=True,  # Override any existing configuration
    )

    # Get logger for this module
    logger = logging.getLogger("mcp_proxy")
    logger.setLevel(numeric_level)

    return logger


def get_logger(name: str = "mcp_proxy") -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ or module name)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)

