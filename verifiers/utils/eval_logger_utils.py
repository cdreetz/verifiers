"""
Structured logging utilities for vf-eval.

This module provides a structlog-based logger specifically for the vf-eval CLI,
offering improved developer experience with colored output, clean tracebacks,
and structured key-value logging without affecting the main verifiers logger.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import Processor


def _add_log_level_style(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add styling hints for log levels (used by ConsoleRenderer)."""
    return event_dict


def setup_eval_logging(level: str = "INFO", force_colors: bool | None = None) -> None:
    """
    Configure structlog for vf-eval with nice console output.

    This sets up structlog with:
    - Colored, formatted console output
    - Clean exception/traceback rendering
    - Timestamp and log level prefixes
    - Key-value context displayed inline

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
        force_colors: Force color output (True/False) or auto-detect (None).
    """
    level_upper = level.upper()
    numeric_level = getattr(logging, level_upper, logging.INFO)

    # Shared processors for all logging paths
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure structlog
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Create a console renderer with nice formatting
    console_processor = structlog.dev.ConsoleRenderer(
        colors=force_colors if force_colors is not None else sys.stderr.isatty(),
        exception_formatter=structlog.dev.plain_traceback,
    )

    # Create formatter using structlog's ProcessorFormatter
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            console_processor,
        ],
    )

    # Set up the handler for eval loggers
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    # Configure the eval-specific logger namespace
    eval_logger = logging.getLogger("vf_eval")
    eval_logger.handlers.clear()
    eval_logger.addHandler(handler)
    eval_logger.setLevel(numeric_level)
    eval_logger.propagate = False

    # Also configure verifiers.scripts and verifiers.utils.eval_utils
    # to use the same handler when called from vf-eval
    for namespace in ["verifiers.scripts.eval", "verifiers.utils.eval_utils"]:
        ns_logger = logging.getLogger(namespace)
        ns_logger.handlers.clear()
        ns_logger.addHandler(handler)
        ns_logger.setLevel(numeric_level)
        ns_logger.propagate = False


def get_eval_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structlog logger for vf-eval.

    Args:
        name: Logger name. If None, uses 'vf_eval'. If provided,
              it will be prefixed with 'vf_eval.' for namespacing.

    Returns:
        A structlog BoundLogger instance with nice formatting.

    Example:
        >>> logger = get_eval_logger(__name__)
        >>> logger.info("Starting evaluation", model="gpt-4", num_examples=10)
    """
    if name is None:
        logger_name = "vf_eval"
    elif name.startswith("verifiers."):
        # Keep original name for verifiers modules
        logger_name = name
    else:
        logger_name = f"vf_eval.{name}"

    return structlog.get_logger(logger_name)


def bind_eval_context(**kwargs: Any) -> None:
    """
    Bind context variables that will be included in all subsequent log messages.

    This is useful for adding context like env_id, model, etc. once
    and having it automatically included in all logs.

    Args:
        **kwargs: Key-value pairs to bind to the logging context.

    Example:
        >>> bind_eval_context(env_id="gsm8k", model="gpt-4")
        >>> logger.info("Starting rollout")  # Will include env_id and model
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_eval_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def unbind_eval_context(*keys: str) -> None:
    """
    Remove specific context variables.

    Args:
        *keys: Names of context variables to remove.
    """
    structlog.contextvars.unbind_contextvars(*keys)
