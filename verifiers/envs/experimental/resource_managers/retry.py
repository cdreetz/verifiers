"""Retry configuration with tenacity."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import tenacity as tc

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff and jitter.

    Attributes:
        max_attempts: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff.
        jitter: Random jitter factor (0.1 = ±10%).
        retry_on: Exception types to retry on (whitelist). If set, only these are retried.
        no_retry_on: Exception types to NOT retry on (blacklist). If set, all except these are retried.
        retry_predicate: Custom callable (Exception) -> bool for complex retry logic.

    Note:
        retry_on, no_retry_on, and retry_predicate are mutually exclusive.
        If none are set, all exceptions are retried.

    Examples:
        # Retry only on connection errors
        RetryConfig(retry_on=(ConnectionError, TimeoutError))

        # Retry on everything except validation errors
        RetryConfig(no_retry_on=(ValueError, TypeError))

        # Custom predicate
        RetryConfig(retry_predicate=is_transient_error)
    """

    max_attempts: int = 5
    initial_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1

    # Exception filtering (mutually exclusive)
    retry_on: tuple[type[Exception], ...] | None = None
    no_retry_on: tuple[type[Exception], ...] | None = None
    retry_predicate: Callable[[Exception], bool] | None = None

    def __post_init__(self) -> None:
        """Validate that exception filtering options are mutually exclusive."""
        options_set = sum([
            self.retry_on is not None,
            self.no_retry_on is not None,
            self.retry_predicate is not None,
        ])
        if options_set > 1:
            raise ValueError(
                "retry_on, no_retry_on, and retry_predicate are mutually exclusive. "
                "Only one can be set."
            )

    def _build_retry_condition(self) -> tc.retry_base | None:
        """Build the tenacity retry condition based on config."""
        if self.retry_on is not None:
            return tc.retry_if_exception_type(self.retry_on)
        elif self.no_retry_on is not None:
            return tc.retry_if_not_exception_type(self.no_retry_on)
        elif self.retry_predicate is not None:
            return tc.retry_if_exception(self.retry_predicate)
        return None  # Retry on all exceptions (tenacity default)

    def build_retryer(self, logger: logging.Logger | None = None) -> tc.AsyncRetrying:
        """Build a tenacity AsyncRetrying instance with this config."""
        kwargs: dict[str, Any] = {
            "stop": tc.stop_after_attempt(self.max_attempts),
            "wait": tc.wait_exponential_jitter(
                initial=self.initial_delay,
                exp_base=self.exponential_base,
                max=self.max_delay,
                jitter=self.jitter,
            ),
            "reraise": True,
        }

        retry_condition = self._build_retry_condition()
        if retry_condition is not None:
            kwargs["retry"] = retry_condition

        if logger:
            kwargs["before_sleep"] = tc.before_sleep_log(logger, logging.WARNING)

        return tc.AsyncRetrying(**kwargs)

    def wrap(self, func: Callable[..., T], logger: logging.Logger | None = None) -> Callable[..., T]:
        """Wrap a function with retry logic."""
        return self.build_retryer(logger).wraps(func)


# Default configs for common use cases
DEFAULT_RETRY_CONFIG = RetryConfig()

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=10,
    initial_delay=0.25,
    max_delay=60.0,
    jitter=0.2,
)

CONSERVATIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=10.0,
    jitter=0.05,
)


# Common retry predicates
def is_transient_error(exc: Exception) -> bool:
    """Retry on transient/temporary errors."""
    transient_keywords = ["timeout", "transient", "temporary", "retry", "unavailable", "503", "429"]
    exc_str = str(exc).lower()
    return any(kw in exc_str for kw in transient_keywords)


def is_connection_error(exc: Exception) -> bool:
    """Retry on connection-related errors."""
    connection_types = (ConnectionError, TimeoutError, OSError)
    if isinstance(exc, connection_types):
        return True
    connection_keywords = ["connection", "network", "socket", "refused", "reset"]
    return any(kw in str(exc).lower() for kw in connection_keywords)
