"""Braintrust tracing integration for verifiers v1 (Taskset/Harness API).

Activate with a single call::

    from verifiers.v1.experimental.braintrust_tracing import setup_v1_tracing
    setup_v1_tracing()

This monkey-patches the v1 ``Env``, ``Harness``, and ``Runtime`` classes so
that rollouts, model requests, tool calls, and scoring are traced as nested
spans in Braintrust.  The patching is idempotent.
"""

from .integration import setup_v1_tracing

__all__ = ["setup_v1_tracing"]
