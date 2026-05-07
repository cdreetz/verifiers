"""OpenTelemetry-based telemetry for verifiers.

Enable with ``VF_TELEMETRY=1``.  When disabled (the default) every public
helper is a lightweight no-op so there is zero overhead in production runs
that don't need observability.

Metrics are exported via OTLP (gRPC by default) so they can be scraped by
the OpenTelemetry Collector → Prometheus → Grafana pipeline.  Traces go
to the same collector endpoint and can be viewed in Tempo / Jaeger.

Environment variables (all optional):
    VF_TELEMETRY            – "1" to enable, anything else or unset to disable.
    OTEL_EXPORTER_OTLP_ENDPOINT – Collector endpoint (default http://localhost:4317).
    OTEL_SERVICE_NAME       – Override service name (default "verifiers").
    VF_TELEMETRY_CONSOLE    – "1" to also print metrics/traces to stderr.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy initialisation flag
# ---------------------------------------------------------------------------
_enabled: bool | None = None
_initialised: bool = False


def is_enabled() -> bool:
    """Return ``True`` when telemetry is active."""
    global _enabled
    if _enabled is None:
        _enabled = os.getenv("VF_TELEMETRY", "").strip() == "1"
    return _enabled


# ---------------------------------------------------------------------------
# OTel singletons (populated by ``_init`` on first real use)
# ---------------------------------------------------------------------------
_tracer: Any = None
_meter: Any = None

# Metric instruments – created once in ``_init``.
_instruments: dict[str, Any] = {}


def _init() -> None:  # noqa: C901 – one-time setup, complexity is fine
    """Bootstrap OTel providers and create all metric instruments."""
    global _initialised, _tracer, _meter, _instruments

    if _initialised:
        return
    _initialised = True

    if not is_enabled():
        return

    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "VF_TELEMETRY=1 but opentelemetry packages are not installed. "
            "Install with:  uv pip install 'verifiers[telemetry]'"
        )
        _initialised = False
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "verifiers")
    resource = Resource.create({"service.name": service_name})

    # -- Traces --
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    use_console = os.getenv("VF_TELEMETRY_CONSOLE", "").strip() == "1"
    if use_console:
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        tp.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    otel_trace.set_tracer_provider(tp)
    _tracer = otel_trace.get_tracer("verifiers")

    # -- Metrics --
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(), export_interval_millis=5000
    )
    readers = [reader]

    if use_console:
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader as PMR,
        )

        readers.append(PMR(ConsoleMetricExporter(), export_interval_millis=10000))

    mp = MeterProvider(resource=resource, metric_readers=readers)
    otel_metrics.set_meter_provider(mp)
    _meter = otel_metrics.get_meter("verifiers")

    # ---------------------------------------------------------------
    # Create instruments
    # ---------------------------------------------------------------

    # -- Rollout pipeline --
    _instruments["rollout_duration"] = _meter.create_histogram(
        "vf.rollout.duration",
        unit="s",
        description="Rollout wall-clock duration in seconds",
    )
    _instruments["rollout_count"] = _meter.create_counter(
        "vf.rollout.count",
        description="Number of rollouts started/completed/errored",
    )
    _instruments["rollout_active"] = _meter.create_up_down_counter(
        "vf.rollout.active",
        description="Currently active rollouts",
    )
    _instruments["group_duration"] = _meter.create_histogram(
        "vf.group.duration",
        unit="s",
        description="Group wall-clock duration in seconds",
    )
    _instruments["group_count"] = _meter.create_counter(
        "vf.group.count",
        description="Number of groups completed",
    )
    _instruments["idle_duration"] = _meter.create_histogram(
        "vf.idle.duration",
        unit="s",
        description="Time between rollout completions (non-generation idle)",
    )
    _instruments["scoring_duration"] = _meter.create_histogram(
        "vf.scoring.duration",
        unit="s",
        description="Scoring duration in seconds",
    )
    _instruments["setup_duration"] = _meter.create_histogram(
        "vf.setup.duration",
        unit="s",
        description="Setup duration in seconds",
    )
    _instruments["generate_duration"] = _meter.create_histogram(
        "vf.generate.duration",
        unit="s",
        description="Full generate() call duration",
    )

    # -- Token usage --
    _instruments["tokens_input"] = _meter.create_counter(
        "vf.tokens.input",
        description="Input tokens consumed",
    )
    _instruments["tokens_output"] = _meter.create_counter(
        "vf.tokens.output",
        description="Output tokens consumed",
    )
    _instruments["tokens_total"] = _meter.create_counter(
        "vf.tokens.total",
        description="Total tokens consumed",
    )

    # -- Model / Client --
    _instruments["model_request_duration"] = _meter.create_histogram(
        "vf.model.request.duration",
        unit="s",
        description="Model request latency in seconds",
    )
    _instruments["model_request_count"] = _meter.create_counter(
        "vf.model.request.count",
        description="Number of model requests",
    )
    _instruments["model_request_error"] = _meter.create_counter(
        "vf.model.request.error",
        description="Number of model request errors",
    )

    # -- Browser --
    _instruments["browser_session_created"] = _meter.create_counter(
        "vf.browser.session.created",
        description="Browser sessions created",
    )
    _instruments["browser_session_destroyed"] = _meter.create_counter(
        "vf.browser.session.destroyed",
        description="Browser sessions destroyed",
    )
    _instruments["browser_session_active"] = _meter.create_up_down_counter(
        "vf.browser.session.active",
        description="Currently active browser sessions",
    )
    _instruments["browser_action_count"] = _meter.create_counter(
        "vf.browser.action.count",
        description="Browser actions executed",
    )
    _instruments["browser_action_duration"] = _meter.create_histogram(
        "vf.browser.action.duration",
        unit="s",
        description="Browser action duration in seconds",
    )
    _instruments["browser_action_error"] = _meter.create_counter(
        "vf.browser.action.error",
        description="Browser action errors",
    )
    _instruments["browser_retry_count"] = _meter.create_counter(
        "vf.browser.retry.count",
        description="Browser operation retries",
    )
    _instruments["sandbox_created"] = _meter.create_counter(
        "vf.sandbox.created",
        description="Sandboxes created",
    )
    _instruments["sandbox_deleted"] = _meter.create_counter(
        "vf.sandbox.deleted",
        description="Sandboxes deleted",
    )
    _instruments["sandbox_error"] = _meter.create_counter(
        "vf.sandbox.error",
        description="Sandbox errors",
    )

    # -- Tool calls --
    _instruments["tool_call_count"] = _meter.create_counter(
        "vf.tool.call.count",
        description="Tool calls executed",
    )
    _instruments["tool_call_duration"] = _meter.create_histogram(
        "vf.tool.call.duration",
        unit="s",
        description="Tool call duration in seconds",
    )
    _instruments["tool_call_error"] = _meter.create_counter(
        "vf.tool.call.error",
        description="Tool call errors",
    )

    # -- Errors --
    _instruments["error_count"] = _meter.create_counter(
        "vf.error.count",
        description="All errors by category",
    )

    # -- Turns --
    _instruments["turn_count"] = _meter.create_counter(
        "vf.env.turn.count",
        description="Turns in multi-turn environments",
    )

    logger.info("Verifiers telemetry initialised (OTLP exporter active)")


# ---------------------------------------------------------------------------
# Public helpers – always safe to call (no-op when disabled)
# ---------------------------------------------------------------------------


def record_counter(
    name: str, value: int = 1, attributes: dict[str, Any] | None = None
) -> None:
    """Increment a counter metric."""
    if not is_enabled():
        return
    _init()
    inst = _instruments.get(name)
    if inst is not None:
        inst.add(value, attributes or {})


def record_histogram(
    name: str, value: float, attributes: dict[str, Any] | None = None
) -> None:
    """Record a value into a histogram metric."""
    if not is_enabled():
        return
    _init()
    inst = _instruments.get(name)
    if inst is not None:
        inst.record(value, attributes or {})


def record_up_down(
    name: str, value: int = 1, attributes: dict[str, Any] | None = None
) -> None:
    """Adjust an up-down counter (gauge-like)."""
    if not is_enabled():
        return
    _init()
    inst = _instruments.get(name)
    if inst is not None:
        inst.add(value, attributes or {})


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Context manager that creates an OTel span (or no-op when disabled)."""
    if not is_enabled():
        yield None
        return
    _init()
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name, attributes=attributes or {}) as s:
        yield s


@contextmanager
def timed(
    histogram_name: str,
    counter_name: str | None = None,
    error_counter_name: str | None = None,
    span_name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Combined timing context: records histogram, bumps counter, creates span.

    Yields a mutable dict so the caller can add attributes (e.g. error info)
    after the block runs.

    Usage::

        with timed("rollout_duration", counter_name="rollout_count",
                    span_name="rollout", attributes={"env_id": "foo"}) as ctx:
            ...run rollout...
            ctx["status"] = "completed"
    """
    ctx: dict[str, Any] = {"status": "ok"}
    attrs = dict(attributes or {})
    t0 = time.monotonic()

    if not is_enabled():
        yield ctx
        return

    _init()

    otel_span = None
    tok = None
    _otel_context = None
    _otel_trace = None
    if span_name and _tracer:
        from opentelemetry import context as otel_context
        from opentelemetry import trace as otel_trace

        _otel_context = otel_context
        _otel_trace = otel_trace
        otel_span = _tracer.start_span(span_name, attributes=attrs)
        tok = otel_context.attach(otel_trace.set_span_in_context(otel_span))

    try:
        yield ctx
    except Exception as exc:
        ctx["status"] = "error"
        ctx["error_type"] = type(exc).__name__
        if error_counter_name:
            record_counter(
                error_counter_name,
                attributes={**attrs, "error_type": type(exc).__name__},
            )
        if otel_span is not None and _otel_trace is not None:
            otel_span.set_status(_otel_trace.StatusCode.ERROR, str(exc))
            otel_span.record_exception(exc)
        raise
    finally:
        elapsed = time.monotonic() - t0
        merged = {
            **attrs,
            **{k: v for k, v in ctx.items() if k not in ("status",) or True},
        }
        record_histogram(histogram_name, elapsed, merged)
        if counter_name:
            record_counter(counter_name, attributes=merged)
        if otel_span is not None:
            for k, v in ctx.items():
                if isinstance(v, (str, int, float, bool)):
                    otel_span.set_attribute(f"vf.{k}", v)
            otel_span.end()
        if tok is not None and _otel_context is not None:
            _otel_context.detach(tok)


def record_tokens(
    input_tokens: float, output_tokens: float, attributes: dict[str, Any] | None = None
) -> None:
    """Record token usage."""
    if not is_enabled():
        return
    _init()
    attrs = attributes or {}
    total = input_tokens + output_tokens
    inst_in = _instruments.get("tokens_input")
    inst_out = _instruments.get("tokens_output")
    inst_total = _instruments.get("tokens_total")
    if inst_in is not None:
        inst_in.add(int(input_tokens), attrs)
    if inst_out is not None:
        inst_out.add(int(output_tokens), attrs)
    if inst_total is not None:
        inst_total.add(int(total), attrs)


def record_error(
    category: str,
    error: BaseException | None = None,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record an error with a category label for the dashboard."""
    if not is_enabled():
        return
    _init()
    attrs = dict(attributes or {})
    attrs["category"] = category
    if error is not None:
        attrs["error_type"] = type(error).__name__
    inst = _instruments.get("error_count")
    if inst is not None:
        inst.add(1, attrs)
