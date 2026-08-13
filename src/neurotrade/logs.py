"""Structured logging.

Every log line is a record with fields, not a sentence with values glued into
it. That is what makes "show me every rejected order for this strategy on this
day" a query rather than a grep, and it is why §11 specifies structured JSON
logs rather than formatted text.

**Named ``logs`` rather than ``logging``** so that nothing in this package can
shadow the standard library module for a reader skimming imports. Python 3's
absolute imports would make ``logging.py`` safe, but not obvious.

**Every line carries the run context**: the run id, the profile, and the config
hash in force. Six months later, a line in a log file is enough to identify
exactly which configuration produced it — which is the operational half of
§6.3's reconstructability requirement, the other half being the trade journal.

**Logs are diagnostics, not the audit trail.** The event store is the record of
what happened; these are the record of what the process was doing. A log line is
allowed to carry a wall-clock timestamp during a replay, because it describes
the replay run rather than the session being replayed. Nothing may read a log
line back as an input — that would make behaviour depend on the log level.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO, cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor

from neurotrade.config import Profile, Settings, config_hash
from neurotrade.core.clock import Clock
from neurotrade.core.ids import RunId

__all__ = [
    "bind",
    "clear_context",
    "configure",
    "get_logger",
]


def _renderer(profile: Profile) -> Processor:
    """Pick the output format for a profile.

    Research is read by a person while they work, so it gets aligned,
    colourised console output. Paper and live are read by a machine — Grafana,
    Loki, or a grep three weeks later — so they get JSON.

    Args:
        profile: The running profile.

    Returns:
        A structlog renderer processor.
    """
    if profile is Profile.RESEARCH:
        return structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    return structlog.processors.JSONRenderer(sort_keys=True)


def configure(
    settings: Settings,
    clock: Clock,
    *,
    stream: TextIO | None = None,
) -> RunId:
    """Set up logging for this process and return the run identifier.

    Called once at startup, before anything logs. Binds the run context so that
    every subsequent line carries it without callers having to remember.

    Args:
        settings: Resolved settings. Supplies the level, the renderer choice and
            the config hash.
        clock: The run's clock. Its `now_ns()` seeds the run id — which is why a
            replay under a `SimClock` produces identical logs every time, while
            a live run under `LiveClock` is unique without needing randomness.
        stream: Where to write. Defaults to stderr, so that stdout stays clean
            for command output and can be piped without log noise.

    Returns:
        The `RunId` for this run, already bound into the logging context.

    Example:
        >>> from neurotrade.core.clock import SimClock
        >>> from neurotrade.config import load_settings
        >>> run_id = configure(load_settings("research"), SimClock(1_000))
        >>> str(run_id).startswith("run_")
        True
    """
    fingerprint = config_hash(settings)
    run_id = RunId.derive(config_hash=fingerprint, started_ns=clock.now_ns())

    logging.basicConfig(
        format="%(message)s",
        stream=stream or sys.stderr,
        level=settings.log_level.upper(),
        force=True,  # replaces any handler a library installed at import time
    )

    structlog.configure(
        processors=[
            # Context bound via `bind` rides on contextvars, so it survives
            # across await points without being threaded through call sites.
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Wall clock on purpose: this timestamps the run, not the session
            # being replayed. Simulated time is bound explicitly where it matters.
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _renderer(settings.profile),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(stream or sys.stderr),
        cache_logger_on_first_use=True,
    )

    clear_context()
    bind(run_id=str(run_id), profile=settings.profile.value, config_hash=fingerprint)
    return run_id


def bind(**values: object) -> None:
    """Add fields to every subsequent log line in this context.

    Uses context variables, so bound values survive across `await` boundaries
    and do not have to be passed down through call signatures.

    Args:
        **values: Fields to attach, e.g. `symbol="AAPL.NASDAQ"`.

    Example:
        >>> bind(symbol="AAPL.NASDAQ")
        >>> clear_context()
    """
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    """Drop all bound fields.

    Called between runs and between tests. Without it, context from one session
    leaks into the next, which is how a log line ends up claiming the wrong
    config hash.
    """
    structlog.contextvars.clear_contextvars()


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a logger for a module.

    Args:
        name: Usually `__name__`, so lines are attributable to a module.

    Returns:
        A bound logger. Call it with an event name and fields —
        `log.info("order_submitted", order_id=..., symbol=...)` — rather than a
        formatted sentence, so the fields stay queryable.

    Example:
        >>> log = get_logger(__name__)
        >>> hasattr(log, "info")
        True
    """
    # structlog.get_logger is untyped. The cast is sound because `configure`
    # sets wrapper_class to a filtering bound logger; it would be a lie if
    # someone called structlog.configure directly with a different class.
    return cast(FilteringBoundLogger, structlog.get_logger(name))
