"""The system's only source of time.

Every component takes time from a `Clock`. Nothing calls `datetime.now()`,
`time.time()` or `time.time_ns()` directly — those live here and nowhere else,
which is what makes a session replayable bit-for-bit (gate G1). A single stray
wall-clock read in a strategy or a feature turns a deterministic replay into a
run that quietly differs every time, and the resulting digest mismatch is
extremely hard to trace back to its cause.

**Time is integer nanoseconds since the Unix epoch, UTC.** Not `datetime`, which
holds only microseconds — market data routinely carries nanosecond timestamps,
and two events inside the same microsecond must still order deterministically.
`datetime` is available through `now()` for session-boundary logic and display,
where microsecond truncation is harmless.

`SimClock` is monotonic by construction: it refuses to move backwards. Replay
feeds events in `(ts_event, seq)` order, so a clock that could rewind would let
a component observe a future it should not yet know about.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

__all__ = [
    "Clock",
    "LiveClock",
    "Nanos",
    "SimClock",
    "to_datetime",
    "to_nanos",
]

# Nanoseconds since the Unix epoch, UTC. The canonical timestamp everywhere.
type Nanos = int

_NS_PER_SECOND = 1_000_000_000
_NS_PER_MICROSECOND = 1_000
_SECONDS_PER_DAY = 86_400

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# Both conversions below use integer arithmetic throughout. Routing via
# `datetime.timestamp()` or `datetime.fromtimestamp()` passes through a float,
# and epoch nanoseconds need 19 significant digits where float64 offers about
# 16 — which introduces errors of roughly +/-100 ns. That is small enough to
# look like nothing and large enough to reorder two events, break a round trip,
# and make a replay digest differ between machines.


def to_datetime(ns: Nanos) -> datetime:
    """Convert epoch nanoseconds to a timezone-aware UTC datetime.

    Lossy below microseconds — `datetime` cannot hold nanoseconds. The
    sub-microsecond remainder is **truncated, not rounded**, so the result is
    always the instant at or before `ns`. Rounding could push an event across a
    session boundary it had not actually reached.

    Use for session logic and display; never for ordering or equality.
    """
    seconds, remainder = divmod(ns, _NS_PER_SECOND)
    return _EPOCH + timedelta(seconds=seconds, microseconds=remainder // _NS_PER_MICROSECOND)


def to_nanos(moment: datetime) -> Nanos:
    """Convert a timezone-aware datetime to epoch nanoseconds. Exact.

    Naive datetimes are rejected. A naive timestamp is ambiguous by definition,
    and guessing at it silently shifts every bar by the local UTC offset — which
    on this machine would be hours.
    """
    if moment.tzinfo is None:
        raise ValueError(f"datetime must be timezone-aware: {moment!r}")
    delta = moment - _EPOCH
    return (
        delta.days * _SECONDS_PER_DAY + delta.seconds
    ) * _NS_PER_SECOND + delta.microseconds * _NS_PER_MICROSECOND


@runtime_checkable
class Clock(Protocol):
    """A source of the current time.

    Implemented by `LiveClock` in production and `SimClock` in backtest and
    replay. Components depend on this protocol, never on a concrete clock, so
    the same code runs in both without modification (§3.6).
    """

    def now_ns(self) -> Nanos:
        """Current time as epoch nanoseconds. The authoritative reading."""
        ...

    def now(self) -> datetime:
        """Current time as a timezone-aware UTC datetime. Microsecond precision."""
        ...


class LiveClock:
    """Wall-clock time. The only place in the system that reads the OS clock."""

    __slots__ = ()

    def now_ns(self) -> Nanos:
        return time.time_ns()

    def now(self) -> datetime:
        return datetime.now(UTC)


class SimClock:
    """A clock advanced explicitly by the caller.

    Used by the replay engine and by every test that touches time. Monotonic:
    moving it backwards raises rather than silently reordering causality.
    """

    __slots__ = ("_now_ns",)

    def __init__(self, start: Nanos | datetime = 0) -> None:
        self._now_ns: Nanos = to_nanos(start) if isinstance(start, datetime) else start

    def now_ns(self) -> Nanos:
        return self._now_ns

    def now(self) -> datetime:
        return to_datetime(self._now_ns)

    def set_time_ns(self, ns: Nanos) -> None:
        """Move the clock to an absolute time. Must not go backwards."""
        if ns < self._now_ns:
            raise ValueError(f"clock cannot move backwards: {ns} is before current {self._now_ns}")
        self._now_ns = ns

    def set_time(self, moment: datetime) -> None:
        self.set_time_ns(to_nanos(moment))

    def advance_ns(self, delta: int) -> None:
        """Move the clock forward by `delta` nanoseconds."""
        if delta < 0:
            raise ValueError(f"cannot advance by a negative duration: {delta}")
        self._now_ns += delta

    def __repr__(self) -> str:
        return f"SimClock({self._now_ns})"
