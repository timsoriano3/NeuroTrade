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

type Nanos = int
"""Nanoseconds since the Unix epoch, UTC. The canonical timestamp everywhere in
the system — every `ts_event`, `ts_init` and duration is one of these."""

_NS_PER_SECOND = 1_000_000_000
_NS_PER_MICROSECOND = 1_000
_SECONDS_PER_DAY = 86_400

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)  # the zero point for both conversions

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

    Args:
        ns: Nanoseconds since the Unix epoch. May be negative (pre-1970).

    Returns:
        The same instant as a UTC-aware `datetime`, truncated to microseconds.

    Example:
        >>> to_datetime(1_773_495_000_000_000_000)
        datetime.datetime(2026, 3, 14, 13, 30, tzinfo=datetime.timezone.utc)
    """
    seconds, remainder = divmod(ns, _NS_PER_SECOND)
    return _EPOCH + timedelta(seconds=seconds, microseconds=remainder // _NS_PER_MICROSECOND)


def to_nanos(moment: datetime) -> Nanos:
    """Convert a timezone-aware datetime to epoch nanoseconds. Exact.

    Args:
        moment: A timezone-aware `datetime`. Naive values are rejected, because
            a naive timestamp is ambiguous by definition and guessing at it
            silently shifts every bar by the local UTC offset.

    Returns:
        Nanoseconds since the Unix epoch.

    Raises:
        ValueError: If `moment` has no timezone attached.

    Example:
        >>> to_nanos(datetime(2026, 3, 14, 13, 30, tzinfo=UTC))
        1773495000000000000
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

    Example:
        >>> isinstance(SimClock(0), Clock)
        True
    """

    def now_ns(self) -> Nanos:
        """Current time as epoch nanoseconds. The authoritative reading."""
        ...

    def now(self) -> datetime:
        """Current time as a timezone-aware UTC datetime. Microsecond precision."""
        ...


class LiveClock:
    """Wall-clock time. The only place in the system that reads the OS clock.

    Example:
        >>> LiveClock().now().tzinfo
        datetime.timezone.utc
    """

    __slots__ = ()

    def now_ns(self) -> Nanos:
        """Read the OS clock in nanoseconds.

        Returns:
            Nanoseconds since the Unix epoch, at the resolution the OS provides.
        """
        return time.time_ns()

    def now(self) -> datetime:
        """Read the OS clock as a UTC-aware datetime.

        Returns:
            The current instant, truncated to microseconds.
        """
        return datetime.now(UTC)


class SimClock:
    """A clock advanced explicitly by the caller.

    Used by the replay engine and by every test that touches time. Monotonic:
    moving it backwards raises rather than silently reordering causality.

    Example:
        >>> clock = SimClock(1_000)
        >>> clock.advance_ns(500)
        >>> clock.now_ns()
        1500
    """

    __slots__ = ("_now_ns",)  # keeps the object small; millions are created in replay

    def __init__(self, start: Nanos | datetime = 0) -> None:
        """Create a clock stopped at a given instant.

        Args:
            start: Where the clock begins, either as epoch nanoseconds or a
                timezone-aware `datetime`. Defaults to the epoch.

        Raises:
            ValueError: If `start` is a naive `datetime`.

        Example:
            >>> SimClock(datetime(2026, 3, 14, 13, 30, tzinfo=UTC)).now_ns()
            1773495000000000000
        """
        self._now_ns: Nanos = to_nanos(start) if isinstance(start, datetime) else start

    def now_ns(self) -> Nanos:
        """Current simulated time in nanoseconds.

        Returns the same value on every call until the clock is advanced, which
        is precisely the property that makes replay reproducible.

        Example:
            >>> clock = SimClock(42)
            >>> (clock.now_ns(), clock.now_ns())
            (42, 42)
        """
        return self._now_ns

    def now(self) -> datetime:
        """Current simulated time as a UTC-aware datetime.

        Example:
            >>> SimClock(1_773_495_000_000_000_000).now()
            datetime.datetime(2026, 3, 14, 13, 30, tzinfo=datetime.timezone.utc)
        """
        return to_datetime(self._now_ns)

    def set_time_ns(self, ns: Nanos) -> None:
        """Move the clock to an absolute time.

        Args:
            ns: The instant to move to, in epoch nanoseconds. Must not be
                earlier than the current time; equal is allowed, since several
                events can share a timestamp.

        Raises:
            ValueError: If `ns` is before the current time. Rewinding would let
                a component observe a future it should not yet know about.

        Example:
            >>> clock = SimClock(1_000)
            >>> clock.set_time_ns(2_000)
            >>> clock.now_ns()
            2000
        """
        if ns < self._now_ns:
            raise ValueError(f"clock cannot move backwards: {ns} is before current {self._now_ns}")
        self._now_ns = ns

    def set_time(self, moment: datetime) -> None:
        """Move the clock to an absolute time given as a datetime.

        Args:
            moment: A timezone-aware `datetime` at or after the current time.

        Raises:
            ValueError: If `moment` is naive, or is before the current time.

        Example:
            >>> clock = SimClock(0)
            >>> clock.set_time(datetime(2026, 3, 14, 13, 30, tzinfo=UTC))
            >>> clock.now_ns()
            1773495000000000000
        """
        self.set_time_ns(to_nanos(moment))

    def advance_ns(self, delta: int) -> None:
        """Move the clock forward by a duration.

        Args:
            delta: Nanoseconds to advance. Zero is allowed; negative is not.

        Raises:
            ValueError: If `delta` is negative.

        Example:
            >>> clock = SimClock(0)
            >>> clock.advance_ns(60_000_000_000)     # one minute
            >>> clock.now_ns()
            60000000000
        """
        if delta < 0:
            raise ValueError(f"cannot advance by a negative duration: {delta}")
        self._now_ns += delta

    def __repr__(self) -> str:
        return f"SimClock({self._now_ns})"
