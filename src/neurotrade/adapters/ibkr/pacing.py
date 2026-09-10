"""Staying inside IBKR's historical data rate limit.

IBKR allows roughly **60 historical requests per 10 minutes**. Exceed it and the
response is not an error you can retry immediately — it is a pacing violation
that locks further historical requests out for a while. A crawler that discovers
the limit by hitting it makes the multi-week backfill slower than one that never
approaches it.

`MarketDataPort` says implementations pace themselves rather than letting callers
find the limit, so this lives in the adapter.

**Time comes from a `Clock`, and waiting is the caller's job.** The pacer says
how long to wait; it does not sleep. That keeps it testable without real delays
and keeps the no-wall-clock invariant intact — a component that called
`time.sleep` would be untestable and would read the OS clock.
"""

from __future__ import annotations

from collections import deque

from neurotrade.core.clock import Clock, Nanos

__all__ = ["HistoricalPacer"]

_NS_PER_SECOND = 1_000_000_000

DEFAULT_MAX_REQUESTS = 55
"""Under IBKR's documented 60, because the limit is enforced on their clock and
ours drifts. Five requests of headroom costs about eight percent of throughput
and avoids a lockout that costs minutes."""

DEFAULT_WINDOW_SECONDS = 600
"""Ten minutes, as documented."""


class HistoricalPacer:
    """Tracks recent requests and says how long to wait before the next.

    A sliding window rather than a fixed one: IBKR counts the last ten minutes
    continuously, so resetting a counter every ten minutes would allow 110
    requests across a boundary and trip the limit.

    Example:
        >>> from neurotrade.core.clock import SimClock
        >>> clock = SimClock(0)
        >>> pacer = HistoricalPacer(clock, max_requests=2, window_seconds=10)
        >>> pacer.record(); pacer.record()
        >>> pacer.wait_seconds()          # window is full; wait for the oldest to age out
        10.0
    """

    __slots__ = ("_clock", "_max_requests", "_sent", "_window_ns")

    def __init__(
        self,
        clock: Clock,
        *,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        """Create a pacer.

        Args:
            clock: Supplies the current time. `LiveClock` in production.
            max_requests: Requests permitted per window.
            window_seconds: Length of the sliding window.

        Raises:
            ValueError: If either bound is not positive, which would either
                block every request or permit unlimited ones.
        """
        if max_requests < 1:
            raise ValueError(f"max_requests must be at least 1, got {max_requests}")
        if window_seconds < 1:
            raise ValueError(f"window_seconds must be at least 1, got {window_seconds}")
        self._clock = clock
        self._max_requests = max_requests
        self._window_ns = window_seconds * _NS_PER_SECOND
        self._sent: deque[Nanos] = deque()

    def record(self) -> None:
        """Note that a request was just sent.

        Example:
            >>> from neurotrade.core.clock import SimClock
            >>> pacer = HistoricalPacer(SimClock(0), max_requests=5)
            >>> pacer.record()
            >>> pacer.in_window
            1
        """
        self._sent.append(self._clock.now_ns())

    def wait_seconds(self) -> float:
        """How long to wait before the next request may be sent.

        Returns:
            Zero when there is room in the window, otherwise the seconds until
            the oldest request ages out of it.

        Example:
            >>> from neurotrade.core.clock import SimClock
            >>> HistoricalPacer(SimClock(0), max_requests=5).wait_seconds()
            0.0
        """
        self._expire()
        if len(self._sent) < self._max_requests:
            return 0.0
        oldest = self._sent[0]
        remaining = (oldest + self._window_ns) - self._clock.now_ns()
        return max(0.0, remaining / _NS_PER_SECOND)

    @property
    def in_window(self) -> int:
        """Requests counted in the current window.

        Reported so a crawler can log headroom, which is the difference between
        noticing pacing pressure and discovering it as a lockout.
        """
        self._expire()
        return len(self._sent)

    @property
    def headroom(self) -> int:
        """Requests that could be sent right now without waiting."""
        return max(0, self._max_requests - self.in_window)

    def _expire(self) -> None:
        """Drop requests that have aged out of the window."""
        cutoff = self._clock.now_ns() - self._window_ns
        while self._sent and self._sent[0] <= cutoff:
            self._sent.popleft()

    def __repr__(self) -> str:
        return f"HistoricalPacer({self.in_window}/{self._max_requests} in window)"
