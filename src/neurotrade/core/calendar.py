"""Trading calendars — which days a venue traded, and between which times.

**This is not in `TRADER_PLAN.md`.** §12.1 goes straight from "build the store"
to "start the backfill crawler" and never defines trading hours, holidays or
half days. The code found the gap on its own: `DuckDBCatalog.missing_sessions`
takes the expected sessions as an argument because deriving them "requires an
exchange calendar … and that is not storage's business", and `ports.py`,
`cli.py`, `schemas.py` and `IbkrMarketData` all defer to "the venue calendar" in
comments. Nothing could supply it. This module is that missing piece.

Why it blocks everything else: a crawler cannot know what to fetch without
knowing which days should have data, and a corpus cannot be called complete
without knowing what complete means. "Every weekday" is wrong often enough to
matter — roughly ten holidays a year per venue, plus half days, plus the days a
venue closed and its neighbour did not. TSX trades through US Thanksgiving;
NYSE trades through Canada Day.

What lives here is only the *shape* of a session. Which dates are sessions, and
what their bounds are, is a fact about the outside world and therefore an
adapter's job — see `CalendarPort` in `neurotrade.core.ports`.

Scope: the regular session only. Pre- and post-market bounds are deliberately
absent until a feed actually delivers pre/post bars; `MarketSession` in
`core.events` already names those phases for when it does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from neurotrade.core.clock import Nanos
from neurotrade.core.events import BarInterval
from neurotrade.core.types import Venue

__all__ = ["TradingSession"]

# A session longer than this is a bug in the adapter, not an unusual day. The
# longest real regular session on any venue here is 6.5 hours; the bound is
# loose on purpose so that it catches nonsense (a swapped open and close, a
# date parsed as an epoch) without ever arguing with a genuine schedule.
_MAX_SESSION_NS = 24 * 60 * 60 * 1_000_000_000


@dataclass(frozen=True, slots=True, kw_only=True)
class TradingSession:
    """One venue trading day, with the bounds that day actually had.

    Half days are ordinary instances with a shorter span rather than a special
    case, so callers that compute from `open_ns` and `close_ns` handle them
    correctly without knowing they exist. `is_early_close` is there for
    reporting and for tests, not as something callers must branch on.

    Example:
        >>> from datetime import UTC, datetime
        >>> from neurotrade.core.clock import to_nanos
        >>> july_third = TradingSession(
        ...     venue=Venue.NASDAQ,
        ...     session_date=date(2024, 7, 3),
        ...     open_ns=to_nanos(datetime(2024, 7, 3, 13, 30, tzinfo=UTC)),
        ...     close_ns=to_nanos(datetime(2024, 7, 3, 17, 0, tzinfo=UTC)),
        ...     is_early_close=True,
        ... )
        >>> july_third.expected_bars(BarInterval.MIN_1)
        210
    """

    venue: Venue  # whose calendar this came from; venues differ on holidays
    session_date: date  # the date the venue labels this session, in venue-local terms
    open_ns: Nanos  # regular session open, UTC nanoseconds
    close_ns: Nanos  # regular session close, UTC nanoseconds
    is_early_close: bool  # a half day — shorter span, same handling

    def __post_init__(self) -> None:
        if self.venue is Venue.SMART:
            raise ValueError("SMART is an order route, not a venue with a calendar")
        if self.close_ns <= self.open_ns:
            raise ValueError(
                f"session must close after it opens: {self.open_ns} -> {self.close_ns}"
            )
        if self.duration_ns > _MAX_SESSION_NS:
            raise ValueError(f"session spans more than a day: {self.duration_ns} ns")

    @property
    def duration_ns(self) -> int:
        """Length of the regular session in nanoseconds.

        Example:
            >>> from datetime import UTC, datetime
            >>> from neurotrade.core.clock import to_nanos
            >>> TradingSession(
            ...     venue=Venue.NYSE,
            ...     session_date=date(2024, 7, 8),
            ...     open_ns=to_nanos(datetime(2024, 7, 8, 13, 30, tzinfo=UTC)),
            ...     close_ns=to_nanos(datetime(2024, 7, 8, 20, 0, tzinfo=UTC)),
            ...     is_early_close=False,
            ... ).duration_ns // 1_000_000_000
            23400
        """
        return self.close_ns - self.open_ns

    def expected_bars(self, interval: BarInterval) -> int:
        """How many bars a complete session holds at this interval.

        390 one-minute bars in a full US regular session, 210 in a half day.
        Deriving it from the session's own bounds rather than hardcoding 390 is
        what makes early closes and venue differences come out right.

        This is the argument `Coverage.is_complete` has taken since it was
        written without having a source for it — `Coverage` lives in
        `adapters.storage.duckdb_catalog`, so the wiring is the caller's job,
        not an import from here. Note it compares with `>=` rather than `==`,
        to tolerate a session that also holds extended-hours bars.

        Args:
            interval: Bar size.

        Returns:
            Whole bars fitting between open and close. A partial trailing bar
            is not counted — it would not be a complete bar of that interval.
            An interval at least as long as the session gives **one**: a daily
            bar summarises the session it spans, however short that session
            was.

        Example:
            >>> from datetime import UTC, datetime
            >>> from neurotrade.core.clock import to_nanos
            >>> full_day = TradingSession(
            ...     venue=Venue.NYSE,
            ...     session_date=date(2024, 7, 8),
            ...     open_ns=to_nanos(datetime(2024, 7, 8, 13, 30, tzinfo=UTC)),
            ...     close_ns=to_nanos(datetime(2024, 7, 8, 20, 0, tzinfo=UTC)),
            ...     is_early_close=False,
            ... )
            >>> full_day.expected_bars(BarInterval.MIN_1)
            390
            >>> full_day.expected_bars(BarInterval.MIN_30)
            13
            >>> full_day.expected_bars(BarInterval.DAY_1)
            1
        """
        # Floor division alone answers 0 for a daily bar, because a 6.5-hour
        # session does not contain a 24-hour one. `Coverage.is_complete` reads
        # that as "nothing missing", so every daily cell would look filled and
        # the crawler would fetch nothing at all (§12.1 stage 3).
        if interval.nanos >= self.duration_ns:
            return 1
        return self.duration_ns // interval.nanos

    def contains(self, ts: Nanos) -> bool:
        """Whether the venue was open at this instant.

        Inclusive at both ends. For deciding which session a **bar** belongs
        to, use `holds_bar` instead — the two differ at the open, and the
        difference is a real bug rather than a nicety.

        Args:
            ts: An instant, UTC nanoseconds.

        Example:
            >>> from datetime import UTC, datetime
            >>> from neurotrade.core.clock import to_nanos
            >>> session = TradingSession(
            ...     venue=Venue.NYSE,
            ...     session_date=date(2024, 7, 8),
            ...     open_ns=to_nanos(datetime(2024, 7, 8, 13, 30, tzinfo=UTC)),
            ...     close_ns=to_nanos(datetime(2024, 7, 8, 20, 0, tzinfo=UTC)),
            ...     is_early_close=False,
            ... )
            >>> session.contains(to_nanos(datetime(2024, 7, 8, 12, 0, tzinfo=UTC)))
            False
            >>> session.contains(session.open_ns)
            True
        """
        return self.open_ns <= ts <= self.close_ns

    def holds_bar(self, bar_ts_event: Nanos) -> bool:
        """Whether a bar closing at this instant belongs to this session.

        Open-exclusive, close-inclusive — and that asymmetry is the whole point
        of having a second method. `Bar.ts_event` is the bar's **close**, so the
        first bar of a US session closes at 09:31 and the last closes at exactly
        16:00. A bar stamped 09:30 closed *at* the open, which means it covers
        the minute before the bell and belongs to the previous session. Treating
        it as the first bar of this one shifts every session by a bar and
        quietly corrupts the opening range.

        Args:
            bar_ts_event: A bar's `ts_event`, i.e. its close, UTC nanoseconds.

        Example:
            >>> from datetime import UTC, datetime
            >>> from neurotrade.core.clock import to_nanos
            >>> session = TradingSession(
            ...     venue=Venue.NYSE,
            ...     session_date=date(2024, 7, 8),
            ...     open_ns=to_nanos(datetime(2024, 7, 8, 13, 30, tzinfo=UTC)),
            ...     close_ns=to_nanos(datetime(2024, 7, 8, 20, 0, tzinfo=UTC)),
            ...     is_early_close=False,
            ... )
            >>> session.holds_bar(session.open_ns)  # closed at the bell
            False
            >>> session.holds_bar(session.close_ns)  # the final bar
            True
        """
        return self.open_ns < bar_ts_event <= self.close_ns
