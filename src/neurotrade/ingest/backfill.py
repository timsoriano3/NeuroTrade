"""The backfill work queue: which instrument-sessions the corpus is missing.

§12.1 stage 1 calls for a crawler that is "pacing-aware, resumable, runs
continuously". This module is the *resumable* half, and it gets there by having
no state of its own. The queue is recomputed from two facts that already exist:
which days each venue traded, and how many bars are on disk for each of them.

**The corpus is its own progress record.** There is no checkpoint file to keep
in step, because a checkpoint that disagreed with the data on disk would be
worse than no checkpoint at all. Writes are idempotent, so re-fetching a session
that is already complete costs a request and changes nothing; and a crawler
killed mid-session resumes by observing that the session is short.

**Held is not complete.** An interrupted fetch leaves a session with some of its
bars. Comparing a count against `TradingSession.expected_bars` is what separates
the two, which is why `CatalogPort` returns counts rather than a set of dates.

Known limitation: a session that is *permanently* short — a symbol halted for
the afternoon, or one that listed midway through the day — never reaches its
expected count and is therefore re-offered on every pass. Each pass costs it one
request, so it delays nothing, but it never goes away either. Halt marking is
§12.1 stage 5, timed for Phase 1, and that is the fact this needs in order to
tell a hole from a day that genuinely had fewer bars.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import Nanos
from neurotrade.core.events import BarInterval
from neurotrade.core.ports import CalendarPort, CatalogPort
from neurotrade.core.types import Symbol, Venue
from neurotrade.core.universe import Universe

__all__ = ["BackfillCell", "plan_backfill"]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackfillCell:
    """One unit of crawler work: one instrument, one session, one bar size.

    Carries the session itself rather than just its date, because everything the
    fetch needs — the window to request, how many bars to expect — comes from
    the session's own bounds, and re-deriving them at the point of the request
    is how a half day ends up fetched as a full one.

    Example:
        >>> from datetime import UTC, datetime
        >>> from neurotrade.core.clock import to_nanos
        >>> half_day = TradingSession(
        ...     venue=Venue.NASDAQ,
        ...     session_date=date(2024, 7, 3),
        ...     open_ns=to_nanos(datetime(2024, 7, 3, 13, 30, tzinfo=UTC)),
        ...     close_ns=to_nanos(datetime(2024, 7, 3, 17, 0, tzinfo=UTC)),
        ...     is_early_close=True,
        ... )
        >>> cell = BackfillCell(
        ...     symbol=Symbol("AAPL", Venue.NASDAQ),
        ...     session=half_day,
        ...     interval=BarInterval.MIN_1,
        ...     held_bars=200,
        ... )
        >>> (cell.expected_bars, cell.missing_bars, cell.is_untouched)
        (210, 10, False)
    """

    symbol: Symbol  # instrument to fetch
    session: TradingSession  # the trading day, with the bounds it actually had
    interval: BarInterval  # bar size being filled
    held_bars: int  # bars already on disk for this instrument-session

    def __post_init__(self) -> None:
        """Reject a cell that cannot describe real work.

        Raises:
            ValueError: If the symbol's listing venue is not the session's
                venue, or if the held count is negative.
        """
        if self.symbol.venue is not self.session.venue:
            raise ValueError(
                f"{self.symbol} is not listed on {self.session.venue.value}, whose session this is"
            )
        if self.held_bars < 0:
            raise ValueError(f"held_bars must not be negative: {self.held_bars}")

    @property
    def session_date(self) -> date:
        """The trading day, in the venue's own terms."""
        return self.session.session_date

    @property
    def expected_bars(self) -> int:
        """Bars a complete session holds at this interval."""
        return self.session.expected_bars(self.interval)

    @property
    def missing_bars(self) -> int:
        """How many bars are still absent, floored at zero.

        Floored because a session may legitimately hold *more* than the regular
        session's count once extended-hours bars arrive, and a negative shortfall
        is not a thing.
        """
        return max(self.expected_bars - self.held_bars, 0)

    @property
    def is_untouched(self) -> bool:
        """Whether nothing at all is held for this session.

        Worth distinguishing from merely incomplete: an untouched session is
        almost certainly a fetch that has not happened, while a short one may be
        a halt the corpus cannot yet recognise.
        """
        return self.held_bars == 0

    @property
    def start_ns(self) -> Nanos:
        """Inclusive lower bound for the fetch, the session open."""
        return self.session.open_ns

    @property
    def end_ns(self) -> Nanos:
        """Inclusive upper bound for the fetch, the session close.

        `Bar.ts_event` is a bar's close, so the bars of this session land in
        `(start_ns, end_ns]` — the first one closes an interval after the bell.
        """
        return self.session.close_ns

    def __str__(self) -> str:
        counts = f"{self.held_bars}/{self.expected_bars}"
        return f"{self.symbol} {self.session_date} {self.interval.value} {counts}"


def plan_backfill(
    universe: Universe,
    calendar: CalendarPort,
    catalog: CatalogPort,
    *,
    start: date,
    end: date,
    interval: BarInterval = BarInterval.MIN_1,
) -> Iterator[BackfillCell]:
    """Work the crawler still has to do, most recent sessions first.

    **Order is recency-major, not symbol-major.** Every symbol's most recent
    session is offered before any symbol's older ones, so an interrupted crawl
    leaves a corpus that is shallow across the whole universe rather than deep
    for the alphabetically early part of it. A cross-sectional study needs many
    symbols on the same dates; it cannot use AAPL back to 2021 and nothing else.

    The plan is a **snapshot**. Each symbol's counts are read once, when
    iteration reaches it, so a long-running crawl should re-plan periodically
    rather than hold one iterator open for a week.

    Args:
        universe: Instruments to fill.
        calendar: Which days each venue traded.
        catalog: What the corpus already holds.
        start: First session date to consider, inclusive.
        end: Last session date to consider, inclusive.
        interval: Bar size to fill. §12.1 targets one-minute bars.

    Yields:
        Incomplete cells, ordered by session date descending, then by the
        universe's own symbol order. Complete sessions are absent.

    Raises:
        ValueError: If the range runs backwards, or if the calendar lists a date
            as a session and then denies it — two answers that cannot both be
            right, and a discrepancy that would otherwise be fetched as a
            zero-length window.

    Example:
        >>> from datetime import UTC, datetime
        >>> from neurotrade.core.clock import to_nanos
        >>> july_second = TradingSession(
        ...     venue=Venue.NASDAQ,
        ...     session_date=date(2024, 7, 2),
        ...     open_ns=to_nanos(datetime(2024, 7, 2, 13, 30, tzinfo=UTC)),
        ...     close_ns=to_nanos(datetime(2024, 7, 2, 20, 0, tzinfo=UTC)),
        ...     is_early_close=False,
        ... )
        >>> class OneSessionCalendar:
        ...     def sessions(self, venue, start, end):
        ...         return (july_second.session_date,)
        ...     def session(self, venue, session_date):
        ...         return july_second
        >>> class EmptyCatalog:
        ...     def bar_counts(self, symbol, interval):
        ...         return {}
        >>> plan = plan_backfill(
        ...     Universe([Symbol("AAPL", Venue.NASDAQ)]),
        ...     OneSessionCalendar(),
        ...     EmptyCatalog(),
        ...     start=date(2024, 7, 1),
        ...     end=date(2024, 7, 5),
        ... )
        >>> [str(cell) for cell in plan]
        ['AAPL.NASDAQ 2024-07-02 1m 0/390']
    """
    if end < start:
        raise ValueError(f"backfill range must move forwards: {start} -> {end}")

    sessions_by_venue = {venue: calendar.sessions(venue, start, end) for venue in universe.venues}
    cache: dict[tuple[Venue, date], TradingSession] = {}

    def session_for(venue: Venue, day: date) -> TradingSession:
        cached = cache.get((venue, day))
        if cached is not None:
            return cached
        found = calendar.session(venue, day)
        if found is None:
            raise ValueError(
                f"calendar listed {day} as a {venue.value} session then reported it closed"
            )
        cache[(venue, day)] = found
        return found

    # Held counts are per symbol, so the shortfall has to be worked out symbol
    # by symbol; the interleaving by date happens afterwards, over dates alone.
    pending: dict[Symbol, dict[date, int]] = {}
    for symbol in universe:
        held = catalog.bar_counts(symbol, interval)
        short = {
            day: held.get(day, 0)
            for day in sessions_by_venue[symbol.venue]
            if held.get(day, 0) < session_for(symbol.venue, day).expected_bars(interval)
        }
        if short:
            pending[symbol] = short

    # sorted() of a set, because set iteration order is not reproducible and the
    # order bars are fetched in decides what a half-finished corpus contains.
    days = sorted({day for short in pending.values() for day in short}, reverse=True)

    for day in days:
        for symbol in universe:
            shortfall = pending.get(symbol)
            if shortfall is None or day not in shortfall:
                continue
            yield BackfillCell(
                symbol=symbol,
                session=session_for(symbol.venue, day),
                interval=interval,
                held_bars=shortfall[day],
            )
