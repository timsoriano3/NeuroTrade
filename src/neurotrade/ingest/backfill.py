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

**Planning and fetching use different units.** `plan_backfill` works one session
at a time because that is the grain completeness is decided at. `plan_windows`
then groups a symbol's consecutive missing sessions into one request each, which
is the grain a history feed charges at — IBKR bills a 30-day span of one-minute
bars as a single request. Nothing about resumability changes: the window is
derived from the plan on every pass and holds no state of its own.

Known limitation: a session that is *permanently* short — a symbol halted for
the afternoon, or one that listed midway through the day — never reaches its
expected count and is therefore re-offered on every pass. Windowing makes this
cheaper still, since such a session usually rides along in a request that was
going to be made anyway. Halt marking is §12.1 stage 5, timed for Phase 1, and
that is the fact this needs in order to tell a hole from a day that genuinely
had fewer bars.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date

from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import Nanos
from neurotrade.core.events import BarInterval
from neurotrade.core.ports import CalendarPort, CatalogPort
from neurotrade.core.types import Symbol, Venue
from neurotrade.core.universe import Universe

__all__ = [
    "DEFAULT_WINDOW_DAYS",
    "BackfillCell",
    "BackfillWindow",
    "plan_backfill",
    "plan_windows",
]

DEFAULT_WINDOW_DAYS = 30
"""Calendar days one fetch window may span.

The plan is drawn one session at a time because completeness is a per-session
fact, but that is not the unit a history feed sells. IBKR answers a 30-day
request for one-minute bars with every session in it — about 22 — for the
same single request against the same pacing budget. Filling 58 symbols over
five years costs ~73,000 requests one session at a time and ~3,500 in 30-day
windows: nine days of crawling against eleven hours.

Thirty because that is the largest window IBKR serves at one-minute bars,
measured against a live Gateway. Its own duration table says a one-minute
request may cover one day; the table is wrong, and the observed behaviour is
what is true. A feed with a tighter limit gets a smaller number passed in
rather than a chunking rule of its own.
"""


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


@dataclass(frozen=True, slots=True, kw_only=True)
class BackfillWindow:
    """One request: a run of a single instrument's missing sessions, fetched together.

    The plan is drawn per session because completeness is a per-session fact —
    bars held against `TradingSession.expected_bars`. Fetching is a different
    unit: a history feed charges one request for a span, whatever it holds. A
    window is the bridge, and it deliberately keeps the cells rather than
    flattening them into a date range, so the pass still reports per session and
    still writes each session under its own date.

    **A window is not a contiguous date range.** It holds only the sessions that
    are *missing*, which is why the first and last cell bound it and the ones
    between may be absent. The request covers the whole span anyway; anything
    the feed returns for a session already complete is dropped on the way in,
    because a complete session is not in the plan and so has no cell to hold it.

    Example:
        >>> from datetime import UTC, datetime
        >>> from neurotrade.core.clock import to_nanos
        >>> def session(day):
        ...     return TradingSession(
        ...         venue=Venue.NASDAQ,
        ...         session_date=day,
        ...         open_ns=to_nanos(datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC)),
        ...         close_ns=to_nanos(datetime(day.year, day.month, day.day, 20, 0, tzinfo=UTC)),
        ...         is_early_close=False,
        ...     )
        >>> aapl = Symbol("AAPL", Venue.NASDAQ)
        >>> cells = tuple(
        ...     BackfillCell(
        ...         symbol=aapl, session=session(day), interval=BarInterval.MIN_1, held_bars=0
        ...     )
        ...     for day in (date(2024, 7, 1), date(2024, 7, 2))
        ... )
        >>> window = BackfillWindow(symbol=aapl, interval=BarInterval.MIN_1, cells=cells)
        >>> (str(window), window.span_days, window.has_untouched)
        ('AAPL.NASDAQ 2024-07-01..2024-07-02 1m 2 sessions', 2, True)
    """

    symbol: Symbol  # instrument this one request is for
    interval: BarInterval  # bar size being filled
    cells: tuple[BackfillCell, ...]  # the missing sessions it covers, ascending by date

    def __post_init__(self) -> None:
        """Reject a window that cannot be fetched as one request.

        Raises:
            ValueError: If it holds no cells, if any cell belongs to another
                instrument or bar size, or if the cells are not in ascending
                date order. Order is load-bearing rather than cosmetic: the
                bars that come back are split across the cells by walking both
                in step, and an unordered window would file bars under the
                wrong session.
        """
        if not self.cells:
            raise ValueError(f"a {self.symbol} window must cover at least one session")
        for cell in self.cells:
            if cell.symbol != self.symbol or cell.interval is not self.interval:
                raise ValueError(
                    f"{cell} does not belong in a {self.symbol} {self.interval.value} window"
                )
        dates = [cell.session_date for cell in self.cells]
        if dates != sorted(set(dates)):
            raise ValueError(
                f"{self.symbol} window sessions must ascend without repeats, got {dates}"
            )

    @property
    def first_date(self) -> date:
        """Oldest session in the window."""
        return self.cells[0].session_date

    @property
    def last_date(self) -> date:
        """Newest session in the window."""
        return self.cells[-1].session_date

    @property
    def span_days(self) -> int:
        """Calendar days the request covers, inclusive of both ends.

        Calendar rather than trading days because that is the unit a feed's
        duration limit is expressed in.
        """
        return (self.last_date - self.first_date).days + 1

    @property
    def start_ns(self) -> Nanos:
        """Inclusive lower bound for the fetch, the oldest session's open."""
        return self.cells[0].start_ns

    @property
    def end_ns(self) -> Nanos:
        """Inclusive upper bound for the fetch, the newest session's close."""
        return self.cells[-1].end_ns

    @property
    def has_untouched(self) -> bool:
        """Whether any session in the window has nothing at all on disk.

        A window of only short sessions is the one whose request mostly
        confirms what is already there — see `crawler.order_windows`.
        """
        return any(cell.is_untouched for cell in self.cells)

    def __str__(self) -> str:
        span = f"{self.first_date}..{self.last_date}"
        count = len(self.cells)
        plural = "session" if count == 1 else "sessions"
        return f"{self.symbol} {span} {self.interval.value} {count} {plural}"


def plan_windows(
    cells: Iterable[BackfillCell],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> Iterator[BackfillWindow]:
    """Group a plan's cells into the fewest requests that still fetch it.

    **The date axis is chunked first, the universe second.** Dates are cut into
    spans of at most `window_days`, newest first; every symbol's window inside a
    span is offered before any symbol's window in an older one. That is
    `plan_backfill`'s recency-major order at a coarser grain, and keeping it is
    the point: grouping symbol-major would fetch five years of AAPL before
    touching MSFT, and a cross-sectional study cannot use that corpus.

    The cells are consumed eagerly — grouping by symbol needs all of them — so
    this is a generator only in how it hands them back.

    Args:
        cells: The plan, in any order. `plan_backfill`'s order is preserved
            where it is meaningful: dates descending, symbols in the order they
            first appear.
        window_days: Most calendar days one window may span, inclusive of both
            ends. One reproduces the old request-per-session behaviour.

    Yields:
        Windows, newest span first, symbols within a span in first-seen order.

    Raises:
        ValueError: If `window_days` is less than one, or if the cells name
            more than one bar size — a window is one request, and one request
            asks for one bar size.

    Example:
        >>> from datetime import UTC, datetime
        >>> from neurotrade.core.clock import to_nanos
        >>> def cell(ticker, day):
        ...     session = TradingSession(
        ...         venue=Venue.NASDAQ,
        ...         session_date=day,
        ...         open_ns=to_nanos(datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC)),
        ...         close_ns=to_nanos(datetime(day.year, day.month, day.day, 20, 0, tzinfo=UTC)),
        ...         is_early_close=False,
        ...     )
        ...     return BackfillCell(
        ...         symbol=Symbol(ticker, Venue.NASDAQ),
        ...         session=session,
        ...         interval=BarInterval.MIN_1,
        ...         held_bars=0,
        ...     )
        >>> plan = [
        ...     cell("AAPL", date(2024, 7, 2)),
        ...     cell("MSFT", date(2024, 7, 2)),
        ...     cell("AAPL", date(2024, 7, 1)),
        ...     cell("MSFT", date(2024, 7, 1)),
        ... ]
        >>> for window in plan_windows(plan):
        ...     print(window)
        AAPL.NASDAQ 2024-07-01..2024-07-02 1m 2 sessions
        MSFT.NASDAQ 2024-07-01..2024-07-02 1m 2 sessions
        >>> for window in plan_windows(plan, window_days=1):
        ...     print(window)
        AAPL.NASDAQ 2024-07-02..2024-07-02 1m 1 session
        MSFT.NASDAQ 2024-07-02..2024-07-02 1m 1 session
        AAPL.NASDAQ 2024-07-01..2024-07-01 1m 1 session
        MSFT.NASDAQ 2024-07-01..2024-07-01 1m 1 session
    """
    if window_days < 1:
        raise ValueError(f"window_days must be at least 1, got {window_days}")

    # dict rather than set for both: insertion order is the plan's order, and
    # set iteration order is not reproducible — which symbol's window a crawl
    # reaches before it is killed decides what a half-filled corpus contains.
    by_symbol: dict[Symbol, dict[date, BackfillCell]] = {}
    days: dict[date, None] = {}
    interval: BarInterval | None = None
    for cell in cells:
        if interval is None:
            interval = cell.interval
        elif cell.interval is not interval:
            raise ValueError(f"one plan, one bar size: {cell} does not match {interval.value}")
        by_symbol.setdefault(cell.symbol, {})[cell.session_date] = cell
        days[cell.session_date] = None

    if interval is None:
        return

    # Descending, so a span is anchored on its newest date and grows backwards.
    # Anchoring forwards would put the partial span at the recent end, which is
    # the end a crawl that is killed early most needs whole.
    spans: list[list[date]] = []
    for day in sorted(days, reverse=True):
        if spans and (spans[-1][0] - day).days < window_days:
            spans[-1].append(day)
        else:
            spans.append([day])

    for span in spans:
        for symbol, held in by_symbol.items():
            covered = tuple(held[day] for day in reversed(span) if day in held)
            if covered:
                yield BackfillWindow(symbol=symbol, interval=interval, cells=covered)
