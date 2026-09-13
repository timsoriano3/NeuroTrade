"""The backfill crawler's fetch loop: drain the work queue into the corpus.

§12.1 stage 1 calls for a crawler that is "pacing-aware, resumable, runs
continuously". `plan_backfill` is the resumable half. This module is the part
that fetches: it takes the plan, asks the feed for each instrument-session, and
hands what comes back to the store.

**One call to `crawl` is one pass.** A pass plans once, drains that plan, and
returns a report. Running continuously is a loop of passes owned by the caller,
because the plan is a snapshot and a week-long crawl should re-read the corpus
rather than trust a queue computed days ago.

**Pacing belongs to the feed.** `MarketDataPort` requires implementations to
pace themselves, and the IBKR adapter does — it waits on its own
`HistoricalPacer` before every request. A second limiter here would count the
same requests twice and halve throughput for nothing.

**No clock, no state.** Nothing here reads the time or remembers what it did.
Progress is whatever the store now holds, which the next pass's plan observes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from neurotrade.core.events import BarInterval
from neurotrade.core.ports import CalendarPort, CatalogPort, MarketDataPort, StoragePort
from neurotrade.core.types import Symbol
from neurotrade.core.universe import Universe
from neurotrade.ingest.backfill import BackfillCell, plan_backfill

__all__ = ["CellOutcome", "CellStatus", "CrawlReport", "crawl", "order_cells"]

DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
"""Failures in a row before a pass gives up.

One symbol failing is ordinary — the seed universe's venues are unverified, and
the first crawl is what confirms them. Five different symbols failing back to
back is not a bad listing; it is a dead Gateway, and every further attempt
would spend a paced request to learn the same thing."""


class CellStatus(StrEnum):
    """What happened to one cell during a pass.

    Example:
        >>> CellStatus.FILLED.value
        'filled'
    """

    FILLED = "filled"  # the feed returned bars and they were written
    EMPTY = "empty"  # the feed answered with nothing — a halt, or not yet listed
    FAILED = "failed"  # the feed raised; nothing was written
    SKIPPED = "skipped"  # not requested, because this symbol already failed this pass


@dataclass(frozen=True, slots=True, kw_only=True)
class CellOutcome:
    """The result of offering one cell to the feed.

    Example:
        >>> from neurotrade.core.types import Venue
        >>> outcome = CellOutcome(
        ...     symbol=Symbol("AAPL", Venue.NASDAQ),
        ...     session_date=date(2024, 7, 2),
        ...     status=CellStatus.EMPTY,
        ...     written=0,
        ... )
        >>> str(outcome)
        'AAPL.NASDAQ 2024-07-02 empty 0'
    """

    symbol: Symbol  # instrument the cell was for
    session_date: date  # trading day, in the venue's terms
    status: CellStatus  # what happened
    written: int  # bars handed to the store; zero unless FILLED
    error: str | None = None  # the feed's message, when FAILED

    def __str__(self) -> str:
        line = f"{self.symbol} {self.session_date} {self.status.value} {self.written}"
        return f"{line} — {self.error}" if self.error else line


@dataclass(frozen=True, slots=True, kw_only=True)
class CrawlReport:
    """Everything one pass did, in the order it did it.

    Example:
        >>> report = CrawlReport(outcomes=(), planned=0)
        >>> (report.requests, report.bars_written, report.completed)
        (0, 0, True)
    """

    outcomes: tuple[CellOutcome, ...]  # one per cell offered, in fetch order
    planned: int  # cells the plan held before any limit was applied
    stopped: str | None = None  # why the pass ended early; None if it ran out of work

    @property
    def completed(self) -> bool:
        """Whether the pass ran to the end of its work rather than giving up."""
        return self.stopped is None

    @property
    def requests(self) -> int:
        """Requests sent to the feed — every outcome except a skip."""
        return sum(1 for outcome in self.outcomes if outcome.status is not CellStatus.SKIPPED)

    @property
    def bars_written(self) -> int:
        """Bars handed to the store across the pass.

        An upper bound on what the corpus gained: the store de-duplicates, so a
        re-fetched short session counts the bars it already had.
        """
        return sum(outcome.written for outcome in self.outcomes)

    def count(self, status: CellStatus) -> int:
        """How many cells ended with `status`.

        Example:
            >>> CrawlReport(outcomes=(), planned=0).count(CellStatus.FAILED)
            0
        """
        return sum(1 for outcome in self.outcomes if outcome.status is status)


def order_cells(cells: Iterable[BackfillCell]) -> list[BackfillCell]:
    """Put never-fetched sessions ahead of short ones, keeping recency within each.

    An untouched session is almost certainly a fetch that has not happened; a
    short one may be a halt the corpus cannot recognise yet (see the known
    limitation in `backfill`), and re-requesting it will likely return the same
    shortfall. Spending the paced budget on the first kind grows the corpus;
    spending it on the second mostly confirms what is already there.

    The sort is stable, so `plan_backfill`'s recency-major order survives inside
    each group.

    Example:
        >>> from datetime import UTC, datetime
        >>> from neurotrade.core.calendar import TradingSession
        >>> from neurotrade.core.clock import to_nanos
        >>> from neurotrade.core.types import Venue
        >>> session = TradingSession(
        ...     venue=Venue.NASDAQ,
        ...     session_date=date(2024, 7, 2),
        ...     open_ns=to_nanos(datetime(2024, 7, 2, 13, 30, tzinfo=UTC)),
        ...     close_ns=to_nanos(datetime(2024, 7, 2, 20, 0, tzinfo=UTC)),
        ...     is_early_close=False,
        ... )
        >>> def cell(ticker, held):
        ...     return BackfillCell(symbol=Symbol(ticker, Venue.NASDAQ), session=session,
        ...                         interval=BarInterval.MIN_1, held_bars=held)
        >>> [c.symbol.ticker for c in order_cells([cell("AAPL", 200), cell("MSFT", 0)])]
        ['MSFT', 'AAPL']
    """
    return sorted(cells, key=lambda cell: not cell.is_untouched)


async def crawl(
    universe: Universe,
    calendar: CalendarPort,
    catalog: CatalogPort,
    feed: MarketDataPort,
    store: StoragePort,
    *,
    start: date,
    end: date,
    source: str,
    interval: BarInterval = BarInterval.MIN_1,
    limit: int | None = None,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    on_outcome: Callable[[CellOutcome], None] | None = None,
) -> CrawlReport:
    """Run one backfill pass: plan, fetch each missing session, store it.

    A symbol whose request fails is skipped for the rest of the pass. The usual
    cause is a listing the feed cannot resolve, which fails identically for
    every session, so retrying it on each date would spend a request per
    session to learn nothing. The next pass tries it again.

    Args:
        universe: Instruments to fill.
        calendar: Which days each venue traded.
        catalog: What the corpus already holds.
        feed: Where bars come from. Expected to pace itself.
        store: Where bars go. Must be idempotent, as `StoragePort` requires.
        start: First session date to consider, inclusive.
        end: Last session date to consider, inclusive.
        source: Provenance recorded on every row, as a storage `Source` value.
            A string rather than the enum because that enum lives in the
            storage adapter, which this layer may not import.
        interval: Bar size to fill.
        limit: Most cells to offer this pass; all of them when None. Skips
            count towards it, so a limit bounds the pass's length, not only
            its requests.
        max_consecutive_failures: Failed requests in a row that end the pass.
        on_outcome: Called after each cell, so a pass that takes a day can be
            watched rather than waited on.

    Returns:
        What the pass did. `stopped` is set if it gave up early.

    Raises:
        ValueError: If `limit` or `max_consecutive_failures` is not positive,
            from `plan_backfill` for a backwards range, or if the feed
            returns a bar for another symbol or interval. Also re-raises
            whatever the store raises: a write that fails is a local fault —
            a full disk, a permission — that every later cell would hit too.

    Example:
        >>> import asyncio
        >>> from datetime import UTC, datetime
        >>> from neurotrade.core.calendar import TradingSession
        >>> from neurotrade.core.clock import to_nanos
        >>> from neurotrade.core.types import Venue
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
        >>> class SilentFeed:
        ...     async def fetch_bars(self, symbol, interval, start, end):
        ...         return ()
        ...     async def is_connected(self):
        ...         return True
        >>> class NullStore:
        ...     def write_bars(self, bars, *, source, session_date): pass
        ...     def read_bars(self, symbol, interval, start, end):
        ...         return iter(())
        >>> report = asyncio.run(crawl(
        ...     Universe([Symbol("AAPL", Venue.NASDAQ)]),
        ...     OneSessionCalendar(), EmptyCatalog(), SilentFeed(), NullStore(),
        ...     start=date(2024, 7, 1), end=date(2024, 7, 5), source="ibkr",
        ... ))
        >>> [str(outcome) for outcome in report.outcomes]
        ['AAPL.NASDAQ 2024-07-02 empty 0']
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if max_consecutive_failures < 1:
        raise ValueError(
            f"max_consecutive_failures must be at least 1, got {max_consecutive_failures}"
        )

    cells = order_cells(
        plan_backfill(universe, calendar, catalog, start=start, end=end, interval=interval)
    )
    planned = len(cells)
    if limit is not None:
        cells = cells[:limit]

    outcomes: list[CellOutcome] = []
    failed_symbols: set[Symbol] = set()
    consecutive_failures = 0
    stopped: str | None = None

    for cell in cells:
        if cell.symbol in failed_symbols:
            outcome = CellOutcome(
                symbol=cell.symbol,
                session_date=cell.session_date,
                status=CellStatus.SKIPPED,
                written=0,
            )
        else:
            outcome = await _fetch_cell(cell, feed, store, source=source)
            if outcome.status is CellStatus.FAILED:
                failed_symbols.add(cell.symbol)
                consecutive_failures += 1
            else:
                consecutive_failures = 0

        outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

        if consecutive_failures >= max_consecutive_failures:
            stopped = f"{consecutive_failures} consecutive failed requests; last: {outcome.error}"
            break

    return CrawlReport(outcomes=tuple(outcomes), planned=planned, stopped=stopped)


async def _fetch_cell(
    cell: BackfillCell,
    feed: MarketDataPort,
    store: StoragePort,
    *,
    source: str,
) -> CellOutcome:
    """Fetch one instrument-session and write it.

    Raises:
        ValueError: If the feed returns a bar for another symbol or interval.
            That is a broken adapter, not a missing listing, and writing it
            would file one instrument's prices under another's name.
    """
    # The port's range is inclusive-start, exclusive-end on `ts_event`, but a
    # session's bars close in (open, close]: a bar closing at the bell belongs
    # to the previous session. Shifting both bounds by a nanosecond asks for
    # exactly this session's bars and nothing either side of it.
    try:
        bars = await feed.fetch_bars(cell.symbol, cell.interval, cell.start_ns + 1, cell.end_ns + 1)
    except Exception as error:
        # The port documents its failures only as adapter-specific, so this
        # layer cannot name a narrower type without importing an adapter.
        # CancelledError is a BaseException and still propagates, so an
        # interrupted crawl stops rather than recording a failure.
        return CellOutcome(
            symbol=cell.symbol,
            session_date=cell.session_date,
            status=CellStatus.FAILED,
            written=0,
            error=f"{type(error).__name__}: {error}",
        )

    for bar in bars:
        if bar.symbol != cell.symbol or bar.interval is not cell.interval:
            raise ValueError(
                f"feed answered a request for {cell} with a {bar.interval.value} bar "
                f"for {bar.symbol}"
            )

    # Trimmed again here rather than trusting every adapter to do it: a bar
    # filed under the wrong session date is invisible to the plan, which counts
    # by date, and would sit in the corpus looking like a complete day.
    in_session = [bar for bar in bars if cell.session.holds_bar(bar.ts_event)]
    if not in_session:
        return CellOutcome(
            symbol=cell.symbol,
            session_date=cell.session_date,
            status=CellStatus.EMPTY,
            written=0,
        )

    store.write_bars(in_session, source=source, session_date=cell.session_date)
    return CellOutcome(
        symbol=cell.symbol,
        session_date=cell.session_date,
        status=CellStatus.FILLED,
        written=len(in_session),
    )
