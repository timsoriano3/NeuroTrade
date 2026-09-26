"""Daily bars from Yahoo Finance, through the `yfinance` package (§12.1 stage 3).

Satisfies `MarketDataPort` structurally, so daily bars reach the corpus through
the same crawler the IBKR backfill and the vendor samples use — one ingestion
path, not a third one (§3.6).

Three things about this feed differ from the intraday ones and are deliberate:

**One request per symbol, not per session.** Yahoo answers a multi-year daily
history in a single call, while the crawler offers work one instrument-session
at a time. The whole configured range is downloaded on the first cell for a
symbol and cached; every later cell for that symbol is served from memory. A
literal request per cell would be a thousand HTTP calls per symbol and would be
rate-limited into uselessness.

**Bars are stamped at the venue's session close.** Yahoo indexes a daily row at
local midnight — the *start* of the day — and `Bar.ts_event` is a bar's close
everywhere in this system. Keeping Yahoo's stamp would say AAPL's whole
2023-09-29 range was known at 00:00 that morning, which is lookahead of the
purest kind. The close comes from `CalendarPort`, so early closes need no
special case.

**Prices are unadjusted.** `auto_adjust=False`, so the OHLC written is what
traded on the day, and `Adj Close` is discarded rather than stored. Raw data is
immutable and adjustment is recomputable from corporate actions (§12.1 stage 5);
storing a back-adjusted price would bake today's split history into a bar and
make the corpus disagree with itself after the next split.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Final, Protocol

from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.core.clock import Clock, Nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.ports import CalendarPort
from neurotrade.core.types import Price, Quantity, Symbol, Venue

__all__ = [
    "DailyDownloader",
    "DailyRow",
    "YFinanceDailyFeed",
    "YahooDownloader",
    "yahoo_ticker",
]

_TIMEOUT_SECONDS: Final = 30.0
"""Per-request ceiling. Yahoo is usually sub-second; a symbol that hangs
longer than this is a stall, and a crawl of two thousand of them cannot afford
to wait on each."""

_PRICE_DECIMALS: Final = 4
"""Decimal places Yahoo's prices are rounded to before becoming `Price`.

Yahoo answers JSON float64, so a $169.34 close arrives as
`169.33999633789062`. `Price.from_float` would store that — rounded to the
corpus scale, `169.33999633` — and the corpus would accept it.

Four places is the finest tick a North American equity quotes in: a penny above
$1.00 and $0.0001 below it. Rounding there recovers the price that actually
traded instead of preserving the float's noise, which matters because a bar
whose close reads `169.33999633` will never compare equal to IBKR's `169.34`
when the two sources are cross-checked."""

_VOLUME_DECIMALS: Final = 0
"""Shares are whole. Yahoo sends them as a float anyway."""

_YAHOO_SUFFIX: Final[Mapping[Venue, str]] = {
    Venue.TSX: ".TO",
    Venue.TSXV: ".V",
}
"""Yahoo's exchange suffixes. US listings carry none — a bare `AAPL` is the
NASDAQ line — which is also why Yahoo cannot distinguish TD on NYSE from TD on
the TSX without one, and why `Symbol` keeps the venue in the first place."""


@dataclass(frozen=True, slots=True)
class DailyRow:
    """One daily row as Yahoo gave it, before any calendar or type conversion.

    The seam between the network and the domain: tests build these by hand, so
    every conversion below is exercised without reaching Yahoo.
    """

    session_date: date  # the row's date in the VENUE's timezone, not UTC
    open: float  # unadjusted
    high: float  # unadjusted
    low: float  # unadjusted
    close: float  # unadjusted; Yahoo's `Adj Close` is deliberately dropped
    volume: float  # shares; Yahoo gives it as a float even though it is whole


class DailyDownloader(Protocol):
    """How the feed reaches Yahoo. Injected so tests never do.

    Conformance is structural, so a plain function of the right shape is a
    downloader:

    Example:
        >>> def fixed(ticker, *, start, end):
        ...     return (DailyRow(start, 1.0, 1.0, 1.0, 1.0, 100.0),)
        >>> fixed("AAPL", start=date(2024, 7, 1), end=date(2024, 7, 1))[0].close
        1.0
    """

    def __call__(self, ticker: str, *, start: date, end: date) -> Sequence[DailyRow]:
        """Fetch one Yahoo symbol's daily rows.

        Args:
            ticker: Yahoo's own symbol, suffix included — `SHOP.TO`, not `SHOP`.
            start: First date to ask for, inclusive.
            end: Last date to ask for, **inclusive** — unlike Yahoo's own API,
                whose `end` is exclusive. The exclusive bound is an easy
                off-by-one to write and a hard one to notice, since it silently
                drops the most recent session.

        Returns:
            Rows in ascending date order.

        Raises:
            FeedError: If Yahoo refuses or answers with something unparseable.
        """
        ...


def yahoo_ticker(symbol: Symbol) -> str:
    """Yahoo's name for an instrument.

    Args:
        symbol: The instrument, with its listing venue.

    Returns:
        The ticker Yahoo knows it by. Canadian listings take an exchange
        suffix; US ones are bare. A dot inside a ticker becomes a hyphen, which
        is Yahoo's convention for share classes and preferreds — `BRK.B` is
        `BRK-B` there, and `BCE.PR.A` on the TSX is `BCE-PR-A.TO`.

    Example:
        >>> yahoo_ticker(Symbol("AAPL", Venue.NASDAQ))
        'AAPL'
        >>> yahoo_ticker(Symbol("SHOP", Venue.TSX))
        'SHOP.TO'
        >>> yahoo_ticker(Symbol("BRK.B", Venue.NYSE))
        'BRK-B'
    """
    return symbol.ticker.replace(".", "-") + _YAHOO_SUFFIX.get(symbol.venue, "")


class YahooDownloader:
    """The real downloader: `yfinance`, one `Ticker.history` call per symbol.

    Imports `yfinance` lazily. The package pulls in pandas, numpy and a
    networking stack, and nothing else in the system needs it — paying that
    import cost on every `neurotrade` invocation, including `--help`, would be
    a second or more for a command that will never touch Yahoo.

    Example:
        >>> YahooDownloader(timeout=5.0).timeout
        5.0
    """

    __slots__ = ("timeout",)

    def __init__(self, timeout: float = _TIMEOUT_SECONDS) -> None:
        """Create the downloader.

        Args:
            timeout: Seconds to wait on one Yahoo request.
        """
        self.timeout = timeout

    def __call__(self, ticker: str, *, start: date, end: date) -> Sequence[DailyRow]:
        """Fetch one symbol's unadjusted daily history.

        Args:
            ticker: Yahoo's symbol, suffix included.
            start: First date, inclusive.
            end: Last date, inclusive.

        Returns:
            Rows in ascending date order. Empty when Yahoo knows the ticker but
            has no rows in the range — a young listing, or a range before it
            came to market.

        Raises:
            FeedError: If Yahoo refuses the request, or gives a row this cannot
                read.
        """
        import yfinance

        # yfinance hides exceptions by default and answers an empty frame
        # instead. An empty frame is indistinguishable from "this symbol has no
        # data here", so a 404, a rate limit and a genuine gap would all enter
        # the corpus as silence. This setting is global to the package, which
        # is why it is set here rather than asked of the caller.
        yfinance.config.debug.hide_exceptions = False

        # yfinance traces every step of every request at DEBUG — cookie
        # fetches, crumb lookups, the SQL of its own cache. Under the research
        # profile, which runs at DEBUG, a crawl of two thousand symbols would
        # bury its own progress in vendor tracing. Raise it back deliberately
        # when debugging this adapter.
        logging.getLogger("yfinance").setLevel(logging.WARNING)

        try:
            frame = yfinance.Ticker(ticker).history(
                start=start,
                # Yahoo's end is EXCLUSIVE; ours is inclusive. Without this the
                # last session of every crawl is quietly missing.
                end=end + timedelta(days=1),
                interval=BarInterval.DAY_1.value,
                auto_adjust=False,
                actions=False,
                timeout=self.timeout,
            )
        except Exception as error:
            raise FeedError(f"yfinance failed for {ticker}: {error}") from error

        rows: list[DailyRow] = []
        for stamp, open_, high, low, close, volume in zip(
            frame.index,
            frame["Open"],
            frame["High"],
            frame["Low"],
            frame["Close"],
            frame["Volume"],
            strict=True,
        ):
            values = (float(open_), float(high), float(low), float(close), float(volume))
            if not all(math.isfinite(value) for value in values):
                # Yahoo drops all-NaN rows itself, so one arriving here means a
                # partially written row. Guessing which half is right is how a
                # corrupt bar enters a corpus that is never re-downloaded.
                raise FeedError(f"{ticker}: non-finite values on {stamp.date()}: {values}")
            rows.append(
                DailyRow(
                    # The index is tz-aware in the EXCHANGE's timezone, so this
                    # is already the venue's session date. `.astimezone(UTC)`
                    # here would move a Toronto row onto the next day.
                    session_date=stamp.date(),
                    open=values[0],
                    high=values[1],
                    low=values[2],
                    close=values[3],
                    volume=values[4],
                )
            )
        rows.sort(key=lambda row: row.session_date)
        return tuple(rows)

    def __repr__(self) -> str:
        return f"YahooDownloader(timeout={self.timeout})"


class YFinanceDailyFeed:
    """A `MarketDataPort` serving unadjusted daily bars from Yahoo.

    One download per symbol covers the whole configured range; the crawler's
    per-session cells are then served from the cache.

    Example:
        >>> from neurotrade.adapters.calendar.venue_calendar import VenueCalendar
        >>> from neurotrade.core.clock import SimClock
        >>> from neurotrade.core.ports import MarketDataPort
        >>> feed = YFinanceDailyFeed(
        ...     VenueCalendar(),
        ...     SimClock(0),
        ...     start=date(2024, 7, 1),
        ...     end=date(2024, 7, 5),
        ...     downloader=lambda ticker, *, start, end: (),
        ... )
        >>> isinstance(feed, MarketDataPort)
        True
    """

    __slots__ = ("_cache", "_calendar", "_clock", "_downloader", "_end", "_start", "_unmatched")

    def __init__(
        self,
        calendar: CalendarPort,
        clock: Clock,
        *,
        start: date,
        end: date,
        downloader: DailyDownloader | None = None,
    ) -> None:
        """Create the feed.

        Args:
            calendar: Supplies each session's close, which is where a daily bar
                is stamped.
            clock: Stamps `ts_init` on every bar returned.
            start: First session date the feed will be asked about, inclusive.
                One download per symbol covers `start` to `end`, so a range
                wider than the crawl's wastes bandwidth and a narrower one
                starves it.
            end: Last session date, inclusive.
            downloader: How to reach Yahoo. Defaults to `YahooDownloader()`;
                tests inject a fake.

        Raises:
            ValueError: If the range runs backwards.
        """
        if end < start:
            raise ValueError(f"range runs backwards: {start} to {end}")
        self._calendar = calendar
        self._clock = clock
        self._start = start
        self._end = end
        self._downloader = downloader if downloader is not None else YahooDownloader()
        self._cache: dict[Symbol, tuple[Bar, ...]] = {}
        self._unmatched: dict[Symbol, tuple[date, ...]] = {}

    async def is_connected(self) -> bool:
        """Always true: Yahoo holds no session, so there is nothing to lose.

        A failed request raises from `fetch_bars` instead, which is where the
        crawler already handles a source that will not answer.
        """
        return True

    async def fetch_bars(
        self,
        symbol: Symbol,
        interval: BarInterval,
        start: Nanos,
        end: Nanos,
    ) -> Sequence[Bar]:
        """Daily bars for one symbol over a half-open range.

        Args:
            symbol: Instrument to fetch.
            interval: Must be `BarInterval.DAY_1`. Yahoo serves intraday too,
                but only over a short trailing window and with partial volume,
                so this feed refuses rather than seeding the corpus with bars
                that disagree with IBKR's.
            start: Inclusive lower bound on `ts_event` (the session close).
            end: Exclusive upper bound on `ts_event`.

        Returns:
            Bars in ascending `ts_event` order, unadjusted.

        Raises:
            FeedError: If `interval` is not daily, or Yahoo will not answer.

        Example:
            >>> import asyncio
            >>> from neurotrade.adapters.calendar.venue_calendar import VenueCalendar
            >>> from neurotrade.core.clock import SimClock
            >>> feed = YFinanceDailyFeed(
            ...     VenueCalendar(), SimClock(0), start=date(2024, 7, 1), end=date(2024, 7, 5)
            ... )
            >>> asyncio.run(
            ...     feed.fetch_bars(Symbol("AAPL", Venue.NASDAQ), BarInterval.MIN_1, 0, 1)
            ... )
            Traceback (most recent call last):
                ...
            neurotrade.adapters.feeds.errors.FeedError: yfinance daily feed only has 1d bars, not 1m
        """
        if interval is not BarInterval.DAY_1:
            raise FeedError(
                f"yfinance daily feed only has {BarInterval.DAY_1.value} bars, not {interval.value}"
            )

        bars = self._cache.get(symbol)
        if bars is None:
            bars = await self._download(symbol)
            self._cache[symbol] = bars

        received_at = self._clock.now_ns()
        return tuple(
            # ts_init belongs to this call, not to the download — the bar was
            # cached, but the caller's clock has moved on since.
            _restamped(bar, received_at)
            for bar in bars
            if start <= bar.ts_event < end
        )

    def unmatched(self) -> Mapping[Symbol, tuple[date, ...]]:
        """Dates Yahoo returned that the venue calendar denies, per symbol.

        Never empty for free: either Yahoo invented a row or our calendar has
        the venue's holidays wrong, and both are worth a human look. The rows
        are dropped rather than stamped, because a bar with no session has no
        close to be stamped at.

        Returns:
            A mapping of symbol to the offending dates, ascending. Only symbols
            that had one appear.
        """
        return dict(self._unmatched)

    async def _download(self, symbol: Symbol) -> tuple[Bar, ...]:
        """Fetch and convert one symbol's whole range.

        Runs the (blocking) HTTP call off the event loop, so a slow symbol does
        not stall anything else the crawler is doing.
        """
        rows = await asyncio.to_thread(
            self._downloader, yahoo_ticker(symbol), start=self._start, end=self._end
        )

        bars: list[Bar] = []
        unmatched: list[date] = []
        for row in rows:
            session = self._calendar.session(symbol.venue, row.session_date)
            if session is None:
                unmatched.append(row.session_date)
                continue
            bars.append(
                Bar(
                    symbol=symbol,
                    ts_event=session.close_ns,
                    ts_init=0,  # replaced per call in fetch_bars
                    interval=BarInterval.DAY_1,
                    open=Price.from_float(round(row.open, _PRICE_DECIMALS)),
                    high=Price.from_float(round(row.high, _PRICE_DECIMALS)),
                    low=Price.from_float(round(row.low, _PRICE_DECIMALS)),
                    close=Price.from_float(round(row.close, _PRICE_DECIMALS)),
                    volume=Quantity.from_float(round(row.volume, _VOLUME_DECIMALS)),
                )
            )
        if unmatched:
            self._unmatched[symbol] = tuple(sorted(unmatched))
        bars.sort(key=lambda bar: bar.ts_event)
        return tuple(bars)

    def __repr__(self) -> str:
        return f"YFinanceDailyFeed(start={self._start}, end={self._end}, cached={len(self._cache)})"


def _restamped(bar: Bar, ts_init: Nanos) -> Bar:
    """Copy a cached bar with the current arrival time."""
    return replace(bar, ts_init=ts_init)
